"""Single-device training entry using official Gemino architecture and loss modules."""
import argparse, json, math, random, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from core import *
from first_order_model.modules.model import GeneratorFullModel, DiscriminatorFullModel

def initialize_worker(worker):
    seed=torch.initial_seed()%2**32;random.seed(seed);np.random.seed(seed)

def train(args):
    run_started=time.perf_counter()
    cfg=read_config(args.config);assert_ready(cfg);seed_all(cfg['seed']);torch.set_num_threads(args.cpu_threads)
    device=torch.device(args.device);out=resolve(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    g,k,d=models(cfg,device)
    t=cfg['train_params']
    opt=[torch.optim.Adam(module.parameters(),lr=t[name],betas=(.5,.999)) for module,name in
         [(g,'lr_generator'),(k,'lr_kp_detector'),(d,'lr_discriminator')]]
    sched=[torch.optim.lr_scheduler.MultiStepLR(o,t['epoch_milestones'],gamma=.1) for o in opt]
    start_epoch=0;step=0;start_batch=-1
    if args.resume:
        state=torch.load(resolve(args.resume),map_location='cpu',weights_only=False)
        if state['config']['model_params']!=cfg['model_params']:raise ValueError('Resume model config mismatch')
        for module,key in [(g,'generator'),(k,'kp_detector'),(d,'discriminator')]:module.load_state_dict(state[key],strict=True)
        for o,s in zip(opt,state['optimizers']):o.load_state_dict(s)
        for s,value in zip(sched,state['schedulers']):s.load_state_dict(value)
        start_epoch=state['epoch'];step=state['global_step'];start_batch=state['batch_index']
        torch.set_rng_state(state['torch_rng']);np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
        if state['cuda_rng'] and device.type=='cuda':torch.cuda.set_rng_state_all(state['cuda_rng'])
        # Workers/augmentation replay is not bit-exact mid-epoch; report this explicitly.
    else:initialize_fomm(g,k,resolve(cfg['initial_checkpoint']),out/'weight_match.json')
    snapshot_steps={int(s) for s in args.snapshot_steps.split(',') if s.strip()}
    def snapshot():
        folder=out/'snapshots';folder.mkdir(exist_ok=True)
        save_atomic(dict(generator=g.state_dict(),kp_detector=k.state_dict(),config=cfg,global_step=step,
                         note='Inference-only trial snapshot; optimizer is in last.pt'),folder/f'step_{step:06d}.pt')
    if step in snapshot_steps:snapshot()
    if args.max_total_steps is not None and step>=args.max_total_steps:
        print('Requested total-step cap already reached:',step,flush=True);return
    gf=GeneratorFullModel(k,g,d,t).to(device)
    df=DiscriminatorFullModel(k,g,d,t).to(device)
    dataset=Pairs(cfg,'train')
    batch_size=args.batch_size or t['batch_size']
    if len(dataset)<batch_size:raise ValueError('Not enough training samples')
    all_times=[];updated_steps=0
    log=(out/'metrics.jsonl').open('a',encoding='utf-8')
    try:
        for epoch in range(start_epoch,t['num_epochs']):
            order=torch.Generator().manual_seed(cfg['seed']+epoch)
            loader=DataLoader(dataset,batch_size=batch_size,shuffle=True,generator=order,num_workers=args.workers,
                              worker_init_fn=initialize_worker,drop_last=True,pin_memory=device.type=='cuda')
            g.train();k.train();d.train()
            iteration_boundary=time.perf_counter()
            for batch_index,batch in enumerate(loader):
                if epoch==start_epoch and batch_index<=start_batch:continue
                start=time.perf_counter();batch={n:v.to(device) for n,v in batch.items()}
                if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
                reset_cache(g)
                for o in opt:o.zero_grad(set_to_none=True)
                # D participates in G loss, but its parameters are updated only by D loss.
                for p in d.parameters():p.requires_grad_(False)
                losses,generated=gf(batch)
                total=sum(v.mean() for v in losses.values())
                if not torch.isfinite(total):raise FloatingPointError('Nonfinite generator loss')
                total.backward()
                for module in [g,k]:
                    for p in module.parameters():
                        if p.grad is not None and not torch.isfinite(p.grad).all():raise FloatingPointError('Nonfinite gradient')
                opt[0].step();opt[1].step()
                for p in d.parameters():p.requires_grad_(True)
                disc_losses={}
                if t['loss_weights']['discriminator_gan']:
                    disc_losses=df(batch,generated);disc_total=sum(v.mean() for v in disc_losses.values())
                    if not torch.isfinite(disc_total):raise FloatingPointError('Nonfinite discriminator loss')
                    disc_total.backward()
                    for p in d.parameters():
                        if p.grad is not None and not torch.isfinite(p.grad).all():raise FloatingPointError('Nonfinite D gradient')
                    opt[2].step()
                if device.type=='cuda':torch.cuda.synchronize()
                elapsed=time.perf_counter()-start;all_times.append(elapsed);step+=1;updated_steps+=1
                row=dict(epoch=epoch,step=step,batch=batch_index,seconds=elapsed,
                         iteration_wall_seconds=time.perf_counter()-iteration_boundary,
                         wall_since_start=time.perf_counter()-run_started,
                         losses={n:float(v.detach().mean()) for n,v in {**losses,**disc_losses}.items()},
                         peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None)
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                # Drop graphs and cached features before next iteration/checkpoint.
                del generated,total,losses,disc_losses,batch;reset_cache(g)
                if t['loss_weights']['discriminator_gan']:del disc_total
                stop=(args.max_steps is not None and updated_steps>=args.max_steps) or \
                     (args.max_total_steps is not None and step>=args.max_total_steps) or \
                     (args.max_minutes is not None and time.perf_counter()-run_started>=args.max_minutes*60)
                if step in snapshot_steps:snapshot()
                if step%args.save_every==0 or stop:
                    save_atomic(checkpoint_state(g,k,d,opt,sched,cfg,epoch,step,batch_index),out/'last.pt')
                if stop:break
                iteration_boundary=time.perf_counter()
            else:
                for scheduler in sched:scheduler.step()
                save_atomic(checkpoint_state(g,k,d,opt,sched,cfg,epoch+1,step,-1),out/'last.pt')
                # Fixed reference/target pairs from held-out source videos; cheap validation between epochs.
                g.eval();k.eval();values=[]
                with torch.no_grad():
                    validation=Pairs(cfg,'val')
                    for i in range(min(4,len(validation))):
                        b={n:v[None].to(device) for n,v in validation[i].items()};reset_cache(g)
                        p=g(b['source'],kp_source=k(b['source']),kp_driving=k(b['driving_lr']),
                            driving_lr=b['driving_lr'],update_source=True)['prediction']
                        values.append(float(F.l1_loss(p,b['driving'])))
                        del b,p
                log.write(json.dumps(dict(epoch=epoch,validation_l1=float(np.mean(values))))+'\n');log.flush()
                reset_cache(g);continue
            break
    finally:log.close()
    summary=dict(updated_steps=updated_steps,total_steps=step,device=str(device),torch=torch.__version__,
                 purpose=cfg['purpose'],median_step_seconds=float(np.median(all_times)) if all_times else None,
                 excludes_data_loading=True,resume_bit_exact=False,checkpoint=str(out/'last.pt'))
    summary['wall_seconds_including_startup_data_and_saving']=time.perf_counter()-run_started
    (out/'summary.json').write_text(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/pilot.yaml');p.add_argument('--output',default='runs/pilot')
    p.add_argument('--device',default='cuda');p.add_argument('--workers',type=int,default=4);p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--batch-size',type=int);p.add_argument('--max-steps',type=int);p.add_argument('--save-every',type=int,default=100)
    p.add_argument('--max-total-steps',type=int,help='Absolute cap including steps restored from checkpoint')
    p.add_argument('--max-minutes',type=float,help='Stop and checkpoint after this wall time, checked after each update')
    p.add_argument('--snapshot-steps',default='',help='Comma-separated global steps for lightweight inference snapshots')
    p.add_argument('--resume');train(p.parse_args())
