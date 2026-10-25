"""Additional held-out clips and pre-encode face-ROI checks for the low-head failure."""
import argparse,hashlib,json,time,sys
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
from core import *
from evaluate import frames_for_record

ROI=(256,32,864,736) # Fixed, predeclared rectangle containing both reference and lowered face.

def metric_values(pred,target,lp):
    p=pred[0].detach().cpu().permute(1,2,0).numpy();t=target[0].detach().cpu().permute(1,2,0).numpy()
    with torch.no_grad():distance=float(lp(pred*2-1,target*2-1))
    return {'psnr':float(peak_signal_noise_ratio(t,p,data_range=1)),
            'ssim':float(structural_similarity(t,p,data_range=1,channel_axis=2)),
            'lpips':distance,'mae':float(np.abs(p-t).mean())}

def main(a):
    import lpips
    torch.set_num_threads(4)
    state=torch.load(resolve(a.checkpoint),map_location='cpu',weights_only=False);cfg=state['config'];stage=state['global_step']
    device=torch.device(a.device);g,k,d=models(cfg,device);del d
    g.load_state_dict(state['generator'],strict=True);k.load_state_dict(state['kp_detector'],strict=True);del state
    g.eval();k.eval();lp=lpips.LPIPS(net='vgg').to(device).eval()
    out=resolve(a.output);out.mkdir(parents=True,exist_ok=True)
    rows=json.loads(resolve(cfg['data_manifest']).read_text());val=[r for r in rows if r['split']=='val']
    groups={}
    for row in val:groups.setdefault(row['source_id'],[]).append(row)
    # Two original validation sources, three additional 5s clips per source. No test data.
    cases=[]
    for identity,items in groups.items():
        for index in [3,6,9]:
            r=items[index];cases.append(('additional',r,set(range(0,150,5)),{15,75,135}))
    original=next(r for r in val if r['source_id']=='zw2UJpfIBaI')
    cases.insert(0,('low_head',original,{105,110,115,120,125,130,135,140,145,149},{105,120,135,149}))
    reports=[]
    for kind,record,indices,images in cases:
        codec=Codec(**cfg['codec']);reference=None;values=[];hashes=hashlib.sha256()
        identifier=f"{kind}_{record['source_id']}_{record['virtual_start_seconds']}"
        for index,rgb in enumerate(frames_for_record(record,cfg['output_size'],150)):
            low=codec.frame(rgb);hashes.update(rgb.tobytes());hashes.update(low.tobytes())
            if reference is None:
                reference=tensor(rgb)[None].to(device);reset_cache(g)
                with torch.no_grad():source_kp=k(reference)
            # Forward every frame exactly as in the primary video evaluation; only metric sampling is sparse.
            low_tensor=tensor(low)[None].to(device);target=tensor(rgb)[None].to(device)
            with torch.no_grad():
                pred=g(reference,kp_source=source_kp,kp_driving=k(low_tensor),driving_lr=low_tensor,update_source=False)['prediction'].clamp(0,1)
                bicubic=F.interpolate(low_tensor,size=target.shape[-2:],mode='bicubic',align_corners=False).clamp(0,1)
            x0,y0,x1,y1=ROI
            if index in indices:
                entry={'frame':index,'seconds':index/30}
                for name,image in [('gemino',pred),('bicubic',bicubic)]:
                    roi=metric_values(image[:,:,y0:y1,x0:x1],target[:,:,y0:y1,x0:x1],lp)
                    entry.update({name+'_face_'+key:value for key,value in roi.items()})
                    if kind=='additional':
                        full=metric_values(image,target,lp);entry.update({name+'_'+key:value for key,value in full.items()})
                values.append(entry)
            if index in images:
                parts=[]
                for label,image in [('Original',target),('Bicubic',bicubic),(f'{stage} steps',pred)]:
                    arr=(image[0,:,y0:y1,x0:x1].detach().cpu().permute(1,2,0).numpy()*255).round().astype('uint8')
                    parts.append(Image.fromarray(arr))
                canvas=Image.new('RGB',((x1-x0)*3,y1-y0+26),'white');draw=ImageDraw.Draw(canvas)
                for col,(label,im) in enumerate(zip(['Original','Bicubic',f'Gemino {stage}'],parts)):
                    canvas.paste(im,(col*(x1-x0),26));draw.text((col*(x1-x0)+5,4),f'{label}, t={index/30:.2f}s',fill='black')
                dest=out/f'{identifier}_f{index:04d}'
                if kind=='low_head':canvas.save(dest.with_suffix('.png'))
                else:canvas.save(dest.with_suffix('.jpg'),quality=95,subsampling=0)
        mean={key:float(np.mean([row[key] for row in values])) for key in values[0] if key not in ['frame','seconds']}
        report={'kind':kind,'stage':stage,'source_id':record['source_id'],'source_start':record['virtual_start_seconds'],
                'metric_samples':len(values),'decoded_frames':150,'face_roi_xyxy':ROI,'inputs_sha256':hashes.hexdigest(),
                'metrics_before_preview_encoding':True,'means':mean,'frames':values,
                'note':'Face ROI is a fixed rectangle, not a ground-truth face/ghost annotation. Metrics do not by themselves prove ghost removal.'}
        reports.append(report);(out/'metrics.json').write_text(json.dumps(reports,indent=2))
        print(json.dumps({'case':identifier,'samples':len(values),'means':mean}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');main(p.parse_args())
