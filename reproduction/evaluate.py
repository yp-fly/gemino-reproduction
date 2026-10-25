"""Continuous low-resolution codec evaluation; fixed losslessly stored first reference."""
import argparse,json,time,io
import av,numpy as np,torch
from PIL import Image,ImageDraw
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
from core import *

def frames_for_record(record,size,limit):
    count=0;maximum=min(record['frames'],limit) if limit else record['frames']
    with av.open(str(resolve(record['path']))) as container:
        if 'virtual_start_seconds' in record:
            start=record['virtual_start_seconds'];container.seek(int(start*av.time_base),backward=True)
            for frame in container.decode(video=0):
                if frame.time is None:continue
                while count<maximum and frame.time+1e-7>=start+count/30:
                    rgb=frame.to_ndarray(format='rgb24');x,y,w,h=record['crop'];rgb=rgb[y:y+h,x:x+w]
                    if rgb.shape[:2]!=(h,w):raise ValueError('Invalid evaluation crop')
                    if rgb.shape[:2]!=(size,size):rgb=np.asarray(Image.fromarray(rgb).resize((size,size),Image.Resampling.BICUBIC))
                    yield rgb;count+=1
                if count==maximum:break
        else:
            for frame in container.decode(video=0):
                if count>=maximum:break
                yield frame.reformat(width=size,height=size,format='rgb24').to_ndarray();count+=1
    if count!=maximum:raise ValueError(f'Insufficient decoded frames: {count}/{maximum}')

def main(a):
    checkpoint=torch.load(resolve(a.checkpoint),map_location='cpu',weights_only=False);cfg=checkpoint['config']
    if a.manifest:cfg['data_manifest']=a.manifest
    torch.set_num_threads(a.cpu_threads);device=torch.device(a.device)
    g,k,d=models(cfg,device);g.load_state_dict(checkpoint['generator']);k.load_state_dict(checkpoint['kp_detector']);del d,checkpoint
    g.eval();k.eval();out=resolve(a.output);out.mkdir(parents=True,exist_ok=True)
    metric=None
    if not a.no_lpips:
        import lpips
        metric=lpips.LPIPS(net='vgg').to(device).eval()
    rows=json.loads(resolve(cfg['data_manifest']).read_text());rows=[r for r in rows if r['split']==a.split]
    if a.distinct_sources:
        seen=set();chosen=[]
        for r in rows:
            if r['source_id'] not in seen:chosen.append(r);seen.add(r['source_id'])
        rows=chosen
    if not rows:raise ValueError('No evaluation clips for '+a.split)
    reports=[]
    for record in rows[:a.clips]:
        codec=Codec(**cfg['codec']);values=[];reference=None
        stem=Path(record['path']).stem+'_'+str(record.get('virtual_start_seconds',0))
        dest=out/(stem+'_comparison.mp4')
        video=av.open(str(dest),'w');st=video.add_stream('libx264',rate=30);st.width=cfg['output_size']*3;st.height=cfg['output_size'];st.pix_fmt='yuv420p';st.options={'crf':'18','preset':'fast'}
        try:
            for index,rgb in enumerate(frames_for_record(record,cfg['output_size'],a.frames)):
                if reference is None:
                    buffer=io.BytesIO();Image.fromarray(rgb).save(buffer,format='PNG');reference_bytes=len(buffer.getvalue())
                    (out/(stem+'_reference.png')).write_bytes(buffer.getvalue())
                    reference=tensor(rgb)[None].to(device);reset_cache(g)
                    with torch.no_grad():source_kp=k(reference)
                low=codec.frame(rgb);lr=tensor(low)[None].to(device)
                bicubic=F.interpolate(lr,size=(cfg['output_size'],cfg['output_size']),mode='bicubic',align_corners=False).clamp(0,1)
                if device.type=='cuda':torch.cuda.synchronize()
                start=time.perf_counter()
                with torch.no_grad():
                    prediction=g(reference,kp_source=source_kp,kp_driving=k(lr),driving_lr=lr,update_source=index==0)['prediction'].clamp(0,1)
                if device.type=='cuda':torch.cuda.synchronize()
                elapsed=time.perf_counter()-start
                target=tensor(rgb)[None].to(device)
                item={'index':index,'reconstruction_ms':elapsed*1000}
                images=[]
                for name,t in [('gemino',prediction),('bicubic',bicubic)]:
                    array=t[0].cpu().permute(1,2,0).numpy();truth=rgb.astype('float32')/255
                    item[name+'_psnr']=float(peak_signal_noise_ratio(truth,array,data_range=1))
                    item[name+'_ssim']=float(structural_similarity(truth,array,data_range=1,channel_axis=2))
                    if metric:
                        with torch.no_grad():item[name+'_lpips']=float(metric(t*2-1,target*2-1))
                    images.append((array*255).round().astype('uint8'))
                values.append(item)
                # Comparison order: original / bicubic / Gemino.
                strip=np.concatenate([rgb,images[1],images[0]],axis=1)
                canvas=Image.fromarray(strip);draw=ImageDraw.Draw(canvas)
                labels=['Original','Bicubic','Gemino']
                for col,label in enumerate(labels):
                    left=col*cfg['output_size'];draw.rectangle((left,0,left+cfg['output_size'],25),fill='black')
                    draw.text((left+8,5),label,fill='white')
                if cfg.get('purpose')=='smoke':
                    draw.rectangle((0,cfg['output_size']-27,strip.shape[1],cfg['output_size']),fill='black')
                    draw.text((8,cfg['output_size']-22),'ENGINEERING CHECK ONLY - few updates - NOT trained Gemino quality',fill='white')
                strip=np.asarray(canvas)
                for packet in st.encode(av.VideoFrame.from_ndarray(strip,format='rgb24')):video.mux(packet)
            for packet in st.encode():video.mux(packet)
        finally:video.close()
        duration=len(values)/30
        report={'clip':record['path'],'frames':len(values),'reference_bytes_png':reference_bytes,
                'video_payload_kbps':codec.bytes*8/duration/1000,'with_initial_reference_kbps':(codec.bytes+reference_bytes)*8/duration/1000,
                'reference_policy':'first frame, PNG, once per evaluation clip; no future references',
                'comparison_order':['original','bicubic','gemino'],'metrics':values,
                'means':{key:float(np.mean([v[key] for v in values])) for key in values[0] if key!='index'},
                'note':'Protocol overhead excluded. This does not claim reproduction of paper results.'}
        reports.append(report);(out/'metrics.json').write_text(json.dumps(reports,indent=2));print(report['means'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',default='runs/evaluation')
    p.add_argument('--split',choices=['val','test','train'],default='val');p.add_argument('--device',default='cuda')
    p.add_argument('--clips',type=int,default=1);p.add_argument('--frames',type=int,default=150);p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--manifest',help='Optional data manifest override; model architecture remains from checkpoint')
    p.add_argument('--distinct-sources',action='store_true',help='Pick one clip per source video before applying --clips')
    p.add_argument('--no-lpips',action='store_true');main(p.parse_args())
