"""Standalone Gemino step30500 inference. No training dataset or FOMM download required."""
import argparse,json,sys,time,hashlib,urllib.request
from pathlib import Path
import numpy as np,torch,av,yaml
from PIL import Image
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor/model'))
from first_order_model.modules.generator import OcclusionAwareGenerator
from first_order_model.modules.keypoint_detector import KPDetector
from first_order_model.modules import util

def tensor(a,device):return torch.from_numpy(np.asarray(a).copy()).permute(2,0,1).float().unsqueeze(0).to(device)/255
def load_models(path,device):
 state=torch.load(path,map_location='cpu',weights_only=True);assert state['global_step']==30500
 cfg=state['config'];m=cfg['model_params']
 g=OcclusionAwareGenerator(**m['generator_params'],**m['common_params']).to(device)
 k=KPDetector(**m['kp_detector_params'],**m['common_params']).to(device)
 g.load_state_dict(state['generator'],strict=True);k.load_state_dict(state['kp_detector'],strict=True)
 g.eval();k.eval();return g,k,cfg
def main(a):
 torch.set_num_threads(a.cpu_threads);device=torch.device(a.device)
 if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; install matching PyTorch or pass --device cpu')
 weight=Path(a.weights)
 if not weight.exists():raise FileNotFoundError('Download release weights first: python download_weights.py')
 expected=json.loads((ROOT/'model_manifest.json').read_text())['sha256']
 h=hashlib.sha256()
 with weight.open('rb') as f:
  for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
 if h.hexdigest()!=expected:raise ValueError('Weight SHA256 mismatch')
 g,k,cfg=load_models(weight,device);codec=None
 if a.input_mode=='high':
  from codec import OfficialCodec
  codec=OfficialCodec()
 reference=None
 if a.reference:
  im=Image.open(a.reference).convert('RGB');assert im.size==(1024,1024),'Reference must be1024x1024'
  reference=np.asarray(im)
 if a.input_mode=='low' and reference is None:raise ValueError('--reference is required for decoded low-resolution video')
 source=None;source_kp=None;count=0;times=[];destination=Path(a.output);destination.parent.mkdir(parents=True,exist_ok=True)
 with av.open(a.input) as video,av.open(str(destination),'w') as output:
  st=output.add_stream('libx264',rate=30);st.width=1024;st.height=1024;st.pix_fmt='yuv420p';st.options={'crf':'18','preset':'fast'}
  # Timestamp-select at30Hz for high-res simulation. Low streams must already be256x256@30fps.
  start=None
  for frame in video.decode(video=0):
   if a.max_frames and count>=a.max_frames:break
   if start is None:start=frame.time or 0
   if a.input_mode=='high' and frame.time is not None and frame.time+1e-7<start+count/30:continue
   rgb=frame.to_ndarray(format='rgb24')
   if a.input_mode=='high':
    if rgb.shape[:2]!=(1024,1024):
     if a.crop!='center' or min(rgb.shape[:2])<1024:raise ValueError('High input must be1024x1024; use --crop center for larger native frames')
     y=(rgb.shape[0]-1024)//2;x=(rgb.shape[1]-1024)//2;rgb=rgb[y:y+1024,x:x+1024]
    low=codec.frame(rgb)
    if reference is None:reference=rgb.copy()
   else:
    if rgb.shape[:2]!=(256,256):raise ValueError('Decoded low input must be256x256')
    low=rgb
   with torch.inference_mode():
    if source is None:
     util.coordinate_grid_global=None;util.shape_global=None
     source=tensor(reference,device);source_kp=k(source)
    lr=tensor(low,device)
    if device.type=='cuda':torch.cuda.synchronize()
    started=time.perf_counter();pred=g(source,kp_source=source_kp,kp_driving=k(lr),driving_lr=lr,update_source=False)['prediction'].clamp(0,1)
    if device.type=='cuda':torch.cuda.synchronize()
    times.append(time.perf_counter()-started)
    image=(pred[0].cpu().permute(1,2,0).numpy()*255).round().astype('uint8')
   for packet in st.encode(av.VideoFrame.from_ndarray(image,format='rgb24')):output.mux(packet)
   count+=1
  for packet in st.encode():output.mux(packet)
 if not count:raise RuntimeError('No frames decoded')
 report={'frames':count,'output_size':[1024,1024],'fps':30,'checkpoint_step':30500,'input_mode':a.input_mode,'mean_model_ms':1000*float(np.mean(times)),'note':'Model-only time excludes reference encoding, network and display. Codec label45k uses official internal200k mapping.'}
 if codec:report['video_payload_kbps']=codec.bytes*8/(count/30)/1000
 destination.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--weights',default=str(ROOT/'weights/gemino-step30500.pt'));p.add_argument('--input',required=True);p.add_argument('--output',required=True)
 p.add_argument('--input-mode',choices=['high','low'],default='high');p.add_argument('--reference');p.add_argument('--crop',choices=['none','center'],default='none');p.add_argument('--max-frames',type=int,default=0);p.add_argument('--cpu-threads',type=int,default=4);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');main(p.parse_args())
