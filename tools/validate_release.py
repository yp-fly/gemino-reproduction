from pathlib import Path
import json,subprocess,sys,av,numpy as np
from PIL import Image
root=Path(__file__).resolve().parent
if root.name=='tools':root=root.parent
sys.path.insert(0,str(root))
from codec import OfficialCodec
out=root/'validation';out.mkdir(exist_ok=True)
codec=OfficialCodec();first=None
with av.open(str(root/'examples/synthetic.mp4')) as source,av.open(str(out/'low_lossless.mkv'),'w') as dest:
 st=dest.add_stream('ffv1',rate=30);st.width=256;st.height=256;st.pix_fmt='bgr0'
 for f in source.decode(video=0):
  rgb=f.to_ndarray(format='rgb24')
  if first is None:first=rgb;Image.fromarray(first).save(out/'reference.png')
  low=codec.frame(rgb)
  for packet in st.encode(av.VideoFrame.from_ndarray(low,format='rgb24')):dest.mux(packet)
 for packet in st.encode():dest.mux(packet)
subprocess.run([sys.executable,str(root/'infer.py'),'--input',str(out/'low_lossless.mkv'),'--input-mode','low','--reference',str(out/'reference.png'),'--output',str(out/'low_mode_reconstructed.mp4'),'--device','cuda'],check=True)
def frames(p):
 with av.open(str(p)) as c:return [f.to_ndarray(format='rgb24') for f in c.decode(video=0)]
high=frames(out/'synthetic_reconstructed.mp4');low=frames(out/'low_mode_reconstructed.mp4')
assert len(high)==len(low)==4
diff=max(float(np.abs(a.astype('float32')-b.astype('float32')).mean()) for a,b in zip(high,low));assert diff<1
report={'passed':True,'checkpoint_step':30500,'runtime':'Linux Python3.10 torch2.5.1 CUDA A800','standalone_high_mode_frames':4,'standalone_low_mode_frames':4,'high_vs_low_max_mean_pixel_difference_255':diff,'model_load':'weights_only=True; strict=True; SHA256 verified','note':'Synthetic sample tests runnable pipeline, not face quality. Model quality metrics are separate validation results.'}
(out/'release_validation.json').write_text(json.dumps(report,indent=2));print('RELEASE_VALIDATION_PASSED',flush=True)
