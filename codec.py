"""Original Gemino aiortc VP8 source, isolated from unrelated WebRTC transports."""
import sys,types,importlib,logging
from pathlib import Path
from fractions import Fraction
import av,numpy as np,torch
ROOT=Path(__file__).resolve().parent
def load_codec():
 base=ROOT/'vendor/aiortc_src/aiortc'
 for name,path in [('aiortc',base),('aiortc.codecs',base/'codecs')]:
  if name not in sys.modules:m=types.ModuleType(name);m.__path__=[str(path)];sys.modules[name]=m
 try:module=importlib.import_module('aiortc.codecs.vpx')
 except ImportError as error:raise RuntimeError('Build codec first: python tools/build_codec.py (Linux; needs libvpx-dev and compiler)') from error
 logging.getLogger('aiortc.codecs.vpx').setLevel(logging.ERROR);return module
class OfficialCodec:
 def __init__(self):
  self.m=load_codec();self.encoder=self.m.Vp8Encoder();self.decoder=self.m.Vp8Decoder();self.count=0;self.bytes=0
 def frame(self,rgb):
  t=torch.from_numpy(rgb.copy()).permute(2,0,1).float()[None]/255
  small=torch.nn.functional.interpolate(t,size=256,mode='nearest')[0].permute(1,2,0).numpy()
  f=av.VideoFrame.from_ndarray((small*255).astype('uint8'),format='rgb24');f.pts=self.count*3000;f.time_base=Fraction(1,90000)
  payloads,stamp=self.encoder.encode(f,quantizer=-1,target_bitrate=45000,enable_gcc=False)
  data=b''.join(self.m.vp8_depayload(p) for p in payloads);self.bytes+=len(data);self.count+=1
  frames=self.decoder.decode(self.m.JitterFrame(data=data,timestamp=stamp));assert len(frames)==1
  return frames[0].to_rgb().to_ndarray()
