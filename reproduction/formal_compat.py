"""Audited adapters: upstream model/loss/codec sources remain unchanged."""
import sys,types,json,random,hashlib,logging
from pathlib import Path
from fractions import Fraction
import numpy as np, torch, av
import core

def codec_module():
    # Load the codec subpackage without importing unrelated WebRTC transport dependencies.
    if 'aiortc.codecs.vpx' in sys.modules:return sys.modules['aiortc.codecs.vpx']
    base=core.ROOT/'upstream/aiortc-main/src/aiortc'
    for name,path in [('aiortc',base),('aiortc.codecs',base/'codecs')]:
        mod=types.ModuleType(name);mod.__path__=[str(path)];sys.modules[name]=mod
    import importlib
    module=importlib.import_module('aiortc.codecs.vpx')
    logging.getLogger('aiortc.codecs.vpx').setLevel(logging.ERROR)
    return module

class OfficialCodec:
    def __init__(self,size=256,bitrate=45000,fps=30,codec='vp8'):
        assert codec=='vp8';self.size=size;self.bitrate=bitrate;self.fps=fps;self.count=0;self.bytes=0
        self.module=codec_module();self.encoder=self.module.Vp8Encoder();self.decoder=self.module.Vp8Decoder()
    def frame(self,rgb):
        # Same default nearest F.interpolate as upstream train.py, before RGB->YUV conversion.
        t=core.F.interpolate(core.tensor(rgb)[None],size=self.size,mode='nearest')
        array=(t[0].permute(1,2,0).numpy()*255).astype('uint8')
        frame=av.VideoFrame.from_ndarray(array,format='rgb24');frame.pts=self.count*3000;frame.time_base=Fraction(1,90000)
        payloads,stamp=self.encoder.encode(frame,quantizer=-1,target_bitrate=self.bitrate,enable_gcc=False)
        data=b''.join(self.module.vp8_depayload(p) for p in payloads);self.bytes+=len(data);self.count+=1
        decoded=self.decoder.decode(self.module.JitterFrame(data=data,timestamp=stamp))
        assert len(decoded)==1
        return decoded[0].to_rgb().to_ndarray()

class OfficialPairs(core.Pairs):
    def __init__(self,cfg,split='train'):
        super().__init__(cfg,split)
        import skimage.util
        if not hasattr(skimage.util,'pad'):skimage.util.pad=np.pad  # removed alias, same NumPy operation
        from first_order_model.augmentation import AllAugmentationTransform
        self.transform=AllAugmentationTransform(**cfg['official_augmentation'])
        self.groups={}
        for r in self.rows:self.groups.setdefault(r.get('speaker',r['source_id']),[]).append(r)
    def __getitem__(self,index):
        r=self.rows[index%len(self.rows)]
        if self.split=='train':
            # Mirrors upstream mp4 personalized sampling: two independently selected clips of same person.
            target=random.choice(self.groups[r.get('speaker',r['source_id'])])
            ia=int(np.random.choice(r['frames']-1));ib=int(np.random.choice(target['frames']-1))
        else:target=r;ia=0;ib=min(r['frames']-1,30)
        a=core.read_selected_frames(core.resolve(r['path']),[ia],self.cfg['output_size'],r['virtual_start_seconds'],r['crop'])[0]
        b=core.read_selected_frames(core.resolve(target['path']),[ib],self.cfg['output_size'],target['virtual_start_seconds'],target['crop'])[0]
        if self.split=='train' and self.cfg.get('augment',True):
            a,b=self.transform([a.astype('float32')/255,b.astype('float32')/255])
            a=(np.clip(a,0,1)*255).round().astype('uint8');b=(np.clip(b,0,1)*255).round().astype('uint8')
        low=OfficialCodec(**self.cfg['codec']).frame(b)
        return {'source':core.tensor(a),'driving':core.tensor(b),'driving_lr':core.tensor(low)}

def initialize_official(g,k,path,report):
    metadata=json.loads((core.ROOT/'reports/fomm_official_metadata.json').read_text())
    assert core.sha256(path)==metadata['sha256']
    state=torch.load(path,map_location='cpu',weights_only=False)
    old={n.removeprefix('module.'):v for n,v in state['generator'].items()}
    # Exact exclusion policy in upstream Logger.load_cpk for hourglass_input + decoder + HR skips.
    excluded=('final','sigmoid','up_blocks','dense_motion_network.hourglass','dense_motion_network.mask','dense_motion_network.occlusion','bottleneck')
    selected={n:v for n,v in old.items() if not n.startswith(excluded)}
    own=g.state_dict();bad={n for n,v in selected.items() if n not in own or v.shape!=own[n].shape}
    assert not bad,f'Official initialization mismatches: {bad}'
    g.load_state_dict(selected,strict=False)
    k.load_state_dict({n.removeprefix('module.'):v for n,v in state['kp_detector'].items()},strict=True)
    discriminator=getattr(core,'_formal_discriminator',None);dpolicy='random: no compatible discriminator'
    if discriminator is not None and 'discriminator' in state:
        ds={n.removeprefix('module.'):v for n,v in state['discriminator'].items()};own_d=discriminator.state_dict()
        if ds.keys()==own_d.keys() and all(v.shape==own_d[n].shape for n,v in ds.items()):
            discriminator.load_state_dict(ds);dpolicy='pretrained discriminator loaded exactly'
    audit={'initialization':'upstream Logger.load_cpk exclusions, not prior shape-only trial initialization',
           'generator_loaded':sorted(selected),'generator_random':sorted(set(own)-set(selected)),
           'kp_detector':'all pretrained','discriminator':dpolicy,'sha256':metadata['sha256']}
    Path(report).write_text(json.dumps(audit,indent=2))

def activate():
    codec_module()
    previous_reset=core.reset_cache
    def reset_all(g):
        previous_reset(g)
        # Upstream global Gaussian cache includes batch dimension; reset on train/val transitions.
        from first_order_model.modules import util
        util.coordinate_grid_global=None;util.shape_global=None
    core.reset_cache=reset_all
    # No grid_sample patch: repository env is torch1.6, whose default align_corners=False matches torch2.5.
    core.Pairs=OfficialPairs;core.Codec=OfficialCodec;core.initialize_fomm=initialize_official
    original=core.models
    def wrapped(*args,**kwargs):
        g,k,d=original(*args,**kwargs);core._formal_discriminator=d;return g,k,d
    core.models=wrapped
