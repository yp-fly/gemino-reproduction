"""Portable single-GPU Gemino adapter. Upstream model files remain unchanged."""
import os, sys, json, random, hashlib
from pathlib import Path
from fractions import Fraction
import numpy as np
import torch
import torch.nn.functional as F
import av
import yaml

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'upstream/model-master'))
os.environ['TORCH_HOME']=str(ROOT/'weights/torch')
from first_order_model.modules.generator import OcclusionAwareGenerator
from first_order_model.modules.keypoint_detector import KPDetector
from first_order_model.modules.discriminator import MultiScaleDiscriminator

def read_config(path):
    path=Path(path);path=path if path.is_absolute() else ROOT/path
    return yaml.safe_load(path.read_text(encoding='utf-8'))

def resolve(path):
    p=Path(path);return p if p.is_absolute() else ROOT/p

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def models(config,device='cpu'):
    m=config['model_params']
    g=OcclusionAwareGenerator(**m['generator_params'],**m['common_params']).to(device)
    k=KPDetector(**m['kp_detector_params'],**m['common_params']).to(device)
    d=MultiScaleDiscriminator(**m['discriminator_params'],**m['common_params']).to(device)
    return g,k,d

def reset_cache(g):
    # Cache holds graphs across updates when the same reference is sampled twice.
    for name in ['source_image','encoder_output','skip_connections']:
        if hasattr(g,name):setattr(g,name,None)
    if g.dense_motion_network is not None:
        for name in ['source_image','gaussian_source','source_repeat','identity_grid']:
            if hasattr(g.dense_motion_network,name):setattr(g.dense_motion_network,name,None)

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def initialize_fomm(g,k,path,report):
    # Only load a file verified against the SHA256 in the author's public Yandex metadata.
    meta=json.loads((ROOT/'reports/fomm_official_metadata.json').read_text())
    actual=sha256(path)
    if actual!=meta['sha256']:raise ValueError('FOMM file does not match official SHA256')
    state=torch.load(path,map_location='cpu',weights_only=False)
    audit={'file':str(path),'sha256':actual,'modules':{}}
    for name,module in [('generator',g),('kp_detector',k)]:
        old={n.removeprefix('module.'):v for n,v in state[name].items()}
        own=module.state_dict();matched={n:v for n,v in old.items() if n in own and own[n].shape==v.shape}
        mismatch={n:{'source':list(v.shape),'target':list(own[n].shape)} for n,v in old.items() if n in own and own[n].shape!=v.shape}
        if not matched:raise ValueError('No compatible weights for '+name)
        module.load_state_dict(matched,strict=False)
        audit['modules'][name]={'matched_keys':list(matched),'random_or_new_keys':sorted(set(own)-set(matched)),
                                'shape_mismatch':mismatch,'unused_source_keys':sorted(set(old)-set(own)),
                                'matched_elements':sum(v.numel() for v in matched.values()),
                                'total_elements':sum(v.numel() for v in own.values())}
    Path(report).write_text(json.dumps(audit,indent=2),encoding='utf-8')
    return audit

class Codec:
    """Actual libvpx encode/decode, independent keyframes for training or continuous for eval.
    PyAV/libvpx settings are explicit; they are not claimed bit-exact with authors' aiortc.
    """
    def __init__(self,size=256,bitrate=45000,fps=30,codec='vp8'):
        self.size=size;self.count=0;self.bytes=0;self.fps=fps
        name={'vp8':'libvpx','h264':'libx264'}[codec]
        self.encoder=av.CodecContext.create(name,'w');self.encoder.width=size;self.encoder.height=size
        self.encoder.pix_fmt='yuv420p';self.encoder.bit_rate=bitrate
        self.encoder.time_base=Fraction(1,fps);self.encoder.framerate=Fraction(fps,1)
        self.encoder.thread_count=1;self.encoder.gop_size=60;self.encoder.max_b_frames=0
        self.encoder.options=({'deadline':'realtime','cpu-used':'8','lag-in-frames':'0','auto-alt-ref':'0',
                               'minrate':str(bitrate),'maxrate':str(bitrate),'bufsize':str(bitrate)} if codec=='vp8' else
                              {'preset':'ultrafast','tune':'zerolatency','maxrate':str(bitrate),'bufsize':str(bitrate),'sc_threshold':'0'})
        self.decoder=av.CodecContext.create(codec,'r');self.decoder.thread_count=1
    def frame(self,rgb):
        frame=av.VideoFrame.from_ndarray(np.asarray(rgb,dtype=np.uint8),format='rgb24')
        frame=frame.reformat(width=self.size,height=self.size,format='yuv420p');frame.pts=self.count
        frame.time_base=Fraction(1,self.fps);self.count+=1
        packets=self.encoder.encode(frame);out=[]
        for p in packets:
            self.bytes+=p.size;out.extend(self.decoder.decode(p))
        if len(out)!=1:raise RuntimeError(f'Expected one zero-latency decoded frame, got {len(out)}')
        return out[0].to_ndarray(format='rgb24')

def tensor(rgb):return torch.from_numpy(np.asarray(rgb).copy()).permute(2,0,1).float()/255

def read_selected_frames(path,indices,size,start_seconds=None,crop=None):
    """Sequential decode two frames: bounded RAM, avoids loading a 10s 1024 clip into RAM."""
    if start_seconds is not None:
        result={};wanted=sorted(set(indices));position=0
        with av.open(str(path)) as c:
            st=c.streams.video[0];st.thread_count=1
            c.seek(int((start_seconds+wanted[0]/30)*av.time_base),backward=True)
            for frame in c.decode(video=0):
                if frame.time is None:continue
                while position<len(wanted) and frame.time+1e-7>=start_seconds+wanted[position]/30:
                    rgb=frame.to_ndarray(format='rgb24')
                    x,y,w,h=crop;rgb=rgb[y:y+h,x:x+w]
                    if rgb.shape[:2]!=(h,w):raise ValueError('Crop outside native source')
                    if (h,w)!=(size,size):
                        from PIL import Image
                        rgb=np.asarray(Image.fromarray(rgb).resize((size,size),Image.Resampling.BICUBIC))
                    result[wanted[position]]=rgb;position+=1
                if position==len(wanted):break
        if len(result)!=len(set(indices)):raise ValueError('Source too short for selected times: '+str(path))
        return [result[i] for i in indices]
    result={}
    with av.open(str(path)) as c:
        c.streams.video[0].thread_count=1
        for i,frame in enumerate(c.decode(video=0)):
            if i in indices:
                frame=frame.reformat(width=size,height=size,format='rgb24');result[i]=frame.to_ndarray()
            if i>=max(indices):break
    if len(result)!=len(set(indices)):raise ValueError('Missing requested frames: '+str(path))
    return [result[i] for i in indices]

class Pairs(torch.utils.data.Dataset):
    def __init__(self,cfg,split='train'):
        self.cfg=cfg;self.split=split
        rows=json.loads(resolve(cfg['data_manifest']).read_text(encoding='utf-8'))
        groups={s:{r['source_id'] for r in rows if r['split']==s} for s in ['train','val','test']}
        assert not(groups['train']&groups['val'] or groups['train']&groups['test'] or groups['val']&groups['test'])
        self.rows=[r for r in rows if r['split']==split]
        if not self.rows:raise ValueError('No prepared data for split: '+split)
        self.repeat=cfg['train_params']['num_repeats'] if split=='train' else 1
    def __len__(self):return len(self.rows)*self.repeat
    def __getitem__(self,index):
        r=self.rows[index%len(self.rows)];n=r['frames'];size=self.cfg['output_size']
        indices=random.sample(range(n),2) if self.split=='train' else [0,min(n-1,30)]
        a,b=read_selected_frames(resolve(r['path']),indices,size,r.get('virtual_start_seconds'),r.get('crop'))
        # Paired appearance augmentation before deriving compressed LR target.
        if self.split=='train' and self.cfg.get('augment',True):
            if random.random()<.5:a=a[:,::-1].copy();b=b[:,::-1].copy()
            from PIL import Image
            from torchvision.transforms import ColorJitter, functional as TF
            order,brightness,contrast,saturation,hue=ColorJitter.get_params((.9,1.1),(.9,1.1),(.9,1.1),(-.1,.1))
            def jitter(x):
                x=Image.fromarray(x)
                for op in order.tolist():
                    if op==0:x=TF.adjust_brightness(x,brightness)
                    elif op==1:x=TF.adjust_contrast(x,contrast)
                    elif op==2:x=TF.adjust_saturation(x,saturation)
                    else:x=TF.adjust_hue(x,hue)
                return np.asarray(x)
            a,b=jitter(a),jitter(b)
        codec=Codec(**self.cfg['codec']);low=codec.frame(b)
        return {'source':tensor(a),'driving':tensor(b),'driving_lr':tensor(low)}

def assert_ready(cfg):
    if cfg.get('purpose')=='formal':
        approval=ROOT/'reports/data_review.json'
        if not approval.exists():raise RuntimeError('Dataset visual review missing; run dataset audit and review crops first')
        review=json.loads(approval.read_text())
        if not review.get('approved') or review.get('manifest_sha256')!=sha256(resolve(cfg['data_manifest'])):
            raise RuntimeError('Dataset review incomplete or manifest changed')
        for split in ['train','val','test']:Pairs(cfg,split)
        rows=Pairs(cfg,'train').rows
        if sum(r['frames']/r['fps'] for r in rows)<cfg.get('minimum_training_seconds',900):
            raise RuntimeError('Formal training requires >=900 s curated training footage; pilot config is available for the short dataset')
        if len({r['source_id'] for r in rows})<cfg.get('minimum_training_sources',3):
            raise RuntimeError('Formal training needs at least 3 distinct source videos in this preparation plan')

def checkpoint_state(g,k,d,optimizers,schedulers,cfg,epoch,step,batch_index):
    return dict(generator=g.state_dict(),kp_detector=k.state_dict(),discriminator=d.state_dict(),
                optimizers=[x.state_dict() for x in optimizers],schedulers=[x.state_dict() for x in schedulers],
                config=cfg,epoch=epoch,global_step=step,batch_index=batch_index,
                torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                note='Locally trained Gemino adaptation; not author-provided checkpoint')

def save_atomic(state,path):
    path=Path(path);temp=path.with_suffix('.tmp');torch.save(state,temp);temp.replace(path)
