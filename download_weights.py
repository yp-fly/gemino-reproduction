import hashlib,json,urllib.request,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
 m=json.loads((ROOT/'model_manifest.json').read_text());p=ROOT/'weights/gemino-step30500.pt';p.parent.mkdir(exist_ok=True)
 h=hashlib.sha256()
 if p.exists():
  with p.open('rb') as f:
   for b in iter(lambda:f.read(2**20),b''):h.update(b)
  if h.hexdigest()==m['sha256']:print('Weights already verified');return
 temporary=p.with_suffix('.download')
 request=urllib.request.Request(m['url'],headers={'User-Agent':'gemino-reproduction'})
 with urllib.request.urlopen(request,timeout=120) as response,temporary.open('wb') as output:
  while True:
   data=response.read(2**20)
   if not data:break
   output.write(data)
 h=hashlib.sha256()
 with temporary.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 if h.hexdigest()!=m['sha256']:raise RuntimeError('Downloaded checkpoint SHA256 mismatch')
 temporary.replace(p);print('Downloaded and verified:',p)
if __name__=='__main__':main()
