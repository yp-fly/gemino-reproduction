from pathlib import Path
import shutil
r=Path(__file__).resolve().parents[1];d=r/'reproduction/upstream'
d.mkdir(exist_ok=True)
shutil.copytree(r/'vendor/model',d/'model-master',dirs_exist_ok=True)
shutil.copytree(r/'vendor/aiortc_src',d/'aiortc-main/src',dirs_exist_ok=True)
print('Fetch source videos and verified FOMM/VGG weights as documented before training.')
