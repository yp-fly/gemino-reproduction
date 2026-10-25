from pathlib import Path
import runpy,os
root=Path(__file__).resolve().parents[1]
base=root/'vendor/aiortc_src'
os.chdir(base)
runpy.run_path(str(base/'_cffi_src/build_vpx.py'),run_name='__main__')
