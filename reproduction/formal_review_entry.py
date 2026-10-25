import sys,runpy
import formal_compat
formal_compat.activate()
module=sys.argv.pop(1)
runpy.run_module(module,run_name='__main__')
