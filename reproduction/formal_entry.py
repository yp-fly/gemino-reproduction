import formal_compat
formal_compat.activate()
import runpy,sys
if '--snapshot-steps' not in sys.argv:sys.argv+=['--snapshot-steps','3000,10000']
runpy.run_module('train',run_name='__main__')
