"""Evaluate checkpoints from formal_aligned.yaml with their matching codec and sampling pipeline."""
import formal_compat
formal_compat.activate()
import runpy
runpy.run_module('evaluate',run_name='__main__')
