"""Wrapper: set the CPython GIL switch interval, then exec the target script.
Usage: gil_exec.py <interval_s> <script> [args...]
"""
import runpy
import sys

sys.setswitchinterval(float(sys.argv[1]))
sys.argv = sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
