import runpy
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


if __name__ == '__main__':
    runpy.run_module('png.misc.generate_burst_images', run_name='__main__')
