"""python -m tests: run every self-test, each in its own process."""

import subprocess
import sys
from pathlib import Path

failed = []
for f in sorted(Path(__file__).parent.glob("test_*.py")):
    print(f"== {f.stem}", flush=True)
    if subprocess.run([sys.executable, "-m", f"tests.{f.stem}"]).returncode != 0:
        failed.append(f.stem)
print(f"\n{'ALL TESTS PASS' if not failed else 'FAILED: ' + ', '.join(failed)}")
sys.exit(1 if failed else 0)
