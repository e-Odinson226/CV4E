"""
Gaze- and hand-conditioned V-JEPA 2-AC, and the experiments run on it.

Library modules, used by more than one command:

    predictor  the ego predictor (gaze and hand tokens in place of robot action and state)
    model      building, loading, freezing and running the encoder and the predictor
    data       recordings, timestamps, video frames, and the gaze and hand loaders
    signals    the gaze and hand inputs of a clip, and their variants (hidden, swapped, ...)
    clips      the fixed evaluation clips and how a clip is scored
    stats      the paired comparison used for every Delta
    linprobe   the linear probe on frozen encoder features (Test 3)
    runlog     the log helper, and the parser for training logs

The commands are in ego.commands. Run them from the repository root:

    python -m ego <command> --help

Meta's V-JEPA 2 code must be in vjepa2/ at the repository root (see README.md). It is
imported as `src.*`, so its folder is put on the Python path here.
"""

import sys
from pathlib import Path

VJEPA2_DIR = Path(__file__).resolve().parent.parent / "vjepa2"
if str(VJEPA2_DIR) not in sys.path:
    sys.path.insert(0, str(VJEPA2_DIR))
