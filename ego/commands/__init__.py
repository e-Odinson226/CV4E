"""
The commands, one module each. Run one from the repository root with

    python -m ego <command> [options]

Each module also runs on its own: python -m ego.commands.<module> [options].
"""

# name -> (module, one-line description)
COMMANDS = {
    "extract-csvs":   ("extract_csvs",   "extract the gaze and hand CSVs from the HD-EPIC MPS zips"),
    "fetch-calibrations": ("fetch_calibrations", "fetch the HD-EPIC camera calibrations without the SLAM zips"),
    "draw-gaze":      ("draw_gaze",      "check the gaze projection: overlays, where the points fall, depth effect"),
    "gaze-at-picks":  ("gaze_at_picks",  "check the gaze projection: is the gaze point on the object about to be picked up?"),
    "train":          ("train",          "fine-tune the ego predictor (Test 1; with --signal-dropout 1.0, the Test 2 control)"),
    "evaluate":       ("evaluate",       "paired error, signals hidden vs real, on any participants"),
    "gaze-probe":     ("gaze_probe",     "Test 3a: can gaze be read from the frozen encoder's features?"),
    "control-probe":  ("control_probe",  "Test 3b: the gaze probe with palm position as the target"),
    "sensitivity":    ("sensitivity",    "Test 4a: does the prediction change when gaze changes?"),
    "attention":      ("attention",      "Test 4b: how much attention goes to the gaze token?"),
    "next-object":    ("next_object",    "Test 8: does the gaze point tell which object is picked up next?"),
    "gaze-forms":     ("gaze_forms",     "Test 9: compare the gaze forms on P08 and P09 (every finished run)"),
    "weight-norms":   ("weight_norms",   "Test 4c: did training shrink the gaze layer?"),
    "stock-vs-tuned": ("stock_vs_tuned", "Test 2: the predictor before and after fine-tuning"),
    "signal-dropout": ("signal_dropout", "Test 2: ego_ft_v2 against the model trained without signals"),
    "summarize":      ("summarize",      "JSON and Markdown summary of a training run"),
    "plot":           ("plot",           "figures of a training run"),
    "figures":        ("figures",        "figures of the tests for the notes (docs/EgoVault/figures/)"),
    "watch":          ("watch",          "live terminal view of a running training job"),
}
