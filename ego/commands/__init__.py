"""
The commands, one module each. Run one from the repository root with

    python -m ego <command> [options]

Each module also runs on its own: python -m ego.commands.<module> [options].
"""

# name -> (module, one-line description)
COMMANDS = {
    "extract-csvs":   ("extract_csvs",   "extract the gaze and hand CSVs from the HD-EPIC MPS zips"),
    "train":          ("train",          "fine-tune the ego predictor (T2; with --signal-dropout 1.0, ego_sd1p0)"),
    "evaluate":       ("evaluate",       "paired error, signals hidden vs real, on any participants"),
    "gaze-probe":     ("gaze_probe",     "T4: can gaze be read from the frozen encoder's features?"),
    "control-probe":  ("control_probe",  "T5: the T4 probe with palm position as the target"),
    "sensitivity":    ("sensitivity",    "T6: does the prediction change when gaze changes?"),
    "attention":      ("attention",      "T7: how much attention goes to the gaze token?"),
    "weight-norms":   ("weight_norms",   "T8: did training shrink the gaze layer?"),
    "stock-vs-tuned": ("stock_vs_tuned", "T10, T11: the predictor before and after fine-tuning"),
    "signal-dropout": ("signal_dropout", "T11: ego_ft_v2 against the model trained without signals"),
    "summarize":      ("summarize",      "JSON and Markdown summary of a training run"),
    "plot":           ("plot",           "figures of a training run"),
    "watch":          ("watch",          "live terminal view of a running training job"),
}
