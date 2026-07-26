#### Daily notes

| ![](https://www.notion.so/icons/font_gray.svg)Name                                                                 | ![](https://www.notion.so/icons/calendar_gray.svg)date |
| ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------ |
| [[Load and Infrence]]                                                                                              | May 29, 2026                                           |
| [[Our work’s Readme]]                                                                                              | May 31, 2026                                           |
| [[General Idea]]                                                                                                   |                                                        |
| [[Path 1 — Attention framing through context block initialization]]                                                |                                                        |
| [[Path 2 — condition predictor on human behavioral signals (gaze, hand pose) instead of robot actions and states]] |                                                        |
| [[Path 3 — Language alignment]]                                                                                    |                                                        |

  
  

---

1. why predictor is fine tuned on HD-EPIC but probe is supposed to be trained on EK100? is it ok to do that?
2. why did this result in different dimesion being fed to the probe (1568 vs 1764)?
3. what is ‘pooler’ in [[Path 2 — condition predictor on human behavioral signals (gaze, hand pose) instead of robot actions and states]]?
4. what this:  
    “Also discovered: the predictor was trained with 40% signal dropout, so evaluating with mask tokens on EK100 is _in-distribution_ — meaning the experiment measures what behavioral **training** left in the weights, not what live signals contribute.”  
    means?