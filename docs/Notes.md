
# Notes

## Motivation and Definitions

What is a world model and how does it work

Does conditioning JEPA-style models on behavior signals namely eye gaze and hand cues, improve their ability for action anticipation in egocentric view?

## Questions

- Does the mentioned features (eye gaze and hand tracking) contain information about future actions that user has the intention of, and can they guide the precitor?
 	- Gaze: Eye movements in everyday tasks are proactive and anticipate actions rather than only responding to stimuli. In object manipulation, gaze leads the hand to the landmarks where contact will happen and supports hand movement planning. [nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC1692006)[eneuro](https://www.eneuro.org/lookup/ijlink/YTozOntzOjQ6InBhdGgiO3M6MTQ6Ii9sb29rdXAvaWpsaW5rIjtzOjU6InF1ZXJ5IjthOjQ6e3M6ODoibGlua1R5cGUiO3M6NDoiQUJTVCI7czoxMToiam91cm5hbENvZGUiO3M6Njoiam5ldXJvIjtzOjU6InJlc2lkIjtzOjEwOiIyMS8xNy82OTE3IjtzOjQ6ImF0b20iO3M6MzY6Ii9lbmV1cm8vNC8yL0VORVVSTy4wMDI2LTE3LjIwMTcuYXRvbSI7fXM6ODoiZnJhZ21lbnQiO3M6MDoiIjt9)
 	- Hands: Liu et al. argue that intentional hand movement reveals critical information about the future activity, and model future hand motion as “motor attention”. [wizwand](https://www.wizwand.com/paper/68fa67f91e209131cea198b5)
 	- Counter-evidence exists. One human-robot study found that anticipatory gaze data did not improve intent prediction. On EPIC-KITCHENS-100, even ground-truth future head trajectory adds only 0.78 points, because the task is dominated by long-tail recognition. [aaai](https://aaai.org/?p=84717)[arxiv](https://arxiv.org/pdf/2605.20388)
 
- If they are informative about future actions, how long further? few seconds?
- is this information carried by gaze and hand cues, already extracted (or extractable) by the encode or a simple predictor without explicitly and additionally being fed to or processed by any part of the system?
-
- Benchmarks, datasets, other works? [[2-background]]
- how to process/use these cues to gain preformance in action anticipation?

## Approach

### Method 1

In method 1 we are take V-JEPA 2-AC as the foundation, a video model from Meta focused on robotics tasks specifically by conditioning a world model on robots actuation signals (action conditioned). Its predictor receives the robot's action and state as two extra tokens. We tried to project behaviour cues and feed this way to the same predictor and finetune it with new data.
it did not work apparently...

---

> [!NOTE] Stage 1
> In order to frame my research, I have propused some quetion and searhced for their answers.
