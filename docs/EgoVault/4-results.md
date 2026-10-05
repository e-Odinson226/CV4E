---
type: report
status: running
created: 2026-10-05
updated: 2026-10-05
---

# 4. Results

Every test, who ran it, and what it showed. The tests are grouped by the hypothesis they
address. The hypotheses and their status are in [[1-introduction#^hypotheses|the
introduction]]. How each test is measured is in [[3-method]].

## Part 1. Do the signals help? (M)

### T1. Can the predictor learn with gaze and hand inputs?

Run by Parsa, before 16 June 2026. ^t1

Parsa built a first ego predictor on the smaller ViT-L model. It was trained on HD-EPIC
participant P01 for 20 epochs.

**Result.** The training loss fell from 1.207 to 1.019 (mean per epoch). It changed little
after epoch 5.

**Conclusion.** The predictor can learn with these inputs. This ViT-L model was only a
feasibility check. All later tests use the ViT-G model from T2.

### T2. Does the trained predictor do better with real signals?

Run by Erfan, 31 May – 1 June 2026. ^t2

**Question.** Is the prediction error lower with real gaze and hand than with the signals
hidden?

**Method.** Erfan fine-tuned the ViT-G predictor on P01–P07 for 3 epochs, with 30 clips per
recording. The result is `ego_ft_v2`. Δ was measured on the 96 P08 test clips after each
epoch.

**Result.**

| Epoch | MSE, signals hidden | MSE, real signals | Δ |
|---|---|---|---|
| 0 (before training) | 0.6153 | 0.6235 | −0.0082 |
| 1 | 0.4942 | 0.4933 | +0.0009 |
| 2 | 0.4900 | 0.4885 | +0.0015 |
| 3 | 0.4877 | 0.4866 | +0.0011 |

Erfan also started three longer runs: 8 epochs, 60 clips per recording, 240 test clips. Δ
was between −0.0066 and 0.0000 at every epoch they reached. Erfan stopped them for this
reason. None of them finished.

**Conclusion.** The 3-epoch model has a small positive Δ of +0.0011. This is about 0.2% of
the error. The longer runs did not show it. T11 shows what this Δ measures.

**Limits.** The 96-clip and 240-clip results are not directly comparable. No significance
test was run here. T10 adds one.

### T3. Does the trained predictor help action anticipation on EK100?

Run by Parsa, June–August 2026. Uses `ego_ft_v2`. ^t3

**Question.** Does `ego_ft_v2` give a classifier better information for predicting the next
action?

**Method.** EPIC-KITCHENS-100 (EK100) has an action anticipation benchmark. Each clip ends
1 s before the action starts. A small attention-based classifier (a probe) is trained on
frozen features. EK100 has no gaze or hand data, so the predictor receives its "no signal"
tokens. The probe received one of four inputs. 6b and 6c have the same token count as the
behavioral input. They are needed because extra tokens alone can raise the probe score.

**Result.** Action recall@5, best epoch, mean over seeds:

| Input | Tokens | Recall@5 |
|---|---|---|
| 6a: encoder output only | 1568 | 3.62% |
| 6b: encoder output + 196 zeros | 1764 | 3.60% |
| 6c: encoder output + last frame repeated | 1764 | 3.54% |
| Behavioral: encoder output + predicted next frame | 1764 | 3.61% |

The four results are within 0.1 percentage points of each other. These differences are
smaller than the differences between seeds.

**Conclusion.** The trained predictor does not help this task.

**Limits.**

- Only participant P01 was evaluated (870 test clips). The other participants' videos were
  not on disk, and the loader skipped them without a warning.
- The predictor was trained on 256-pixel frames. This pipeline uses 224-pixel frames.
- The signals are hidden, so this test cannot show the effect of live gaze. It shows what
  training left in the model.
- 6a and Behavioral used three seeds. 6b and 6c used two.

## Part 2. Is gaze already in the image? (R)

### T4. Can gaze be read from the frozen image features?

Run by Erfan, 28–30 July 2026. ^t4

**Question.** Ash asked whether the frozen encoder already contains the gaze direction. If
it does, a gaze input adds no new information.

**Method.** Single frames are encoded with the frozen encoder. A linear model (ridge
regression) predicts the gaze direction from these features. The target is the gaze at the
same moment, or up to 2 s later. The score is skill. A skill of 0 is no better than always
guessing the average gaze. A skill of 1 is perfect. Three ways of splitting the data into
training and test sets were compared.

**Result.** Skill for gaze at the same moment:

| Split | Test data | Skill |
|---|---|---|
| Random | frames close in time to training frames | 0.273 |
| Recordings | new recordings of the same people | 0.116 |
| Participants | new people, in their own kitchens | 0.001 |

Skill drops as the target moves later. In the recordings split it is 0.116 at 0 s, 0.055 at
0.5 s and 0 at 1 s.

**Conclusion.** For a new person, the image features contain no usable gaze information. So
the gaze input is not redundant.

**Limits.** The probe sees one frame, so it cannot use motion. It is linear on purpose. There
are no error bars yet. The participant split also changes the kitchen. T5 checks whether
this matters.

### T5. Is the T4 result specific to gaze?

Run by Erfan, 19 August 2026. ^t5

**Question.** In HD-EPIC, each person cooks in their own kitchen. So a new person also means
a new kitchen. The image features might fail on every target in a new kitchen. Then the T4
result would say nothing specific about gaze.

**Method.** T5 uses the same features, splits and probe as T4. Only the target changes: palm
position, from the same recordings. Gaze is scored again on exactly the same frames. A second
probe predicts which participant a frame comes from.

**Result.** Skill at the same moment:

| Target | New recordings | New people |
|---|---|---|
| Left palm position | 0.391 | 0.346 |
| Right palm position | 0.313 | 0.220 |
| Gaze | 0.116 | 0.001 |

On the frames used for the palm targets, gaze stays near zero for new people (0.082 and
−0.036). The participant can be identified from the features in 88.7% of cases. Chance is
14.3%.

**Conclusion.** The features differ a lot between people and kitchens. Palm position still
transfers to new people. Gaze does not. So the T4 result is specific to gaze, and R is
rejected.

**Limits.** Palm position may be easier to predict than gaze. This test cannot tell whether
gaze differs between people or is only harder to predict.

## Part 3. Does the model use the signals? (H2)

### T6. Does the prediction change when gaze changes?

Run by Erfan, 19 August 2026. Uses `ego_ft_v2`. ^t6

**Question.** If the model ignores gaze, changing the gaze input will not change the
prediction.

**Method.** The video is kept fixed and only the gaze input is changed. The test measures how
much the prediction moves, relative to its size. Two references are used. Running the same
input twice must give exactly 0. Replacing the whole video gives a large change for scale.

**Result.** Change in the prediction, on the 96 P08 clips:

| Change to the input | Change in prediction |
|---|---|
| None (same input twice) | 0 |
| Gaze rotated by 1° | 0.0012 |
| Gaze rotated by 10° | 0.013 |
| Gaze rotated by 45° | 0.081 |
| Gaze from another clip | 0.016 |
| Gaze hidden | 0.084 |
| Gaze and hand hidden | 0.036 |
| Video from another clip | 0.567 |

**Conclusion.**

- The prediction changes with gaze. Larger changes in gaze give larger changes in the
  prediction. The model uses gaze.
- The effect is small. Gaze from another clip moves the prediction 2.9% as much as a
  different video.
- Hiding gaze moves the prediction 5.2 times more than using another clip's gaze. The model
  reacts much more to whether gaze is present than to its value.

**Note.** Hiding gaze alone changes the prediction more than hiding both signals (0.084 vs
0.036). Signal dropout always hides both together. So the model never saw gaze hidden while
the hand signal was present.

### T7. How much attention goes to the gaze token?

Run by Erfan, 19 August 2026. Uses `ego_ft_v2` and the untrained model. ^t7

**Question.** The gaze token can only affect the image tokens through attention. If the model
ignores it, it gets about the same attention as any other token.

**Method.** Each frame has 258 tokens: gaze, hand and 256 image patches. Equal attention
would give the gaze token 0.39%. The test measures how much of each image token's attention
goes to the gaze token, in every block. It was run on the trained model and on the model
before training.

**Result.**

- Across the 24 blocks, image tokens give 2.4% to 26% of their attention to the gaze token.
  The average is 16% in the frozen blocks and 9% in the trained blocks.
- One attention head in the first block gives 99.96% of its attention to the gaze token.
- Before training, the frozen blocks gave the gaze token 38 times the equal share. After
  training, they gave 42 times.

**Conclusion.** The model gives the gaze token a lot of attention. This pattern comes from
robot pretraining, where the same position held the action token. Training changed it
little.

### T8. Did training shrink the gaze layer?

Run by Erfan, 19 August 2026. Uses `ego_ft_v2`. ^t8

**Question.** If training learned to ignore gaze, the weights of the gaze projection layer
would shrink toward zero.

**Method.** The size (norm) of each trained parameter is compared with its size at the start.
Other parameters trained with the same weight decay are the reference. Frozen parameters must
stay exactly the same.

**Result.**

- The gaze layer's weights grew by 15.7%. The hand layer's weights grew by 17.5%.
- Other trained parameters changed by less than 0.05%.
- The gaze layer's bias grew from 0 to 0.572, against a weight norm of 1.270. The bias adds
  the same value for every gaze input.

**Conclusion.** Training made the gaze layer larger. A large part of what it learned does not
depend on the gaze value. This matches T6: the model reacts to gaze being present more than
to its value.

### T9. Does the signal reach the output, and does it change decisions?

Run by Parsa, July–August 2026, for the paper. Uses `ego_ft_v2` on HD-EPIC P01. ^t9

**Question.** Does the signal change the model's output? Does it carry information that
matters for the action?

**Method.** Parsa ran five diagnostics with real signals on HD-EPIC participant P01. The
paper describes them in section 4.3.

**Result.**

| Diagnostic | Result | Reading |
|---|---|---|
| Output change, real vs hidden signals | 9.9% | The signal reaches the output. |
| Average error, real vs hidden (PEVA-1) | 0.055% lower with real signals, in 172 of 220 cases (p ≈ 3e-19) | A small, detectable effect. |
| Margin between correct and wrong candidates (PEVA-2) | same top choice in all 220 cases | The signal does not change the decision. |
| Attention on the gaze and hand positions | 28.6 and 35.4 times the equal share | Both positions get a lot of attention. |
| Change in that attention, real vs hidden | 0.1% or less | Attention does not depend on the signal's value. |
| Gaze and hand alone, predicting the action | 12.2%, vs 10.4% for the class prior | Slightly better than a guess based on class frequency. |

**Conclusion.** The model reads the signal. The signal changes the output a little and does
not change decisions. This agrees with T6–T8. Based on these results, the paper did not
build a stronger gaze pathway.

**Limits.** All diagnostics use one participant, P01. P01 is also in the model's training
data. T7 and this attention test use different setups, so their numbers are not comparable.

## Part 4. What is the signal worth? (M, H1)

### T10. How much of the training gain comes from the signals?

Run by Erfan, 19 August 2026. ^t10

**Question.** Fine-tuning does two things. The model adapts to kitchen video, and it learns
to use the signals. How much does each contribute?

**Method.** The 96 P08 clips are scored with the model before fine-tuning and with
`ego_ft_v2`. Each model is run with real signals and with three kinds of "no signal" input:
the learned "no signal" token, zeros, and the average signal. Differences are tested clip by
clip with a paired Wilcoxon test.

**Result.** MSE:

| Model | Real signals | "No signal" token | Zeros | Average |
|---|---|---|---|---|
| Before fine-tuning | 0.6183 | 0.6126 | 0.6158 | 0.6182 |
| `ego_ft_v2` | 0.4866 | 0.4877 | 0.4888 | 0.4873 |

- Fine-tuning lowered the error by 0.125 with the signals hidden. It did so on all 96 clips
  (p < 1e-16).
- The signals account for 0.0011 of the gain. This is about 0.9%.
- Δ depends on the type of "no signal" input: +0.0011 for the token, +0.0022 for zeros and
  +0.0007 for the average. All three are significant (p ≤ 0.0015).
- Before fine-tuning, real signals make the prediction worse by 0.0057. The new layers start
  random, so at first they add noise.

**Conclusion.** Most of the gain from fine-tuning is adaptation to the video. The signals add
very little. Every Δ must state which "no signal" input it uses.

### T11. Does a model trained without signals do as well?

Run by Erfan, 19 August 2026. ^t11

**Question.** In T2, Δ compares one model with and without its signals. That model was
trained to expect the signals, so hiding them may disrupt it. A fair test needs a model that
never had the signals.

**Method.** Erfan trained a second model, `ego_sd1p0`. All settings match `ego_ft_v2`, except
that signal dropout is 1.0. So this model never sees real gaze or hand. Both models were
scored on the same 96 clips.

**Result.**

| | `ego_ft_v2` (trained with signals) | `ego_sd1p0` (trained without) |
|---|---|---|
| Training loss after 3 epochs | 0.5066 | 0.5069 |
| MSE, signals hidden | 0.4877 | 0.4869 |
| MSE, real signals | 0.4866 | 0.5685 |

| Comparison | Δ | p |
|---|---|---|
| `ego_sd1p0` hidden − `ego_ft_v2` real | +0.0003 | 0.40 |
| `ego_ft_v2` hidden − `ego_ft_v2` real | +0.0011 | 0.0015 |
| `ego_sd1p0` hidden − `ego_ft_v2` hidden | −0.0008 | 0.015 |

**Conclusion.** The model trained without signals predicts as well as the model that uses
them. The difference is +0.0003, with p = 0.40. Training with gaze and hand gives no
measurable benefit. The +0.0011 from T2 is the cost of removing an input the model expects.
This is the main result of the project.

**Limits.** One seed per model, 3 epochs, 96 clips. This test uses the prediction error.
`ego_sd1p0` has not been run through the EK100 probe (T3) yet.

### T12. Do gaze and hand alone predict the action?

Run by Parsa, July–August 2026, for the paper. ^t12

**Question.** Without the predictor, how well do gaze and hand predict the next action,
compared with image features?

**Method.** Separate probes predict verbs and nouns on HD-EPIC participant P01, either from
gaze and hand or from image features. The final version tests verbs on held-out data at four
horizons. It compares the gaze-and-hand probe with a carefully tuned image probe and with the
verb class prior. The paper describes this in sections 4.6 and 4.7.

**Result.**

- First version: gaze and hand scored higher than image features on verbs (0.594 vs 0.480).
  They scored lower on nouns (0.231 vs 0.318).
- Final version, gaze-and-hand score minus image score for verbs: +0.035 at 1 s, +0.054 at
  3 s, +0.103 at 5 s and +0.121 at 10 s.
- At every horizon, the gaze-and-hand probe scores below the verb class prior, by 0.05 to
  0.09.
- The probes train on about 1,900 examples. On the training data, the image probe reaches a
  recall near 1.0 and the gaze-and-hand probe about 0.80.

**Conclusion.** Gaze and hand carry some information about motion (verbs) and less about
objects (nouns). This information is weak. It stays below a guess based on how often each
verb occurs. The paper names the small sample size as the main limit.

**Limits.** One participant. The sample size behind the final table is not confirmed. A note
in the paper source lists two possible values, about 3 times apart.

## Part 5. Is the model undertrained? (H3)

No test has addressed H3 yet. All results in Parts 3 and 4 come from models trained for 3
epochs with 30 clips per recording. A run at the intended size (8 epochs, 60 clips per
recording) has never finished.

Some results bear on H3:

- Fine-tuning already lowered the error by 0.125 (T10). So the training had an effect.
- Parsa's ViT-L run changed little after epoch 5 (T1).
- The longer runs in T2 had Δ at or below zero before they were stopped.

The planned test is in [[6-next-steps]].

## Next tests

The planned tests are in [[6-next-steps]].
