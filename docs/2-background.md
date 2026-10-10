---
type: report
status: running
created: 2026-10-05
updated: 2026-10-09
---

# 2. Background

## Benchmarks: anticipation protocols

- **The Dataset(s):** The specific data models will be tested against.
- **The Task:** What the model is supposed to do (e.g., "translate this text," "draw a bounding box around the pedestrian").
- **The Metric:** The mathematical formula used to grade the model (e.g., Accuracy, F1-score, BLEU score).
- **The Baselines/Leaderboard:** A record of how standard or state-of-the-art models score on this exact setup to provide a point of comparison

Research question: does conditioning JEPA-style models on eye gaze and hand cues improve egocentric action anticipation?

### 1. HoloAssist

166 hours of assisted manipulation tasks recorded with HoloLens 2. It is the only dataset here with an anticipation task, measured gaze and measured 3D hand pose.

- **Signals:** eye gaze and left and right 3D hand pose as per-frame files, synced to RGB.
- **Labels:** 1,887 fine-grained actions from 49 verbs and 165 nouns. 414 coarse-grained actions.
- **Splits:** official train, validation and test.

| Task                         | Setup                                                                                                      | Reference result                                                      |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Action anticipation          | Predict the next fine-grained action. Anticipation time and metric not confirmed.                          | Not read                                                              |
| Action recognition           | Classify trimmed fine-grained and coarse-grained actions. Top-1 and Top-5 accuracy.                        | Hand pose added to RGB improves over RGB. Numbers not read.           |
| Mistake detection            | Label each fine-grained action as correct or mistake. Active EgoVis 2026 challenge.                        | RGB 35.11. RGB plus hands 36.18. Hands only 40.19.                    |
| Intervention type prediction | Predict the instructor's intervention: confirm action, correct mistake or follow-up. Precision and recall. | RGB plus hands plus gaze: 48.31 precision, about 35 points above RGB. |
| 3D hand pose forecasting     | Forecast hand joints 0.5, 1 and 1.5 s ahead. Mean per-joint position error.                                | 9.80, 10.68 and 11.25 cm from hand input.                             |

### 2. MECCANO

20 videos of 20 subjects assembling a toy motorbike. It is small and covers one activity.

- **Signals:** gaze at 200 Hz. 89,628 hand bounding boxes. Depth.
- **Labels:** 61 actions from 12 verbs and 20 objects.
- **Splits:** 11 training videos, 9 validation and test videos.

|Task|Setup|Reference result|
|---|---|---|
|Action anticipation|Predict the next action at eight anticipation times from 2 s to 0.25 s. Recent work reports at 1 s. Top-1 and Top-5 accuracy.|At 1 s: majority class 27.21 Top-1. G3Ego 25.20 Top-1, 61.67 Top-5.|
|Next-active object detection|Detect and classify the objects of a future interaction. 48,024 annotated boxes.|Not read|
|Action recognition|Classify a trimmed segment into 61 actions.|Not read|
|Active object detection and recognition|Detect and classify objects in contact. 64,349 annotated boxes.|Not read|
|Human-object interaction detection|Detect active objects and the verb of the interaction.|Not read|
|Gaze estimation|Predict the 2D gaze location in each frame.|Not read|

Top-1 accuracy is not informative for anticipation here. Report Top-5 and a class-balanced metric.

### 3. EGTEA Gaze+

28 hours of kitchen activity. It has the most prior anticipation results of the six. The tasks are protocols from the literature. There is no official leaderboard.

- **Signals:** gaze on every frame. Hand masks on a subset of frames. Hand pose must be estimated.
- **Labels:** 106 actions from 19 verbs and 51 nouns.
- **Splits:** three official train/test splits.

|Task|Setup|Reference result|
|---|---|---|
|Action anticipation|Predict the next action 0.5 s ahead on split 1. Top-1 and class-mean Top-1 accuracy.|See the table below.|
|Long-term anticipation|Predict the set of actions in the rest of the video. Multi-label mAP.|Not read|
|Action recognition|Classify a trimmed clip into 106 actions. Top-1 and mean class accuracy.|G3Ego, average of 3 splits: 56.49 mean class accuracy.|
|Gaze estimation|Predict the gaze heatmap on observed frames. F1.|GLC: 44.8 F1|
|Gaze anticipation|Predict the gaze heatmap 0.5 s ahead. F1.|SAGE sets the baseline. Number not read.|

Action anticipation at 0.5 s:

|Method|Extra signal|Top-1|Class-mean|
|---|---|---|---|
|FHOI|Predicted hand motion|36.6|25.3|
|AVT|None|43.0|35.2|
|AFFT|None|42.5|35.2|
|InAViT|Hand-object interaction regions|67.8|58.2|
|SAGE|Predicted gaze|Not found|58.4|

InAViT is 24.8 points above AVT on Top-1. Confirm that its splits and protocol match before using it as the reference. AFFT is averaged over three splits.

### 4. EgoExoLearn

120 hours of procedural tasks, with egocentric recordings and demonstration videos. It has gaze and no hand signal, so it tests the gaze half of the claim.

- **Signals:** calibrated gaze on all egocentric videos.
- **Labels:** fine-level and coarse-level actions. The anticipation task uses 19 verbs and 31 nouns.
- **Splits:** train, validation and test.

|Task|Setup|Reference result|
|---|---|---|
|Cross-view action anticipation|Predict the verb and noun of the next fine-level action 1 s ahead. An ego-only setting exists, so cross-view training is optional. Verb and noun accuracy.|Ego-only, validation: full frame 33.4 verb and 37.6 noun. Gaze crop 40.9 and 52.3. Centre crop 33.4 and 38.8.|
|Cross-view action planning|Predict the next 8 coarse-level steps from 27 classes.|Not read|
|Cross-view association|Match egocentric and demonstration clips of the same action. Accuracy.|Validation: 44.15 without gaze, 48.35 with gaze.|
|Cross-view referenced skill assessment|Rate execution skill with a demonstration as reference. Accuracy.|79.875 without gaze, 81.27 with gaze.|
|Cross-view referenced captioning|Caption an egocentric clip with a demonstration as reference.|Not read|

In the anticipation task the gaze crop helps and the centre crop does not. This indicates the gain comes from the gaze location. The association and skill assessment numbers come from a results aggregator.

### 5. Ego-Exo4D

1,286 hours of skilled activities recorded with Aria glasses and static cameras. There is no official anticipation task.

- **Signals:** gaze on all sequences. Manual 3D hand pose on 68K frames only. Automatic hand annotations are about 60 times larger.
- **Labels:** 689 keysteps across 17 procedural activities. Atomic action descriptions.
- **Decision needed:** where dense hand input comes from. The options are the automatic annotations or a hand pose estimator.

Official tasks:

|Task|Setup|
|---|---|
|Procedure understanding|Given a segment and its history, infer prior keysteps, optional or mistaken steps, missing keysteps and next valid keysteps. Closest official task to anticipation. Active EgoVis 2026 challenge.|
|Fine-grained keystep recognition|Classify a trimmed egocentric clip into 689 keysteps.|
|Energy-efficient keystep recognition|Detect keysteps online from audio, IMU and RGB under a 20 mW budget.|
|Ego pose|Estimate 3D body pose and 3D hand pose from egocentric video. MPJPE. Body pose is an active EgoVis 2026 challenge.|
|Proficiency estimation|Classify the wearer's skill level. Localize good and poor execution.|
|Ego-exo correspondence|Match an object mask across ego and exo views.|
|Ego-exo translation|Synthesize the ego clip from an exo clip.|

Tasks defined by other papers:

| Task                                       | Setup                                                                                                                                                                              | Reference result                                                                   |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Atomic-action anticipation (**TrajPilot**) | Predict the following atomic actions from a start clip, for sequences of 3 to 8 steps. 8,472 labels. 140,413 train and 11,425 validation segments, split by take. Future Recall@1. | Frozen V-JEPA 2.1 features, 3 steps: 26.20. With predicted head trajectory: 30.56. |
| Keystep planning (TrajPilot)               | Fill the steps between a start clip and a goal clip. 375 keystep labels. Mid-step mean accuracy.                                                                                   | 16.75 averaged over horizons.                                                      |
| 3D hand forecasting (EgoH4)                | Forecast wrist trajectory and hand pose. 156K train and 34K test sequences. ADE and MPJPE.                                                                                         | 3.4 cm lower ADE and 5.1 cm lower MPJPE than its baseline.                         |
| Future state prediction (EgoAgent)         | Predict the future scene state from egocentric frames and body pose. Top-1 accuracy.                                                                                               | 16.28 points above prior methods.                                                  |

### 6. HD-EPIC

41 hours in 9 kitchens recorded with Aria glasses. It is a validation-only dataset with no training split. There is no anticipation task.

- **Signals:** gaze. 7.7M hand masks. Hand tracking files with wrist and palm poses.
- **Labels:** 59,454 fine-grained action segments.
- **Gaze timing:** objects are fixated 4.0 s before pick-up and 2.6 s before placement, on average.
- **Two ways to use it for anticipation:**
    1. Train on EPIC-KITCHENS-100 with predicted gaze and test on HD-EPIC with measured gaze.
    2. Cross-validate across the 9 kitchens.

|Task|Setup|Reference result|
|---|---|---|
|Video question answering|26K multiple-choice questions in 7 categories: recipes, ingredients, nutrition, fine-grained actions, 3D perception, object motion and gaze. Active EgoVis 2026 challenge.|Gemini Pro: 38.5%|
|Action recognition|Models trained on EPIC-KITCHENS-100, tested on HD-EPIC.|Best model: 51% verb, 37% noun, 24% action.|
|Sound recognition|Models trained on EPIC-Sounds, tested on HD-EPIC. Top-1 accuracy.|Best model: 31.9|
|Long-term video object segmentation|Segment objects through long sequences.|Not read|

### 7. IndustReal

5.8 hours of toy assembly recorded with HoloLens 2: 84 recordings of 27 participants. It has both signals measured, like HoloAssist, and is much smaller. There is no anticipation task.

- **Signals:** gaze, hand tracking and head pose from the headset at 10 fps. Depth at 5 fps.
- **Labels:** 75 fine-grained actions with 9,273 instances. Actions last 1.9 s on average, with about 110 per video.
- **Splits:** by participant: 12 train, 5 validation, 10 test.
- **Gap:** the paper reports no experiment with gaze or hand input.
- **Use for anticipation:** you define the protocol on the action segments.

|Task|Setup|Reference result|
|---|---|---|
|Action recognition|Classify a clip into 75 actions. Top-1 and Top-5 accuracy.|MViTv2 ensemble of RGB, visible light and stereo: 66.45 Top-1, 88.43 Top-5.|
|Assembly state detection|Detect the assembly state in a frame. 22 correct states and 27 error states. mAP.|0.838 on annotated frames. 0.641 on entire videos.|
|Procedure step recognition|Recognize each completed step in order, including steps done wrongly. F1 and average delay.|0.883 F1 with a 22.4 s delay.|

### 8. EgoMe

82.8 hours: 7,902 pairs of a demonstration video and an egocentric video of a person imitating it. 41 scenarios, 37 participants, eye-tracking glasses. It is close to EgoExoLearn in setting and tests the gaze half of the claim.

- **Signals:** binocular gaze at 120 Hz in both views. IMU. No hand pose labels.
- **Labels:** 184 video-level action classes. 69,880 timed fine-level descriptions.
- **Gap:** no published anticipation result uses its gaze. The anticipation benchmark is cross-view only.

Official tasks:

|Task|Setup|Reference result|
|---|---|---|
|Exo-ego gaze prediction|Predict the gaze heatmap in one view, with or without the other view. F1.|Egocentric target: 42.11. Demonstration target: 56.52.|
|Exo-ego cross-modal retrieval|Retrieve the matching video across views. Recall@1.|17.61 from demonstration to egocentric.|
|Imitative action assessment|Judge the imitation against the demonstration. AP@30.|42.74|
|Imitative action reorganization|Ranking accuracy.|21.25|
|Coarse-level video understanding|Caption the whole video. CIDEr.|49.02|
|Fine-level procedural understanding|Caption each step with its time span. CIDEr.|56.63|

Tasks defined by other papers:

| Task                                  | Setup                                                                                                                                                                                                                                      | Reference result                                                                                 |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| Action anticipation (EgoMe-anti)      | Observe 2 s and predict the noun and verb of the action after a 1 s gap. 35 nouns and 29 verbs. Train in one view and adapt to the other at test time. 8,939 egocentric and 8,336 demonstration training samples. Class-mean Top-5 recall. | Demonstration to egocentric, no adaptation: 71.94 noun, 32.46 verb. With DCPGN: 79.03 and 43.84. |
| 3D hand pose forecasting (EgoMe-pose) | Forecast 10 future steps at 15 fps. The hand pose labels are estimated by a pose model, not annotated. 6,067 train, 1,344 validation and 2,648 test episodes. MPJPE.                                                                       | Exo2EgoPose: 49.44 mm. Best baseline: 56.06 mm.                                                  |

---

### 9. Ego4D

3,670 hours of daily-life video from 931 camera wearers at 74 locations. Its anticipation benchmarks are large and have V-JEPA results. They have no measured gaze and no hand pose, so both signals must be estimated.

- **Gaze:** recorded at two of the collection sites only, Indiana University and Georgia Tech, mostly with Pupil Labs Invisible glasses. The paper lists 45 hours. Gaze papers use 31 to 33 hours from 80 participants.
- **What the gaze videos show:** social activities. Group games, music, puzzles, eating and drinking. They contain little object manipulation.
- **Gaze and anticipation labels together:** the paper does not link the gaze videos to the forecasting benchmark. The overlap is not checked. The scenarios suggest it is small.
- **Hands:** 2D boxes for hands and active objects at annotated key frames of each interaction. The key frames are the precondition, contact and post frames, and 0.5, 1 and 1.5 s before the precondition frame. No 3D hand pose. Dense hand input needs a detector.
- **Annotation coverage:** hands and objects benchmark 196.2 h in 88,585 clips. Short-term anticipation 243 h in version 2.

Forecasting tasks:

|Task|Setup|Reference result|
|---|---|---|
|Short-term object interaction anticipation (STA)|From video up to time t, detect the next-active object in the last frame and predict its noun, the verb and the time to contact. Version 2: 98,276 train, 47,395 validation and 19,780 test samples. 128 nouns and 81 verbs. Top-5 mAP over all fields. Active EgoVis 2026 challenge.|See the table below.|
|Long-term action anticipation (LTA)|Observe 8 action segments. Predict the next 20 actions as verb and noun pairs, in 5 candidate sequences. 117 verbs and 521 nouns. 3,472 clips. Edit distance, lower is better. Not on the EgoVis 2026 challenge list.|Best 2025 test entry: 0.8493 action, 0.6346 verb, 0.5986 noun. No JEPA result found.|
|Future hand prediction|Predict the 2D positions of both hands at future key frames. Mean and contact-frame displacement in pixels.|Paper baseline, test: 56.37 px left hand and 56.17 px right hand at the contact frame.|
|Locomotion prediction|Predict future ground-plane trajectories of the camera wearer.|Not read|

STA, version 2 validation, Top-5 mAP over all fields:

|Method|Backbone training|Hand cue|Result|
|---|---|---|---|
|StillFast|End to end|None|3.96|
|STAformer|End to end|None|4.90|
|STAformer with affordances and interaction hotspots|End to end|Hand trajectories from a detector|5.04|
|STAformer, best reported variant|End to end|Hand trajectories from a detector|5.67|
|V-JEPA 2 ViT-g|Frozen encoder|None|6.02|
|V-JEPA 2.1 ViT-g|Frozen encoder|None|6.75|
|V-JEPA 2.1 ViT-G|Frozen encoder|None|7.71|

- **V-JEPA setup:** the frozen encoder reads a low-resolution clip (8 frames over 0.5 s) and the high-resolution last frame. A four-layer attentive probe and a Faster R-CNN style detection head are trained on top.
- **Test numbers disagree across sources.** STAformer reports 6.75 on the test set. The EgoVis 2026 page lists a previous winner at 7.21. The 2026 winner report lists first place at 5.40. Use the validation split for comparisons.

Other official tasks:

|Task|Setup|
|---|---|
|Point-of-no-return temporal localization|Find the frame where an object state change begins.|
|State change object detection|Detect the object that changes state. STAformer fine-tunes its hand-object detector on these boxes.|
|Object state change classification|Classify whether a clip contains a state change.|
|Episodic memory|Visual queries, natural language queries and moment queries. Natural language queries is an active EgoVis 2026 challenge.|
|Goal-Step|Localize the video segment of a described procedural step. Active EgoVis 2026 challenge.|
|Audio-visual diarization|Track speakers, detect the active speaker, transcribe speech.|
|Social interactions|Detect who looks at the camera wearer and who talks to the camera wearer.|

Tasks defined by other papers:

|Task|Setup|Reference result|
|---|---|---|
|Gaze estimation (GLC)|Predict the gaze heatmap on 5 s clips from the gaze subset. F1.|GLC: 43.1|
|Gaze anticipation (CSTS)|Predict the gaze heatmap on future frames, from video and audio. F1.|CSTS is 1.9 above GLC.|
|Future event description (Gaze-VLM)|On the gaze subset: observe 5 s and describe the events 2 s ahead in text. Reference captions are generated by GPT-4V. Gaze is used in training only, as an attention target. Sentence similarity score.|OpenFlamingo: 0.6525 without gaze training, 0.7505 with it.|
|Procedure planning on Goal-Step (TrajPilot)|Fill the steps between a start clip and a goal clip. 310 step labels. Mid-step mean accuracy.|6.86 on frozen V-JEPA 2.1 features.|

Evidence on hand cues from Ego4D:

- STAformer's interaction hotspot module forecasts hand trajectories and reweights detections. On version 1 validation it raises noun mAP from 21.71 to 23.63 and overall mAP from 3.53 to 3.66.
- The best 2025 LTA entry encodes hand-object interaction regions separately from the full frame.

Use for the study:

1. STA version 2 validation allows a direct comparison with V-JEPA 2.1 under the same frozen-encoder protocol.
2. Gaze comes from a gaze estimator. One trained on the Ego4D gaze subset matches the cameras and differs in activity type.
3. Hands come from a hand-object detector fine-tuned on the state change object detection boxes, as STAformer does.
4. A null result is ambiguous here. The signal may not help, or the estimate may be too noisy.

#### Open items

- HoloAssist: anticipation time, observation length, metric and baseline numbers.
- EGTEA Gaze+: which splits InAViT and SAGE report on.
- HD-EPIC: mapping of its action labels to EPIC-KITCHENS-100 verb and noun classes.
#### Sources

- HoloAssist paper: [https://arxiv.org/abs/2309.17024](https://arxiv.org/abs/2309.17024)
- HoloAssist release repository: [https://github.com/Ember-HoloAssist/holoassist-release](https://github.com/Ember-HoloAssist/holoassist-release)
- MECCANO project page: [https://iplab.dmi.unict.it/MECCANO/](https://iplab.dmi.unict.it/MECCANO/)
- MECCANO paper: [https://arxiv.org/abs/2209.08691](https://arxiv.org/abs/2209.08691)
- G3Ego: [https://arxiv.org/abs/2608.20157](https://arxiv.org/abs/2608.20157)
- Predictive Action Guidance (MECCANO anticipation times): [https://arxiv.org/abs/2501.05108](https://arxiv.org/abs/2501.05108)
- AVT: [https://arxiv.org/abs/2106.02036](https://arxiv.org/abs/2106.02036)
- AFFT: [https://arxiv.org/abs/2210.12649](https://arxiv.org/abs/2210.12649)
- ARR (EGTEA Gaze+ comparison table): [https://arxiv.org/abs/2408.02769](https://arxiv.org/abs/2408.02769)
- InAViT supplement: [https://openaccess.thecvf.com/content/WACV2024/supplemental/Roy_Interaction_Region_Visual_WACV_2024_supplemental.pdf](https://openaccess.thecvf.com/content/WACV2024/supplemental/Roy_Interaction_Region_Visual_WACV_2024_supplemental.pdf)
- SAGE: [https://arxiv.org/abs/2607.04017](https://arxiv.org/abs/2607.04017)
- GLC: [https://arxiv.org/abs/2208.04464](https://arxiv.org/abs/2208.04464)
- EgoExoLearn: [https://arxiv.org/abs/2403.16182](https://arxiv.org/abs/2403.16182)
- EgoExoLearn results aggregator: [https://vercel.hyper.ai/en/papers/2403.16182](https://vercel.hyper.ai/en/papers/2403.16182)
- Ego-Exo4D: [https://arxiv.org/abs/2311.18259](https://arxiv.org/abs/2311.18259)
- Ego-Exo4D release note: [https://ai.meta.com/blog/ego-exo4d-video-learning-perception/](https://ai.meta.com/blog/ego-exo4d-video-learning-perception/)
- EgoVis 2026 challenges: [https://egovis.github.io/cvpr26/](https://egovis.github.io/cvpr26/)
- TrajPilot: [https://arxiv.org/abs/2605.20388](https://arxiv.org/abs/2605.20388)
- EgoH4: [https://arxiv.org/abs/2504.08654](https://arxiv.org/abs/2504.08654)
- EgoAgent: [https://arxiv.org/abs/2502.05857](https://arxiv.org/abs/2502.05857)
- HD-EPIC paper: [https://arxiv.org/abs/2502.04144](https://arxiv.org/abs/2502.04144)
- HD-EPIC project page: [https://hd-epic.github.io/](https://hd-epic.github.io/)
- IndustReal: [https://arxiv.org/abs/2310.17323](https://arxiv.org/abs/2310.17323)
- EgoMe: [https://arxiv.org/abs/2501.19061](https://arxiv.org/abs/2501.19061)
- DCPGN (EgoMe-anti, EgoExoLearn cross-view anticipation): [https://arxiv.org/abs/2603.09798](https://arxiv.org/abs/2603.09798)
- Exo2EgoPose (EgoMe-pose): [https://arxiv.org/abs/2607.15890](https://arxiv.org/abs/2607.15890)
- Ego4D paper: [https://arxiv.org/abs/2110.07058](https://arxiv.org/abs/2110.07058)
- Ego4D gaze documentation: [https://ego4d-data.org/docs/data/gaze/](https://ego4d-data.org/docs/data/gaze/)
- Ego4D forecasting documentation: [https://ego4d-data.org/docs/benchmarks/forecasting/](https://ego4d-data.org/docs/benchmarks/forecasting/)
- V-JEPA 2.1 (STA results and setup): [https://arxiv.org/abs/2603.14482](https://arxiv.org/abs/2603.14482)
- STAformer: [https://arxiv.org/abs/2406.01194](https://arxiv.org/abs/2406.01194)
- VISTA (EgoVis 2026 STA winner report): [https://arxiv.org/abs/2605.20901](https://arxiv.org/abs/2605.20901)
- Ego4D LTA Challenge 2025 winner report: [https://arxiv.org/abs/2506.02550](https://arxiv.org/abs/2506.02550)
- ICVL (LTA setup and label space): [https://arxiv.org/abs/2505.01713](https://arxiv.org/abs/2505.01713)
- CSTS (gaze anticipation): [https://arxiv.org/abs/2305.03907](https://arxiv.org/abs/2305.03907)
- Gaze-VLM: [https://arxiv.org/abs/2510.21356](https://arxiv.org/abs/2510.21356)
- Gaze and Set-of-Mark in VLLMs (HD-EPIC gaze interaction anticipation): [https://arxiv.org/abs/2604.03667](https://arxiv.org/abs/2604.03667)

---

---

---
## Methods

