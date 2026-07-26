# General Idea

This research investigates whether a Vision-Language Joint Embedding Predictive Architecture (VL-JEPA) can anticipate user intent earlier than current Vision-Language Models (VLMs) by operating in latent representation space rather than token-generation space.

**Core question:** Can JEPA based models anticipate user intent earlier than VLMs by predicting latent representations of future actions rather than generating text descriptions of them?

The motivation rests on a fundamental asymmetry: VLMs predict intent by decoding token sequences — bottlenecked by language supervision and the need to surface abstract states as natural language — whereas JEPA predictors operate directly in a learned representation space that preserves semantic structure without language as an intermediary. ==The hypothesis is that this representational shortcut enables earlier, more reliable anticipation, particularly beyond the 1–2 second horizons where VLM performance degrades.==

The focus is egocentric video, where behavioral signals — gaze, hand position, hand shape — are directly observable and carry strong intent priors. These are the agent's forward-planning system leaking out before any action executes: gaze fixates on a target 300–800 ms before the hand moves; the hand pre-forms its grasp 200 ms before contact. The key question is whether feeding these signals into the JEPA predictor as conditioning tokens substantially improves intent anticipation, especially at longer horizons where visual context alone is insufficient.

---

### The Gaps Your Research Can Fill

**Gap 1 — No intent-level latent structure**

V-JEPA's latent space is learned purely from local spatiotemporal consistency. There is no objective that encourages goal-equivalent trajectories to cluster, or that separates =="what the scene looks like" from "what the agent intends to do."==

**Gap 2 — No hierarchical temporal abstraction**

Intent operates at multiple timescales simultaneously. Predicting "I will pick up the cup" (2s) and "I am making coffee" (5min) require representations at different levels of abstraction. Current JEPA variants are single-scale.

**Gap 3 — No egocentric inductive biases**

Hand-object interaction regions, gaze allocation, and ego-motion should be structurally privileged in the architecture or training objective, but generic video JEPA treats all patches equally.

**Gap 4 — No benchmark for latent intent evaluation**

Most LTA benchmarks evaluate discrete verb-noun predictions. There is no standard benchmark that evaluates the _==quality of the latent representation==_ ==for downstream intent inference==, which is what your model cares about.

---

# Sharpest research questions:

> **Does language-aligned latent space (VL-JEPA) encode semantic intent structure that pure spatiotemporal latent space (V-JEPA) does not — and does this difference manifest in anticipation performance, particularly at longer horizons?**

---

> **For goal-level intent inference from video, does language-aligned latent prediction (VL-JEPA) provide a structural advantage over token-generating VLMs, and does pure spatiotemporal prediction (V-JEPA) have any role at all?**

This is testable because there are benchmarks closer to your Level 3 goal than EK100. VL-JEPA's paper actually evaluates on **CrossTask** (procedural activity understanding — "what task is this person doing?") and **EgoExo4D** which involve exactly the kind of multi-step, goal-inferred activity recognition you're describing.

The cooking example → CrossTask / EgoExo4D territory.

The car keys example → commonsense goal inference, which would need a different benchmark or a custom evaluation.

---

**V-JEPA 2 and the EK100 benchmark have several limitations:**

- First, V-JEPA 2 does not fully solve EK100, there are failure cases where the model either gets the verb, the noun, or both wrong. We study the distribution of these failures in Appendix D.2.
- Second, we focus here on predicting actions with a 1 second anticipation time. The accuracy of V-JEPA 2 degrades when predicting at longer time horizons, see Appendix D.2.
- Third, the EK100 benchmark is limited to kitchen environments, with a closed well defined vocabulary, and we do not know how well V-JEPA 2 generalizes to other environments. This limits the utility and applicability of models trained on EK100.
- Lastly, actions in EK100 are chosen from a fixed set of categories, making it impossible to generalize to action categories not present in the training set.