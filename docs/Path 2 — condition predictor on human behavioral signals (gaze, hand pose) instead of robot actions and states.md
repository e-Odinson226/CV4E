### Going further while being inspired by [V-JEPA 2](https://www.notion.so/V-JEPA-2-Self-Supervised-Video-Models-Enable-Understanding-Prediction-and-Planning-3690458a8dd4806f9e44c5e10828cadb?pvs=21) papers’ argument and hypothesis:

> Internet video shows the world without action labels — just sequences of states. V-JEPA 2 shows that learning a world model from this purely observational data is enough to develop physical understanding. Then a tiny amount of interaction data (62 hours) is sufficient to ground that understanding into controllable action. This is the data-efficiency argument for JEPA-style learning over generative video models and behaviour-cloning approaches.  
>   
> V-JEPA 2

From (our first step and hypothesis we worked on in Cluj): manipulating context window selection hypothesis, to:

**Feed feature as embedded vectors to the predictor, parallel to V-JEPA 2-AC**

In V-JEPA 2-AC, robot action tokens tell the predictor "the gripper moved Δx=+5cm." That action caused the cup to move. The predictor's future prediction correctly reflects this.

For a human, you do not have explicit action commands — but you have gaze and hand position, which are the human equivalent. Gaze says "I am attending to the cup." Hand velocity says "I am moving toward the cup." Together they are a soft action signal that shapes what will happen next. The mechanism is identical — just a different physical grounding of z.

Using exactly the same mechanism V-JEPA 2-AC uses for robot actions. A 7-dimensional robot action vector gets projected to 1024-dim and fed to the predictor as a z token. A 2-dimensional gaze point, a 126-dimensional hand keypoint vector, a 3-dimensional head orientation — all get projected to 1024-dim and fed in the same way. The predictor's attention then figures out which signals to use and how, purely from the prediction loss.

---

### The Z

**what is z — the question the predictor is asked**

In plain V-JEPA 2, z is just position — "predict what is at location (row, col, time)." The predictor has no idea _why_ the scene will change, it just infers the most likely state based on what it saw. That works for passive observation.

But for a robot, the future depends on what action the robot takes. The same current frame leads to completely different futures depending on whether the gripper opens or closes. So z needs to carry that causal information — the action becomes part of the conditioning variable.

![[image 1.png]]

These 7-dimensional vectors are tiny compared to the 1408-dimensional image tokens, so they cannot simply be concatenated as-is. Each gets its own small learned linear layer that projects it up to the predictor's 1024-dimensional hidden space. Now an action vector and an image patch vector speak the same language and can interact through attention.

**z is what tells the predictor where, when, and why**

The predictor does not generate representations blindly. It is conditioned on a variable z — the extra information shaping what the prediction should be. z is injected as mask tokens: learnable placeholder vectors whose positional embeddings encode the target location. Critically, z is not fixed. It can carry any information relevant to the future.

**Intuition:** think of z as the question the predictor is asked. "What will be at row 3, column 7 in 1 second?" The answer depends entirely on what the question says. Change z and you change what the predictor imagines. z is the handle by which intent enters the model.

**mask token construction**

shared learnable vector + 3D-RoPE(row, col, time) + optional Δ offset embedding. All projected to predictor hidden dim (1024) via learned linear layer.

**why the bottleneck matters**

The predictor (384-dim) is narrower than the encoder (1408-dim). This forces semantic compression — the predictor cannot copy context, it must reason about what should be there.

In plain V-JEPA 2, z is just position — "predict what is at location (row, col, time)." The predictor has no idea _why_ the scene will change, it just infers the most likely state based on what it saw. That works for passive observation.

But for a robot, the future depends on what action the robot takes. The same current frame leads to completely different futures depending on whether the gripper opens or closes. So z needs to carry that causal information — the action becomes part of the conditioning variable.

z is simply the collective name for everything you give the predictor beyond the encoder's context representations. It is the extra information that shapes what the prediction should be.

---

- In I-JEPA, z is just a spatial address — the mask token's positional embedding telling the predictor where in the image to predict. The prediction is purely inferential: "given what I can see, what probably exists at this location?"
- In V-JEPA 2, z extends to spatiotemporal address — row, column, and time. Same idea, just three dimensional.
- In V-JEPA 2-AC, z becomes genuinely causal. The action token aₖ is a 7-number vector encoding exactly how the robot's end-effector moved between the current frame and the next — three numbers for position change (Δx, Δy, Δz), three for orientation change (Δroll, Δpitch, Δyaw), one for gripper state change. The state token sₖ is another 7-number vector encoding where the end-effector currently is in absolute space.  
    These 7-dimensional vectors are tiny compared to the 1408-dimensional image tokens, so they cannot simply be concatenated as-is. Each gets its own small learned linear layer that projects it up to the predictor's 1024-dimensional hidden space. Now an action vector and an image patch vector speak the same language and can interact through attention.

---

from "where/when to predict" into "where/when to predict, given what the agent is already attending to and reaching for."

Gaze lands on an object 300–600ms before the hand moves toward it. The hand moves toward an object 500ms before contact. These are not random — they are **the body's own forward-planning signals** leaking out before the action executes.

---

Meta then built **V-JEPA 2-AC**, which conditions that predictor on robot actions and states — the model doesn't just predict what happens next, it predicts what happens next _given the effector's motion_.

Your project asks the human analogue: **can the predictor be conditioned on human behavioral signals instead of robot actions?** Specifically eye gaze (people fixate a target roughly 0.5–1 s before acting on it, so it's a leading indicator of intention) and hand pose (12D wrist+palm, the human counterpart of the robot end-effector). And does training with those signals produce representations that better encode how human actions — and ultimately intentions — unfold?

Two datasets, and the split between them matters:

- **HD-EPIC** — where the predictor is _trained_. It has Aria eye-tracking gaze and hand pose, plus rich text: recipe steps and fine-grained action descriptions. It's the only source with real behavioral signals.
- **EK100 (EPIC-KITCHENS-100)** — where evaluation happens. The standard action-anticipation benchmark, comparable to published baselines, but with _no_ gaze/hand annotations.

### What has been done

**The architecture work (mine).** You mapped the AC predictor tensor-by-tensor and adapted it — ==the robot action/state projectors were replaced with gaze (2D) and hand (12D) encoders, preserving the interleaving and causal attention.== You then fine-tuned that predictor on HD-EPIC P01–P07 with Ioana's projection-layer setup. That fine-tuned predictor is the artifact the rest of the project tests.

**The evaluation (Parsa).** He built a frozen-representation probe: freeze the encoder and predictor, feed their tokens to a deliberately weak attentive probe, train it on EK100 action anticipation, report recall@5. If the predictor's tokens carry anticipatory information beyond the encoder's, the probe should score higher.

The methodological care here is the valuable part. An early preview looked like a clear win (5.83% vs a 4.27% baseline) and was dismantled the same day: the treatment fed the probe 1764 tokens while the control fed 1568, and since the pooler cross-attends over all tokens, 196 extra keys can raise the score regardless of content. That forced a proper control family — encoder-only, encoder+zeros (does token count alone inflate?), and encoder+repeated-last-block (does prediction beat simple persistence?). Also discovered: the predictor was trained with 40% signal dropout, so evaluating with mask tokens on EK100 is _in-distribution_ — meaning the experiment measures what behavioral **training** left in the weights, not what live signals contribute.

**The result: a clean null at ~2 s.** Across conditions and seeds everything sits in 3.3–3.8%, with between-condition gaps (−0.02 to −0.06pp) smaller than seed noise. An independent paired A/B gave Δ ≈ +0.001 MSE — two different measurements agreeing on no effect. The interpretation was pre-registered rather than invented afterward: at 2 s, visual continuity alone predicts well, leaving behavioral conditioning little room.

**The forward-looking work.** Ioana's literature review argues the frozen V-JEPA 2 encoder is never supervised on visible context tokens, so it isn't forced to encode coherent object-level structure — meaning probing it for "intent" may be underpowered regardless of conditioning. Your direction went toward ==**language-aligned latent prediction**: keep working in embedding space, but align that space with language so the model predicts== _==concepts==_ ==rather than appearance==. VL-JEPA is the anchor, and its EK100 anticipation advantage widens with horizon (+1.5 at 1 s → +4.6 at 10 s) — evidence that semantic structure helps exactly where visual continuity runs out.

---