#### Intent as Latent Alignment with Semantic Structure

==This is what VL-JEPA does — it injects intent-level structure by aligning visual representations with text embeddings in continuous embedding space. You get semantic clustering without token-by-token language decoding.==

**Formulation:** intent = the semantic region of the joint embedding space the current visual state occupies.

**Problem:** you've reintroduced language, just as a _scaffold_ rather than a _decoder_. This is the honest version of your bottleneck argument — language supervision _organizes_ the space, but doesn't require you to generate tokens at inference time. So the bottleneck is at training time, not inference time. That's a different claim than you started with.

---

---

## 1. "V-JEPA has no explicit world knowledge mechanism" — where does this come from?

The V-JEPA 2 paper (2506.09985) is explicit about what V-JEPA is trained on and how. The entire pretraining objective is:

> _"We pretrain V-JEPA 2 on a visual dataset that includes over 1 million hours of video. ==The self-supervised training task is based on mask denoising in representation space.=="_

And the loss function is simply:

> minimize ‖Pϕ(∆y, Eθ(x)) − sg(Eθ(y))‖₁

In plain terms: ==the model sees a video with some patches masked out, and it tries to predict the representations of those missing patches from the visible ones==. **That is the entire training signal.** There is no text, no action label, no goal description, no reward — nothing that says "this sequence of actions leads to this goal." The model only ever learns: _given what I can see, what do the hidden parts of this video probably look like?_

So when we said "no explicit world knowledge mechanism," it means exactly this: nothing in V-JEPA's training ever told it that tomatoes + garlic + pasta = cooking pasta sauce. It has no training signal that connects object co-occurrences to goals or intentions. ==Whatever semantic structure exists in its latent space emerged purely from predicting masked video patches across a billion frames of internet video — which is a very different thing from knowing about the world in a goal-directed sense.==

---

## 2. What is VL-JEPA and how does it work?

VL-JEPA (paper 2512.10942) extends the JEPA idea to connect vision and language. It has four components:

- **X-Encoder** — takes video frames as input, encodes them into visual embeddings (compact vectors, essentially a summary of what's in each video patch).
- **Y-Encoder** — takes the _text description_ (e.g. "a person is cooking pasta") and encodes it into a continuous embedding vector — a point in a semantic space, not a sequence of words.
- **Predictor** — the core. ==It takes the visual embeddings plus a textual query and tries to predict the embedding that the Y-Encoder would produce for the correct text answer.== It maps vision → predicted semantic embedding.
- **Y-Decoder** — a lightweight module that, only when you actually need text output, converts the predicted embedding back into readable words. It is completely bypassed during training.

The training objective is:

> ℒ_VL-JEPA = D(Ŝ_Y, S_Y)

Where SY​ is what the Y-Encoder produces for the correct text, and S^Y​ is what the Predictor predicted. The model is penalized for the distance between these two points in embedding space — not for getting the wrong words.

**Why this matters for your research:** during training, VL-JEPA is exposed to massive amounts of image/video-text pairs — action descriptions, captions, activity annotations. Every time it sees "video of someone gathering ingredients → text: making pasta," it pulls the visual embedding and the text embedding closer together in the shared space. Over billions of examples, this shapes the latent space so that visual scenes cluster near their semantically appropriate text descriptions. That is the mechanism through which world knowledge (in your cooking example sense) enters the latent space — through language-aligned training data, not from text generation at test time.

---

## What is "token generation" and "token-by-token decoding"?

This is the most important concept to understand for your entire research framing. The VL-JEPA paper explains it very clearly.

**A token** is a small piece of text — roughly a word or part of a word. The sentence "the agent is cooking pasta" might become tokens like: `["the", "agent", "is", "cook", "ing", "pasta"]`.

**How a standard VLM works:** it takes video + a question, and then generates the answer _one token at a time_. It first predicts "the", then given "the" it predicts "agent", then given "the agent" it predicts "is", and so on. This is what "autoregressive token-by-token decoding" means — each word depends on all previous words, and you cannot know the answer until you've generated the last token.

The VL-JEPA paper identifies two problems with this:

> _"VLMs are expensive to develop, because they are trained to generate responses Y to queries by capturing both task-relevant semantics with task-irrelevant surface linguistic features such as words choice, style or paraphrasing."_

> _"VLMs rely on autoregressive token-by-token decoding, which must be completed before revealing the underlying semantics of Y. This process introduces unnecessary latency and hampers the ability to update semantics dynamically in real time."_

**The concrete problem for your research:** imagine a model watching live video and trying to track whether the person's goal has changed. A VLM has to finish generating a complete sentence before it can tell you anything. VL-JEPA, by contrast, produces a continuous stream of embedding vectors — a point in semantic space at every moment — and only invokes the text decoder when the embedding changes significantly. The paper calls this _selective decoding_, and it reduces the number of decoding operations by ~2.85× without losing performance.

**The key contrast the paper draws:**

||VLM|VL-JEPA|
|---|---|---|
|Training target|Exact token sequence|Semantic embedding|
|Two semantically equivalent answers ("lamp turns off" / "room goes dark")|Appear orthogonal in token space (no shared tokens)|Appear as nearby points in embedding space|
|Inference|Must decode all tokens before knowing intent|Embedding is available immediately, text optional|
|World knowledge|Baked into LLM weights via language pretraining|Shaped into shared vision-language embedding space via alignment|

---

## Putting it Together for Our Research

V-JEPA: pure visual prediction, no language → no goal/intent structure.  
Standard VLM: language pretraining gives world knowledge, but accessing it requires generating tokens → latency, and bottleneck.  
VL-JEPA: language alignment during training shapes the embedding space with world knowledge → intent-relevant structure accessible immediately as an embedding, text decoding only when needed.

Our research question is essentially: **for goal-level intent prediction from video, does the VL-JEPA middle ground — world knowledge without token decoding — outperform both extremes?**

  

## Arguments

**Current problem:**  
It is classification against a discrete label set — we've reintroduced language supervision but just through the evaluation metric, instead of the model itself.

- **language alignment without decoding overhead lets the model reason over longer temporal contexts** before committing to a goal label.

  

---