# Spot-check of the assistant's grades

Rule in force (author, 2026-11-11): a required point counts when its main idea is stated correctly; specific details are required only when the question asks for them. Wrong facts and citations that don't support their claim make the answer wrong. Grades are of what was **sent**; withheld sentences don't count.

For each item write **agree** or **disagree + your grade**. Disagreements are recorded as overrules.

## 1. Q3 under h256k8

*Why check this one:* Grade changed under the less-strict rule (wavelength progression not given). Tests the rule.

**Question:** The Transformer does not use recurrence or convolution to process sequence order. How does it incorporate positional information into token representations? Why did the authors choose sinusoidal positional encodings over learned positional embeddings, despite observing nearly identical results?

**Key: required points**

1. Positional encodings are added to the input embeddings at the bottoms of the encoder and decoder stacks; same dimension d_model so they can be summed — cites: AIAYN §3.5
2. They use sine and cosine functions of different frequencies (PE(pos,2i)=sin(pos/10000^(2i/d_model)), PE(pos,2i+1)=cos(...)); wavelengths form a geometric progression from 2π to 10000·2π — cites: AIAYN §3.5
3. Learned positional embeddings produced nearly identical results (Table 3 row (E)) — cites: AIAYN §3.5, AIAYN Table 3
4. Sinusoidal was chosen because it may allow the model to extrapolate to sequence lengths longer than those seen in training — cites: AIAYN §3.5

**Answer as sent:**

> The Transformer adds “positional encodings” to the input embeddings at the bottoms of the encoder and decoder stacks so that each token’s representation includes information about its absolute or relative position, and these encodings have the same dimension d_model as the token embeddings, allowing them to be summed [AIAYN §3.5].
> In this work the authors use sine and cosine functions of different frequencies for the positional encodings, because they hypothesized that such sinusoidal functions would let the model easily learn to attend by relative positions, since a fixed offset k can be represented as a linear function of the encodings [AIAYN §3.5].
> They also experimented with learned positional embeddings and found that the results were nearly identical to the base model (see Table 3 row E) [AIAYN §6.2].
> Despite the similar performance, they chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than those seen during training [AIAYN §3.5].

**Assistant's grade:** `correct` (4/4 points). all main ideas; wavelength progression not given (not asked by Q3; strict: wrong); R3 cited 6.2, which supports it

**Your verdict:** Correct. 4/4 points are present and the answer is similar to the key.

---

## 2. Q7 under h256k8

*Why check this one:* Grade changed under the rule (codec numbers 24 kHz / Q=8 / 2048 not given).

**Question:** Mimi is the neural audio codec used by Moshi. Explain how Mimi combines semantic and acoustic information into its audio representation. What role do residual vector quantization and knowledge distillation play, and why is Mimi's causal design important for real-time speech generation?

**Key: required points**

1. Mimi is an autoencoder (SeaNet with Transformers) with a residual vector quantization bottleneck: 24 kHz audio to 12.5 Hz latents, Q = 8 quantizers with codebook size 2048 — cites: MOSHI §3.3.1, MOSHI §3.3
2. Semantic information is distilled from WavLM into the first quantizer level using a cosine distance — cites: MOSHI §3.3.2
3. Split RVQ: distillation into a single RVQ's first level hurt audio quality, so semantic information goes into a plain VQ and acoustic information into a 7-level RVQ in parallel, with outputs summed — cites: MOSHI §3.3.2
4. All convolutions are causal, so Mimi encodes and decodes in streaming fashion: an 80 ms frame yields a latent step that decodes to 80 ms of audio, which makes real-time generation possible — cites: MOSHI §3.3.1

**Answer as sent:**

> Mimi encodes audio by first passing a causal encoder into a latent space and then discretizing that latent representation with a Residual Vector Quantizer that applies vector quantization iteratively to the residuals of each quantizer level [MOSHI §3.3.1].
> During training, semantic information is distilled from a non‑causal speech model (WavLM) into the first quantizer, so that the first level of the RVQ carries phonetic‑discriminative tokens while the subsequent residual quantizers capture the remaining acoustic detail [MOSHI §3.3].
> Because a single RVQ with eight levels forced the first quantizer to trade audio quality for semantic discriminability, the authors introduced a split RVQ: a plain VQ receives the distilled semantic tokens and a parallel seven‑level RVQ handles the acoustic residuals, and their outputs are summed for reconstruction, preserving both semantic and acoustic content [MOSHI §3.3.2].
> All convolutions in Mimi’s encoder and decoder are causal, enabling the auto‑encoder to operate in a streaming fashion and thus supporting low‑latency, real‑time speech generation [MOSHI §3.3.1].
> The causal design also avoids the offline computation required by typical semantic tokens, making Mimi compatible with real‑time generation [MOSHI §3.3].

**Assistant's grade:** `correct` (4/4 points). all main ideas incl. split RVQ; codec numbers not given (not asked by Q7; strict: wrong)

**Your verdict:** Correct. 4/4 points are present and all main ideas included.

---

## 3. Q10 under h256k8

*Why check this one:* Grade changed under the rule; every claim is cited to HNET §1 (the introduction).

**Question:** The authors propose H-Net as an end-to-end hierarchical architecture that avoids conventional fixed-vocabulary tokenization. Describe the complete processing pipeline for byte-level language modelling, from the input sequence through the encoder, dynamic chunking, main network, dechunking and decoder. What does each stage do?

**Key: required points**

1. Encoder networks process the raw (byte-level) input sequence — cites: HNET §2.1.1
2. The chunking layer (routing module + downsampler) compresses the sequence into fewer vectors using dynamically learned boundaries — cites: HNET §2.1.1, HNET §2.2.1
3. The main network processes the compressed sequence and can itself be an H-Net, giving recursive multi-stage hierarchies — cites: HNET §2.1.1
4. The dechunking layer (smoothing module + upsampler) restores the original resolution, combined with a residual Linear(x) from the encoder side — cites: HNET §2.1.1, HNET §2.2.2
5. Decoder networks process the dechunked sequence back at the original resolution to produce outputs — cites: HNET §2.1.1

**Answer as sent:**

> The byte‑level input sequence is first passed through a small encoder network that produces an initial vector representation of the raw data [HNET §1].
> A dynamic‑chunking (DC) module then predicts boundaries between adjacent elements with a routing component and smooths the resulting discrete decisions with a smoothing component, thereby compressing the sequence into a shorter series of “chunks” that average about 4.5–5 bytes per chunk [HNET §1][HNET §1].
> These compressed chunk vectors are fed to the main network, which can be any standard token‑based architecture such as a Transformer or state‑space model and contains the bulk of the model parameters [HNET §1].
> After processing, a dechunking layer uses the stored boundary indicators to upsample the chunk representations back to the original resolution, applying smoothing to convert discrete chunks into interpolated vectors [HNET Figure 1].
> Finally, a decoder network operates on the upsampled, full‑resolution sequence to produce the model’s output predictions [HNET §1].

**Assistant's grade:** `correct` (5/5 points). every stage's main idea, cited to HNET 1 (intro), which states them; recursion and residual Linear(x) not given (not asked; strict: wrong)

**Your verdict:** Correct. 5/5 and all main ideas are present.

---

## 4. Q13 under h256k8

*Why check this one:* Grade changed under the rule (does not say only chunking changes sequence length).

**Question:** Compare the sinusoidal positional encoding mechanism proposed in Attention Is All You Need with the learned dynamic chunking mechanism proposed in Dynamic Chunking for End-to-End Hierarchical Sequence Modeling. What problem does each mechanism solve? How do they differ in terms of injecting sequence structure versus learning content-dependent segmentation?

**Key: required points**

1. Positional encoding solves the lack of order information in a model with no recurrence or convolution: fixed sinusoids of position are added to the embeddings — cites: AIAYN §3.5
2. Dynamic chunking learns content- and context-dependent segmentation end-to-end, replacing tokenization, and compresses the sequence into chunks — cites: HNET Abstract, HNET §2.2, HNET §2.2.1
3. Contrast: positional encoding is fixed (not learned), depends only on position, and leaves sequence length unchanged; dynamic chunking is learned, depends on content (similarity of adjacent representations), and changes sequence length — cites: AIAYN §3.5, HNET §2.2.1

**Answer as sent:**

> The sinusoidal positional encoding in the Transformer is introduced to inject information about the absolute or relative position of each token into a model that otherwise has no recurrence or convolution, thereby giving the model a sense of sequence order [AIAYN §3.5].
> The learned dynamic‑chunking mechanism in H‑Net is introduced to learn content‑ and context‑dependent boundaries that compress raw input vectors into meaningful chunks, effectively providing a data‑driven segmentation of the sequence rather than a fixed positional grid [HNET §1].
> Sinusoidal encodings supply a fixed, mathematically defined positional bias that is the same for every input and can extrapolate to longer lengths, while dynamic chunking learns where to split the sequence based on the actual content, creating variable‑length segments that reflect the structure of the data [AIAYN §3.5][HNET §1].
> Thus, sinusoidal encodings solve the problem of representing order in a position‑agnostic architecture, whereas dynamic chunking solves the problem of replacing handcrafted tokenization with an end‑to‑end, content‑aware segmentation [AIAYN §3.5][HNET §1].

**Assistant's grade:** `correct` (3/3 points). both purposes and the fixed/position vs learned/content contrast, each cited to its paper; length difference not stated (strict: wrong)

**Your verdict:** Correct. all 3/3 points are present and main ideas are included.

---

## 5. Q13 under h512k4

*Why check this one:* Wrong only because one contrast sentence cites HNET §1 for a claim about sinusoids.

**Question:** Compare the sinusoidal positional encoding mechanism proposed in Attention Is All You Need with the learned dynamic chunking mechanism proposed in Dynamic Chunking for End-to-End Hierarchical Sequence Modeling. What problem does each mechanism solve? How do they differ in terms of injecting sequence structure versus learning content-dependent segmentation?

**Key: required points**

1. Positional encoding solves the lack of order information in a model with no recurrence or convolution: fixed sinusoids of position are added to the embeddings — cites: AIAYN §3.5
2. Dynamic chunking learns content- and context-dependent segmentation end-to-end, replacing tokenization, and compresses the sequence into chunks — cites: HNET Abstract, HNET §2.2, HNET §2.2.1
3. Contrast: positional encoding is fixed (not learned), depends only on position, and leaves sequence length unchanged; dynamic chunking is learned, depends on content (similarity of adjacent representations), and changes sequence length — cites: AIAYN §3.5, HNET §2.2.1

**Answer as sent:**

> The sinusoidal positional encoding in the Transformer is introduced to give a model that lacks recurrence or convolution any information about the relative or absolute order of tokens, by adding fixed sine‑cosine vectors to the input embeddings so that sequence position is injected directly into the representation [AIAYN §3.5].
> The learned dynamic‑chunking mechanism in H‑Net is introduced to replace hand‑crafted tokenizers by learning, via gradient‑based optimization, where to place content‑ and context‑aware boundaries in the raw byte stream, thereby compressing the input into meaningful chunks [HNET §5] and [HNET §A.1.4].
> Sinusoidal encodings provide a uniform, position‑based bias that is the same for every token regardless of its content, whereas dynamic chunking learns segmentation decisions that depend on the actual data being processed, creating variable‑length chunks that reflect the structure of the input rather than a fixed positional grid [HNET §1].

**Assistant's grade:** `wrong` (3/3 points). the contrast sentence's claims about sinusoids are cited only to HNET 1, which doesn't support that half (unsupported citation)

**Your verdict:** Correct. But citation is wrong. So maybe partial correct.

---

## 6. Q8 under h512k4rr

*Why check this one:* Correct with padding tokens but no Whisper timestamps: is the main idea there?

**Question:** What is the Inner Monologue technique introduced in Moshi? Explain how time-aligned text tokens are incorporated into audio generation, why this improves linguistic quality, and how changing the delay between text and audio tokens allows the model to derive streaming ASR and TTS behaviour.

**Key: required points**

1. Inner Monologue: Moshi also models the text of its own speech as a time-aligned text stream inserted as a prefix to the semantic tokens — cites: MOSHI §3.4.4
2. Alignment: Whisper word-level timestamps are mapped to the 12.5 Hz frames, with PAD and EPAD special tokens filling the gaps between words — cites: MOSHI §3.4.4
3. It provides scaffolding that increases the linguistic quality of generated speech — cites: MOSHI §3.4.4, MOSHI Table 6, MOSHI Abstract
4. Delay controls which modality decides content: audio ahead of text (sample only text with real audio) gives streaming ASR; text ahead of audio gives streaming TTS — cites: MOSHI §3.4.4, MOSHI §C

**Answer as sent:**

> Inner Monologue is a new training and inference setup for audio language models that predicts time‑aligned text tokens before audio tokens, using the text tokens as a per‑timestep prefix to the semantic audio token stream [MOSHI §1].
> By feeding text tokens ahead of each audio frame, the model can first reason about the linguistic content and then generate the corresponding acoustic tokens, which the authors report “drastically improves the length and quality of generated speech” and “significantly improves the factuality and linguistic quality of generated speech” [MOSHI §1].
> The technique works by aligning the text stream with the audio frame rate using special padding tokens, so that each audio token is generated conditioned on the preceding text token at the same time step [MOSHI §C].
> Because the model receives the textual context early, it can produce more consistent and semantically accurate speech while still operating in a streaming fashion [MOSHI §1].
> Changing the delay between the two streams lets the same model be repurposed: delaying audio tokens by 2 seconds (giving text a look‑ahead) yields a streaming TTS system, while delaying text tokens by 2 seconds (giving audio a look‑ahead) yields a streaming ASR system [MOSHI §5.7][MOSHI §C].
> In both cases the model runs in full streaming mode with a fixed latency of about 2 seconds, simply by swapping which stream is ahead of the other [MOSHI §5.7][MOSHI §C].

**Assistant's grade:** `correct` (4/4 points). R2 main idea present (text aligned to the frame rate with padding tokens); Whisper not named (strict: wrong)

**Your verdict:** Correct. al main ideas present. 

---

## 7. Q6 under h256k8

*Why check this one:* Wrong with 4/5 points: only 'user stream replaced by real audio at inference' is missing.

**Question:** Moshi is described as a full-duplex speech-to-speech model that jointly models multiple streams. What are the two audio streams—whose speech is represented in each—and how are these distinct streams jointly modelled to support simultaneous listening and speaking, overlapping speech and interruptions?

**Key: required points**

1. The two audio streams are Moshi's own speech and the user's speech; they are represented as distinct parallel streams but jointly modelled in one multi-sequence, not generated as two independent dialogue systems. — cites: MOSHI §3.4.3, MOSHI Abstract, MOSHI §1
2. The acoustic delay is applied to both streams and they are concatenated into the jointly modeled multi-sequence — cites: MOSHI §3.4.3, MOSHI §3.4.4
3. There are no explicit turn boundaries: the model always listens and always generates (speech or silence), i.e. full-duplex — cites: MOSHI §3.4.4, MOSHI §1
4. At inference Moshi's stream is sampled while the user's predicted stream is replaced by the real user audio — cites: MOSHI §3.4.4
5. Because there is no speaker-turn segmentation, overlap and interruptions can be modeled — cites: MOSHI §1, MOSHI Abstract

**Answer as sent:**

> Moshi jointly models two audio streams: one stream corresponds to Moshi’s own generated speech and the other stream corresponds to the user’s input speech [MOSHI §1].
> The system treats these as separate autoregressive token streams that are processed in parallel, removing the need for explicit speaker‑turn modeling and enabling full‑duplex interaction [MOSHI §3.1].
> In practice the two streams are combined by applying an acoustic delay to each and concatenating them into a single representation, allowing the model to listen and speak simultaneously and to handle overlapping speech and interruptions [MOSHI §3.4.3].

**Withheld by the gate (not part of the answer):**

> ~~Moshi’s two audio streams are (1) the user’s incoming speech and (2) Moshi’s own generated speech, each represented as a separate autoregressive token stream [MOSHI §1]; the model jointly processes these parallel streams with a hierarchical multi‑stream Transformer that ingests both input and output~~

**Assistant's grade:** `wrong` (4/5 points). R4 (user stream replaced by real audio at inference) missing; a second summary sentence withheld for '2' [gate-FP, no grade effect]

**Your verdict:** Partial correct. idea is present but some poitns are missing.

---

## 8. Q14 under h512k4

*Why check this one:* Wrong for a wrong fact: 'tau = 2 corresponds to a 2-second audio delay'.

**Question:** The original Transformer uses causal masking in its decoder to prevent access to future output tokens. Moshi uses multiple parallel token streams with configurable delays between different token types. Compare these two mechanisms. What dependency does causal masking enforce in the original Transformer, and what purpose do the delay patterns serve in Moshi's multi-stream autoregressive generation? Distinguish the acoustic delay from the text-audio delay used by Inner Monologue.

**Key: required points**

1. Causal masking enforces that each output position depends only on earlier positions (positions < i), preserving the auto-regressive property — cites: AIAYN §3.1, AIAYN §3.2.3
2. Moshi's acoustic delay: a delay of 1 or 2 steps between semantic and acoustic tokens, which gave more stable, higher-quality generation and lets the Temporal Transformer model their inter-dependence — cites: MOSHI §3.4.2
3. Moshi's text–audio delay decides which modality determines content, giving streaming ASR (audio ahead) or TTS (text ahead) — cites: MOSHI §3.4.4
4. Moshi remains autoregressive over time steps: the Temporal Transformer conditions on previous steps — cites: MOSHI §3.4.1

**Answer as sent:**

> Causal masking in the original Transformer decoder blocks leftward information flow by masking out (setting to ‑∞) any attention connections to future positions, thereby enforcing that each output token can attend only to tokens that have already been generated and preserving the auto‑regressive property [AIAYN §3.2.3].
> In Moshi, instead of a single stream, several token streams (acoustic, semantic, and optionally text) are generated in parallel, and configurable delay patterns are inserted between them so that tokens of one type are only made available to the model after a fixed number of steps of another type [MOSHI §5.3].
> These delays serve to trade off latency against quality: a zero‑delay pattern ([0,0,0,0,0,0,0,0]) yields the minimal 80 ms latency at 12.5 Hz, while adding an 80 ms acoustic delay improves speech quality and a 240 ms delay gives further moderate gains [MOSHI §5.3][MOSHI Table 6].
> The acoustic delay specifically refers to the number of steps the acoustic token stream is held back (e.g., τ = 2 corresponds to a 2‑second audio delay in the joint sequence) so that the model can condition acoustic token prediction on earlier context while remaining causal [MOSHI Figure 9][MOSHI Figure 8].
> The text‑audio delay used by Inner Monologue is a separate offset that postpones the availability of text tokens relative to audio tokens, allowing text tokens to act as a prefix that conditions subsequent audio generation without violating causality; this delay is applied only when Inner Monologue is enabled and is distinct from the acoustic‑only delay [MOSHI §5.3][MOSHI Table 6].

**Assistant's grade:** `wrong` (1/4 points). 'tau = 2 corresponds to a 2-second audio delay' [wrong-fact: 2 steps = 160 ms]; R3 lacks ASR/TTS; R4 missing

**Your verdict:** Agree. It is wrong.

---

## 9. Q15 under h512k4rr

*Why check this one:* Wrong, partly for 'the decoder network finally restores the original resolution'.

**Question:** Both Moshi and H-Net use hierarchical processing, but their hierarchies serve different purposes. Compare: - Moshi's Temporal Transformer and Depth Transformer. - H-Net's encoder, dynamic chunking, main network, dechunking and decoder. What dimensions or structures are being decomposed in each architecture, and what computational or modelling problem is each hierarchy intended to address?

**Key: required points**

1. Moshi: the Temporal Transformer models the time axis, mapping past steps to a context vector z_s; the smaller Depth Transformer models the K sub-sequences (codebooks/streams) within one step, conditioned on z_s — cites: MOSHI §3.4.1
2. Moshi's problem: flattening the codebooks would need about 100 steps per second (30,000 for 5 minutes), which is computationally costly and incompatible with streaming — cites: MOSHI §3.4.1
3. H-Net: encoder, then chunking that compresses the sequence length with learned boundaries, then the main network on fewer vectors, then dechunking back to the original resolution; stages can nest — cites: HNET §2.1.1
4. After dechunking, H-Net's decoder network operates on the restored original-resolution sequence and produces the output; decoder layers combine coarse main-network representations with fine-grained encoder residuals. — cites: HNET §2.1.1, HNET §2.1.2, HNET §2.3
5. H-Net's problem: replace fixed tokenization with learned end-to-end segmentation and allocate compute by compressing low-information regions — cites: HNET Abstract, HNET §2.2
6. Contrast: Moshi decomposes the per-step depth (codebook) axis, while H-Net decomposes along sequence length — cites: MOSHI §3.4.1, HNET §2.1.1

**Answer as sent:**

> Moshi splits a long input sequence into two hierarchical dimensions: a “temporal” dimension of S steps that is processed by a large Temporal Transformer, and a “depth” dimension of K sub‑steps that is processed by a much smaller Depth Transformer whose layers have 6 layers, a hidden dimension of 1024 and 16 attention heads [MOSHI §3.4.1]; this two‑level factorisation lets the model keep the number of Transformer steps equal to S instead of K·S, reducing the computational cost of modelling very long flattened sequences while still modelling fine‑grained sub‑sequence dependencies with the depth‑wise transformer [MOSHI §3.4.1][MOSHI Figure 3].
> H‑Net, by contrast, hierarchically decomposes raw data into a series of progressively coarser “chunks”: an encoder network first compresses the raw input, a dynamic‑chunking (DC) module learns data‑dependent segment boundaries, the main network (which may itself be a full H‑Net) processes the compressed chunks, a de‑chunking layer reconstructs finer‑grained representations, and a decoder network finally restores the original resolution [HNET §2.1.1][HNET §1]; this hierarchy is designed to replace fixed‑size pooling with content‑aware segmentation, improving compression efficiency and modelling ability for long‑range structure while preserving autoregressive properties [HNET §2.4].

**Assistant's grade:** `wrong` (2/6 points). R2 only partly (S vs K*S steps, no streaming); R4 says the decoder restores resolution [wrong-fact]; R5, R6 missing

**Your verdict:** Agree with assistant's grade. Wrong.

---

## 10. Q16 under h512k4rr

*Why check this one:* A template refusal from the decide branch (no drafter call): counted as refused correctly.

**Question:** What is the exact power consumption in watts and GPU memory usage of Moshi when running continuous real-time dialogue on an NVIDIA RTX 4090?

**Key: required points**

- Expected: Refuse: no power draw or GPU memory figures for an RTX 4090 in the papers.

**Answer as sent:**

> The three papers in this library do not appear to cover this question, so I won't guess.

**Assistant's grade:** `refused_correctly`. refused at the decide branch (re-rank score below threshold), no drafter call

**Your verdict:** Refused correctly. agree with assistant's grade.

---
