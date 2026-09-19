# Learned Remasking Architecture for Diffusion ASR / ST

The core idea is to **introduce a learned branch that detects errors directly on intermediate generations from the generator, then turns suspected erroneous tokens back into `[MASK]` for the generator to refine.** When the translation is already correct, this branch should learn to keep all tokens intact.

This component can be referred to as a **remasker**. The key premise of this approach is that **remasking decisions are learned from actual generator errors**, rather than relying solely on low-confidence heuristics.

---

### 1. Roles of the Two Branches

Let the acoustic feature input be $A$, the ground-truth target sequence be $Y$, and the intermediate generation at iteration $k$ be $X_k$.

```text
Generator:
    audio + current text canvas
    → predicts text tokens

Remasker:
    audio + intermediate generation X_k
    → decides KEEP or REMASK for each position
```

The remasker must condition on both the audio and the current candidate text. If it only observes the audio, it cannot determine which specific tokens the generator produced incorrectly.

Word-level illustrative example (assuming aligned positions):

```text
Reference:   I   bought  a   red   car   yesterday
Generator:   I   buy     a   blue  bike  today
Remask:      0   1       0   1     1     1

New Canvas:
             I   [MASK]  a   [MASK] [MASK] [MASK]
```

The generator then fills in these four masked positions, conditioning on both the audio input and the preserved tokens as context.

If the generator has already generated everything correctly:

```text
Remask:      0   0       0   0     0     0
```

**No tokens are forcibly remasked.**

---

### 2. Training Data Formulation

For each $(A, Y)$ pair in the existing dataset:

1. Run the generator to collect trajectory states throughout the decoding process (e.g., up to 20 iterations).
2. Sample one or more intermediate states $X_k$, including both early and late iterations.
3. Compare $X_k$ against $Y$ to derive ground-truth remasking targets.
4. Train the remasker on $(A, X_k)$ to predict these targets.

In the simplified setting where canvas length and token positions align with the reference label:

```python
remask_target[i] = int(candidate_ids[i] != label_ids[i])
```

$Y$ **is only used to construct targets and compute the loss**; it is never passed as input to the remasker. During inference, the remasker must detect errors without access to ground-truth labels.

Positions that are already `[MASK]` should be handled separately: they are already awaiting generation rather than generated tokens that require error detection. Special tokens such as BOS, task prefix tokens, and padding should also be excluded from standard remask targets.

This approach reuses the existing dataset while creating an additional supervision stream:

```text
(audio, reference text)
          ↓ generator rollout
(audio, intermediate candidate text, per-position error labels)
```

The key advantage is that the remasker is exposed to **errors genuinely produced by the generator**, including syntactically plausible mistakes or high-confidence errors.

---

### 3. Architecture: Full Diffusion Model vs. Classification Head

The remasker does not necessarily require a full standalone diffusion model. The desired behavior can be implemented using a binary classification head at each sequence position:

```python
p_remask = sigmoid(remask_head(hidden_states))
loss_remask = binary_cross_entropy(p_remask, remask_target)
```

It can share the backbone with the generator:

```text
Audio + text
     ↓
Backbone
     ├── vocabulary head → text token distribution
     └── remask head     → remasking probability
```

However, **the remasking evaluation pass must observe the candidate tokens actually generated**. If the generator has just replaced `[MASK]` with `bike`, the remasker must evaluate the canvas containing `bike`; we cannot assume the hidden states prior to token sampling have observed the newly selected token.

While parameter sharing via a common backbone improves efficiency, an additional forward pass may still be required to evaluate the updated canvas.

---

### 4. Proposed Training Strategy

- **Generator Warm-up**: Pretrain or warm up the generator so that rollout trajectories produce meaningful candidate sequences.
- **Freeze Generator Initially**: Keep the generator weights fixed during the initial phase of training the remasker.
- **Balanced Rollout Sampling**: Sample intermediate rollouts across diverse error rates, including completely accurate translations, so the remasker learns both when to intervene and when to preserve tokens.
- **Generator Fine-tuning on Remasked Canvases**: Next, train the generator on canvases remasked based on realistic error patterns, combined with standard diffusion training.
- **Dynamic Rollout Refresh**: Refresh rollout trajectories periodically as the generator improves to prevent the remasker from overfitting to outdated error distributions.

Training the generator on post-remask canvases is essential: simply improving error detection is insufficient if the generator tends to reproduce the same mistakes on those remasked slots.

At inference time, the iterative loop operates as follows:

```text
Generator fills in MASKs
        ↓
Remasker evaluates candidate text
        ↓
Preserve accepted tokens, remask suspected errors
        ↓
Generator refines remasked positions
```

The loop terminates when no `[MASK]` tokens remain and the remasker requests no further revisions, or when the iteration budget is exhausted. **Note that this is a heuristic stopping condition and does not guarantee absolute ground-truth correctness.**

---

### 5. Considerations for Streaming Translation

1. **"Different from reference" does not necessarily mean "incorrect translation".**
   For example, *"I bought a car"* and *"I purchased a car"* may both be valid. Direct token-matching targets effectively teach the model to *reconstruct the specific reference translation*. While this offers a clear and tractable baseline, it is not a semantic correctness evaluator. Furthermore, when candidate and reference lengths differ, sequence alignment is necessary; position-wise remasking alone does not naturally handle insertions or deletions.

2. **"No revision needed currently" does not mean "finalized permanently".**
   As incoming audio chunks arrive, previously generated text may require revision. To model this dynamic behavior, the remasker training data must include:

   ```text
   Old audio prefix → Generator produces draft
   New audio prefix + Old draft → Remasker re-evaluates
   ```

   If rollouts are only sampled from iterations conditioned on **full audio**, the model only learns error correction under complete acoustic context, rather than the ability to revise hypotheses as acoustic context expands incrementally.

   Additionally, when audio context is incomplete, the remasker should not be penalized for failing to match the complete reference text. It is critical to distinguish between **errors identifiable from audio heard so far** and **tokens that simply lack sufficient future acoustic evidence**.

In the current codebase, [streaming_diffusion_engine.py:204](file:///Users/mainhatduy/Workspaces/projects/aiai/diffusion-speech-recognition/src/model/streaming_diffusion_engine.py#L204) remasks a fixed percentage of tokens based on the lowest confidence scores. The proposed approach replaces this heuristic with a **learned error detector that dynamically selects anywhere from zero to multiple tokens for revision**. Similar concepts appear in existing literature such as [RemeDi](https://arxiv.org/abs/2509.23653).

The primary advantage for streaming translation is **enabling early hypothesis generation while retaining the flexibility to revise as more audio context arrives**. The remasker handles revision; length expansion mechanisms and finalization/commitment policies remain separate design concerns that need dedicated architectures.