# Learned remasking

The implementation adds an optional binary `remask_head` to both the training model
and the Hugging Face model. It shares the audio-conditioned text backbone. A fresh
forward pass evaluates **the generated candidate**, after sampling, and produces
per-position error logits. Existing configurations keep the original decoder and
checkpoint parameter layout (`learned_remasking=false`).

## Training stages

1. Warm up the generator with existing diffusion training.
2. Load that checkpoint with `finetune_from_model`, set `learned_remasking=true`
   and `remask_training_stage="detector"`. Only the remask head is trainable;
   generator dropout is disabled.
3. Start a new run from the detector checkpoint with
   `remask_training_stage="joint"`. The objective combines standard diffusion loss,
   remask classification, and generator reconstruction at positions selected by
   the learned remasker. The pretrained audio encoder remains frozen.

Rollouts are regenerated for every batch, so there is no stale trajectory cache.
They start with all content masked on an oracle-length canvas, then collect
`remask_rollout_steps` greedy candidates using a decreasing confidence-remasking
schedule. Sampling chooses among available zero, low and high error-rate strata,
then a timestep within that stratum. Completely correct generator candidates are
included when present; reference text is not injected as a synthetic clean draft.
The classifier uses class-balanced BCE with logits, including all-KEEP examples
and differentiable zero loss when there are no eligible positions.

| Setting | Default | Meaning |
| --- | --- | --- |
| `learned_remasking` | `false` | Add the head and use learned decoding |
| `remask_training_stage` | `"disabled"` | `disabled`, `detector`, or `joint` |
| `remask_rollout_steps` | `20` | Fresh rollout states per batch |
| `remask_threshold` | `0.5` | Error probability required to request revision |
| `remask_loss_weight` | `1.0` | Classification loss multiplier |
| `remask_reconstruction_weight` | `1.0` | Post-remask reconstruction multiplier |
| `remask_special_token_ids` | `[]` | Extra structural IDs excluded from targets |

BOS, EOS, padding, task tokens, tokenizer special IDs, protected prefix positions,
and existing MASK positions are excluded from remask targets. Labels determine
supervision and sampling, but are never passed into the candidate evaluation pass.
LoRA remask training is currently rejected; use `lora=false` for these stages.

## Smoke training with the validation dataset

The supplied smoke configurations use
`aiai-laboratory/vietspeech-validation-translated`, split `validation`, limited to
32 base samples. The loader reads its embedded audio bytes directly and retains
the three translation task tokens. It does not download the full VietSpeech audio
corpus. Public datasets do not require an explicit HF token; normal Hugging Face
authentication still applies to private datasets.

Run with the project's uv environment:

```bash
uv run python src/train.py configs/test_learned_remasking.json
uv run python src/train.py configs/test_learned_remasking_joint.json
uv run python src/train.py configs/test_learned_remasking_streaming.json
```

Each run performs two optimizer steps with three rollout states per batch. The
first saves `outputs/remasker_mock_detector/checkpoint-2`; the second loads it.
The streaming configuration exercises older-audio drafts with newer-audio
supervision. Change its stage to `joint` and use a separate output directory to
exercise streaming reconstruction too. `uv run --no-sync` can be used when the
required dependencies are already installed in `.venv`.

These runs initialize the generator from XLM-R and Moonshine, so they validate
execution and gradient flow, **not translation quality or convergence**. For real
training, supply a warmed-up diffusion ASR/ST checkpoint and use training data.
The validation data used here should not also be treated as a held-out benchmark.

## Inference

For the standard model, the existing generator and Hugging Face `generate` methods
use learned decoding automatically when the head is enabled. Only MASK slots are
filled. The detector evaluates the filled canvas, preserves accepted tokens, and
may select **zero** revisions. Decoding stops when no masks or revision requests
remain, or when the iteration budget is exhausted. The final iteration returns a
filled candidate even if the detector still suspects errors.

For the streaming model:

```python
from model.streaming_diffusion_engine import StreamingDiffusionEngine

model.eval()
engine = StreamingDiffusionEngine.from_model(model)
# Feed encoded chunks through engine.on_new_audio_chunk with model.backbone,
# model.audio_adapter, model.streaming_length_predictor, and model.audio_resampler.
```

The engine re-evaluates the active draft against updated audio before generation.
It also evaluates each newly filled candidate in a separate backbone pass. Tokens
still flagged at the iteration limit get zero freeze confidence. Frozen context,
length expansion, overflow commitment and end-of-stream flushing retain their
existing policies; accepted active tokens can be reconsidered on the next chunk.

Local training checkpoints include the remask head in their state dict. Keep the
run's `args.json` and tokenizer with the checkpoint. Hugging Face model configs
persist the remask settings, and the packaging scripts include `remasking.py` and
preserve special-token IDs. No model upload is needed for training or tests.

## Baseline boundaries

Targets use aligned token equality, so this is reference reconstruction rather
than semantic translation correctness. Variable-length alignment, insertion and
deletion prediction are not implemented. Streaming supervision excludes positions
beyond `supported_lens`; the existing dataset estimates those lengths from the
audio-prefix ratio, **not forced acoustic alignment**. This is an approximate
support boundary and may be inaccurate for reordered translations. Rollouts use
the previous visible prefix when possible, with the newest chunk withheld; the
classifier and reconstruction pass receive the current prefix. Length prediction
and irreversible commitment remain separate mechanisms.

## Verification

```bash
PYTHONPATH=src uv run pytest test -q
uv run ruff check --select D src/model/remasking.py test/test_remasking.py
```

Tests cover target exclusion, zero-revision behavior, preservation of accepted
tokens, evaluation after sampling, budget exhaustion, frozen-generator gradients,
joint reconstruction gradients, streaming support masks, and checkpoint roundtrips.
