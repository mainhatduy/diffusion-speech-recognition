"""Verify learned remasking targets, gradients, rollouts and streaming revisions."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import RobertaConfig

from model.configuration_dlm import DiscreteDiffusionConfig
from model.dd_model import decoder_out_t
from model.modeling_dlm import DiscreteDiffusionModel
from model.remasking import (
    balanced_error_loss,
    collect_rollout,
    remask_targets,
    sample_rollout,
)
from model.streaming_backbone import StreamingBackboneConfig, StreamingDiffusionBackbone
from model.streaming_diffusion_engine import StreamingDiffusionEngine


def tiny_model(stage="detector"):
    """Create a download-free model for gradient and checkpoint checks."""
    config = DiscreteDiffusionConfig(
        backbone_config=RobertaConfig(
            vocab_size=20,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
        ).to_dict(),
        mask_token_id=3,
        bos_token_id=0,
        eos_token_id=2,
        pad_token_id=1,
        learned_remasking=True,
        remask_training_stage=stage,
        remask_rollout_steps=3,
        remask_special_token_ids=[4],
    )
    return DiscreteDiffusionModel(config)


def test_targets_exclude_masks_specials_prefix_padding_and_future():
    """Supervise only generated content supported by the current audio."""
    reference = torch.tensor([[0, 4, 5, 6, 7, 8, 2, 1]])
    candidate = torch.tensor([[0, 4, 5, 9, 3, 9, 2, 1]])
    supported = torch.tensor([[1, 1, 1, 1, 1, 0, 1, 1]]).bool()
    targets, valid = remask_targets(
        candidate, reference, 3, [0, 1, 2, 4], supported=supported
    )
    assert valid.tolist() == [[False, False, True, True, False, False, False, False]]
    assert targets[valid].tolist() == [0, 1]
    with pytest.raises(ValueError, match="aligned"):
        remask_targets(candidate[:, :-1], reference, 3)


@pytest.mark.parametrize("target,valid", [(0, True), (1, True), (0, False)])
def test_balanced_loss_handles_single_class_and_empty(target, valid):
    """Keep all-correct and empty examples finite and differentiable."""
    logits = torch.zeros(2, 3, requires_grad=True)
    loss = balanced_error_loss(
        logits,
        torch.full_like(logits, target),
        torch.full_like(logits, valid, dtype=torch.bool),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    if not valid:
        assert loss.item() == 0


def test_rollout_hides_reference_content_and_retains_real_candidates():
    """Expose the generator's own errors without teacher-forcing content."""
    seen = []
    reference = torch.tensor([[0, 4, 5, 6, 2]])
    maskable = torch.tensor([[False, False, True, True, False]])

    def forward(canvas):
        seen.append(canvas.clone())
        logits = torch.zeros(1, 5, 20)
        logits[..., 9] = 10
        return logits

    states = collect_rollout(forward, reference, maskable, 3, steps=4)
    assert seen[0].tolist() == [[0, 4, 3, 3, 2]]
    assert states[:, :, 2:4].eq(9).all()
    candidate = sample_rollout(states, reference, maskable)
    assert candidate.tolist() == [[0, 4, 9, 9, 2]]


@pytest.mark.parametrize("stage", ["detector", "joint"])
def test_training_gradients_and_fresh_candidate_pass(stage):
    """Train the head alone first and allow generator gradients during joint tuning."""
    model = tiny_model(stage).train()
    seen = []

    def record(module, args, kwargs):
        if kwargs.get("remask_only"):
            seen.append(args[0].clone())
            assert "remask_training_labels" not in kwargs

    handle = model.register_forward_pre_hook(record, with_kwargs=True)
    reference = torch.tensor([[0, 4, 5, 6, 2]])
    protected = torch.tensor([[True, True, False, False, True]])
    # Ensure joint reconstruction selects content and produces a useful gradient.
    with torch.no_grad():
        model.remask_head.bias.fill_(10)
        model.model.lm_head.bias[9] = 20
    output = model(reference, protected, remask_training_labels=reference)
    output["loss"].backward()
    handle.remove()
    assert seen and seen[0][0, 2:4].eq(9).all()
    assert model.remask_head.weight.grad is not None
    grads = [
        p.grad for n, p in model.named_parameters() if not n.startswith("remask_head")
    ]
    if stage == "detector":
        assert all(g is None for g in grads)
        assert not model.model.training
    else:
        assert any(g is not None and g.abs().sum() > 0 for g in grads)
    assert torch.isfinite(output["loss"])


def test_checkpoint_roundtrip(tmp_path):
    """Preserve classifier parameters and remask configuration in HF checkpoints."""
    model = tiny_model()
    model.save_pretrained(tmp_path)
    restored = DiscreteDiffusionModel.from_pretrained(tmp_path)
    assert restored.config.learned_remasking
    assert restored.config.remask_special_token_ids == [4]
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)


class ScriptedBackbone(nn.Module):
    """Expose token/audio identity to a deterministic mock detector."""

    def __init__(self):
        """Track generated canvases and classifier evaluations."""
        super().__init__()
        self.generated = []
        self.evaluated = []

    def forward(
        self, input_ids, audio_hidden=None, return_hidden_states=False, **kwargs
    ):
        """Generate token six while exposing token five as an audio-dependent error."""
        if return_hidden_states:
            self.evaluated.append(input_ids.clone())
            errors = (
                input_ids.eq(5).float()
                if audio_hidden is not None
                else torch.zeros_like(input_ids).float()
            )
            return (errors * 20 - 10).unsqueeze(-1), []
        self.generated.append(input_ids.clone())
        logits = torch.zeros(*input_ids.shape, 10)
        logits[..., 6] = 10
        return logits, []


def test_streaming_preserves_keep_and_revises_with_new_audio():
    """Revisit an old draft under new audio without overwriting accepted tokens."""
    backbone = ScriptedBackbone()
    engine = StreamingDiffusionEngine(
        3, 2, 1, remask_head=nn.Identity(), special_token_ids=[0, 4]
    )
    engine.active_tokens = [0, 4, 5, 7, 3, 2]
    engine.active_confidence = [1, 1, 1, 1, 0, 1]
    engine.audio_buffer = [torch.ones(1, 1)]
    engine._denoise_active_zone(backbone)
    assert engine.active_tokens == [0, 4, 6, 7, 6, 2]
    assert len(backbone.generated) == 1
    assert backbone.generated[0].tolist() == [[0, 4, 3, 7, 3, 2]]
    assert backbone.evaluated[-1].tolist() == [[0, 4, 6, 7, 6, 2]]


def test_streaming_all_keep_can_stop_without_generation():
    """Permit zero revisions when the current candidate is accepted."""
    backbone = ScriptedBackbone()
    engine = StreamingDiffusionEngine(3, 2, 1, remask_head=nn.Identity())
    engine.active_tokens = [7, 8]
    engine.active_confidence = [0.8, 0.7]
    engine._denoise_active_zone(backbone)
    assert engine.active_tokens == [7, 8]
    assert not backbone.generated


def test_budget_exhaustion_returns_filled_tokens_without_freezing_errors():
    """Return a filled draft even when the detector continues requesting changes."""
    head = nn.Linear(1, 1)
    nn.init.zeros_(head.weight)
    nn.init.constant_(head.bias, 10)
    engine = StreamingDiffusionEngine(
        3, 2, 1, remask_head=head, streaming_denoise_steps=2
    )
    engine.active_tokens, engine.active_confidence = [3, 7], [0, 1]
    engine._denoise_active_zone(ScriptedBackbone())
    assert 3 not in engine.active_tokens
    assert engine.active_confidence == [0, 0]


def test_offline_learned_step_preserves_tokens_and_evaluates_candidate():
    """Use the same fill/evaluate semantics in offline decoding."""
    model = tiny_model("disabled").eval()
    with torch.no_grad():
        model.remask_head.weight.zero_()
        model.remask_head.bias.fill_(-10)
    tokens = torch.tensor([[0, 4, 7, 3, 2]])
    protected = torch.tensor([[True, True, False, False, True]])
    state = decoder_out_t(
        tokens,
        torch.zeros_like(tokens).float(),
        tokens.eq(3),
        ~protected,
        None,
        0,
        3,
        [],
    )
    result = model.learned_denoise_step(state, protected)
    assert result.output_tokens[0, 2] == 7
    assert not result.output_tokens.eq(3).any()
    assert not result.output_masks.any()
    assert len(result.history) == 1


def test_streaming_backbone_exposes_audio_conditioned_hidden_states():
    """Share actual backbone features with the head without computing LM logits."""
    config = StreamingBackboneConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_hidden_layers=2,
        num_ergodic_layers=1,
        num_position_layers=1,
    )
    backbone = StreamingDiffusionBackbone(config, 20).eval()
    tokens = torch.tensor([[0, 5, 6]])
    audio = torch.randn(1, 4, 16)
    hidden, _ = backbone(tokens, audio_hidden=audio, return_hidden_states=True)
    logits, _ = backbone(tokens, audio_hidden=audio)
    torch.testing.assert_close(backbone.lm_head(hidden), logits)


def test_streaming_trainer_uses_older_audio_and_supported_targets():
    """Pass draft audio separately and exclude unsupported reference positions."""
    from trainer.dd_trainer import StreamingDiffusionTrainer

    class TrainingStub(nn.Module):
        """Capture the streaming trainer's remask objective inputs."""

        def __init__(self):
            """Set minimal streaming attributes."""
            super().__init__()
            self.args = SimpleNamespace(remask_training_stage="detector")
            self.config = SimpleNamespace(hidden_size=4)
            self.mask_id, self.pad_id = 3, 1
            self.audio_adapter = nn.Identity()

        def forward(self, reference, protected, **kwargs):
            """Capture masks and return a differentiable scalar loss."""
            self.captured = reference, protected, kwargs
            return {
                "loss": torch.tensor(0.5, requires_grad=True),
                "logits": reference.float(),
            }

    model = TrainingStub()
    trainer = object.__new__(StreamingDiffusionTrainer)
    inputs = {
        "audio_chunks": torch.ones(1, 2, 3, 4),
        "audio_mask": torch.ones(1, 2).bool(),
        "text_ids": torch.tensor([[0, 5, 6, 2]]),
        "text_mask": torch.ones(1, 4).bool(),
        "support_ratios": torch.tensor([0.5]),
        "supported_lens": torch.tensor([2]),
        "task_token_ids": torch.tensor([4]),
        "is_precomputed": True,
    }
    loss = trainer.compute_loss(model, inputs)
    reference, protected, kwargs = model.captured
    assert loss.item() == 0.5
    assert reference.tolist() == [[4, 0, 5, 6, 2]]
    assert protected[0, 0]
    assert kwargs["remask_supported"].tolist() == [[False, True, True, False, False]]
    assert kwargs["projected_audio_mask"].sum() == 6
    assert kwargs["remask_rollout_kwargs"]["projected_audio_mask"].sum() == 3


def test_external_generator_retains_learned_history():
    """Keep requested history when the external generator selects learned decoding."""
    from dd_generator import (
        DiscreteDiffusionGenerator,
        DiscreteDiffusionGeneratorArguments,
    )

    model = tiny_model("disabled").eval()
    with torch.no_grad():
        model.remask_head.weight.zero_()
        model.remask_head.bias.fill_(-10)
    generator = object.__new__(DiscreteDiffusionGenerator)
    generator.args = DiscreteDiffusionGeneratorArguments(return_history=True)
    generator.retain_history = True
    tokens = torch.tensor([[0, 4, 3, 2]])
    protected = torch.tensor([[True, True, False, True]])
    state = decoder_out_t(
        tokens,
        torch.zeros_like(tokens).float(),
        tokens.eq(3),
        ~protected,
        None,
        0,
        3,
        None,
    )
    output = generator.denoise_step(model, state, protected)
    assert len(output.history) == 1
    assert not output.output_tokens.eq(3).any()


def test_legacy_config_does_not_create_classifier_parameters():
    """Leave the checkpoint layout unchanged until remasking is explicitly enabled."""
    config = tiny_model().config
    config.learned_remasking = False
    config.remask_training_stage = "disabled"
    model = DiscreteDiffusionModel(config)
    assert model.remask_head is None
    assert not any(key.startswith("remask_head.") for key in model.state_dict())
