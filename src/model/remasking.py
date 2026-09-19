"""Learn error detection from fresh, position-aligned generator trajectories."""

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn


def eligible_positions(tokens, mask_id, special_ids=(), protected=None):
    """Select generated content tokens, excluding masks and protected positions."""
    valid = tokens.ne(mask_id) & tokens.ge(0)
    for token_id in special_ids:
        if token_id is not None:
            valid &= tokens.ne(token_id)
    if protected is not None:
        valid &= ~protected.bool()
    return valid


def remask_targets(
    candidate, reference, mask_id, special_ids=(), protected=None, supported=None
):
    """Construct reference-match labels only for aligned, observable content."""
    if candidate.shape != reference.shape:
        raise ValueError(
            "Remasking requires position-aligned candidate/reference shapes"
        )
    valid = eligible_positions(candidate, mask_id, special_ids, protected)
    valid &= eligible_positions(reference, mask_id, special_ids, protected)
    if supported is not None:
        valid &= supported.bool()
    return candidate.ne(reference).float(), valid


def balanced_error_loss(logits, targets, valid):
    """Average present KEEP/REMASK classes equally, including all-KEEP batches."""
    losses = F.binary_cross_entropy_with_logits(
        logits.float(), targets.float(), reduction="none"
    )
    keep = valid & targets.eq(0)
    error = valid & targets.eq(1)
    keep_loss = (losses * keep).sum() / keep.sum().clamp_min(1)
    error_loss = (losses * error).sum() / error.sum().clamp_min(1)
    classes = keep.any().float() + error.any().float()
    return (keep_loss + error_loss) / classes.clamp_min(1)


@contextmanager
def evaluating(model):
    """Disable dropout during rollout, restoring each module's original mode."""
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, mode in modes:
            module.training = mode


@torch.no_grad()
def collect_rollout(forward, reference, maskable, mask_id, steps):
    """Collect actual greedy generator candidates under a confidence schedule.

    References provide only fixed prefix/special tokens and oracle canvas length.
    Every content position starts masked; no reference content is teacher-forced.
    """
    canvas = reference.masked_fill(maskable, mask_id)
    states = []
    for step in range(steps):
        logits = forward(canvas).clone()
        logits[..., mask_id] = -torch.inf
        scores, predicted = logits.float().log_softmax(-1).max(-1)
        canvas = torch.where(canvas.eq(mask_id) & maskable, predicted, canvas)
        states.append(canvas.clone())
        if step + 1 < steps:
            # Refresh decreasing numbers of uncertain tokens to expose early/late errors.
            ranks = scores.masked_fill(~maskable, torch.inf).argsort(-1).argsort(-1)
            count = (maskable.sum(-1, keepdim=True) * (1 - (step + 1) / steps)).long()
            canvas = canvas.masked_fill((ranks < count) & maskable, mask_id)
    return torch.stack(states)  # [steps, batch, length]


def sample_rollout(states, reference, valid):
    """Sample available zero/low/high-error strata, then a timestep within each."""
    rates = ((states != reference) & valid).sum(-1) / valid.sum(-1).clamp_min(1)
    strata = torch.where(rates.eq(0), 0, torch.where(rates.le(0.5), 1, 2))
    selected = []
    for b in range(reference.shape[0]):
        available = strata[:, b].unique()
        bucket = available[torch.randint(len(available), (), device=states.device)]
        indices = (strata[:, b] == bucket).nonzero().flatten()
        index = indices[torch.randint(len(indices), (), device=states.device)]
        selected.append(states[index, b])
    return torch.stack(selected)


class LearnedRemaskingMixin:
    """Shared remask head, training objective, and iterative decoding behavior."""

    def init_remasker(self):
        """Create the optional head without changing legacy checkpoint keys."""
        self.remask_head = None
        if getattr(self.args, "learned_remasking", False):
            self.remask_head = nn.Linear(self.config.hidden_size, 1)
        self.configure_remask_training()

    def configure_remask_training(self):
        """Freeze the generator for detector training, before optimizer creation."""
        stage = getattr(self.args, "remask_training_stage", "disabled")
        if stage not in {"disabled", "detector", "joint"}:
            raise ValueError(
                "remask_training_stage must be disabled, detector, or joint"
            )
        if stage != "disabled" and self.remask_head is None:
            raise ValueError("Remask training requires learned_remasking=true")
        if not 0 < getattr(self.args, "remask_threshold", 0.5) < 1:
            raise ValueError("remask_threshold must be between zero and one")
        if getattr(self.args, "remask_rollout_steps", 20) < 1:
            raise ValueError("remask_rollout_steps must be positive")
        if stage == "detector":
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            self.remask_head.requires_grad_(True)
            self.train(self.training)

    def train(self, mode=True):
        """Keep the frozen generator deterministic while training its classifier."""
        super().train(mode)
        if (
            getattr(getattr(self, "args", None), "remask_training_stage", None)
            == "detector"
        ):
            for name, module in self.named_children():
                if name != "remask_head":
                    module.eval()
        return self

    def remask_special_ids(self):
        """Return structural, task and padding token IDs excluded from supervision."""
        ids = [self.bos_id, self.eos_id, self.pad_id]
        ids.extend(getattr(self.args, "remask_special_token_ids", ()))
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is not None:
            ids.extend(tokenizer.all_special_ids)
        return ids

    def remask_training_loss(
        self,
        reference,
        partial_mask,
        supported=None,
        rollout_kwargs=None,
        **forward_kwargs,
    ):
        """Train detection and optionally reconstruction on fresh rollout errors.

        Older audio may be supplied for rollout; candidate evaluation always uses
        forward_kwargs (the current audio). Labels never enter the classifier pass.
        """
        special_ids = self.remask_special_ids()
        maskable = eligible_positions(
            reference, self.mask_id, special_ids, partial_mask
        )
        valid = maskable if supported is None else maskable & supported.bool()
        with evaluating(self), torch.no_grad():
            states = collect_rollout(
                lambda canvas: self(
                    canvas, partial_mask, **(rollout_kwargs or forward_kwargs)
                ),
                reference,
                maskable,
                self.mask_id,
                self.args.remask_rollout_steps,
            )
            candidate = sample_rollout(states, reference, valid)
        targets, valid = remask_targets(
            candidate, reference, self.mask_id, special_ids, partial_mask, supported
        )
        remask_logits = self(
            candidate, partial_mask, remask_only=True, **forward_kwargs
        )
        detection_loss = balanced_error_loss(remask_logits, targets, valid)
        loss = self.args.remask_loss_weight * detection_loss
        if self.args.remask_training_stage == "joint":
            revisions = (
                remask_logits.detach().sigmoid() >= self.args.remask_threshold
            ) & valid
            canvas = candidate.masked_fill(revisions, self.mask_id)
            logits = self(canvas, partial_mask, **forward_kwargs)
            # Empty selections still produce a differentiable, finite zero loss.
            ce = F.cross_entropy(
                logits.float().transpose(1, 2), reference.clamp_min(0), reduction="none"
            )
            reconstruction = (ce * revisions).sum() / revisions.sum().clamp_min(1)
            loss = loss + self.args.remask_reconstruction_weight * reconstruction
        return {"loss": loss, "logits": remask_logits}

    @torch.no_grad()
    def learned_denoise_step(self, decoder_out, partial_mask, **forward_kwargs):
        """Fill masks, evaluate the updated candidate, and selectively request revision."""
        tokens = decoder_out.output_tokens.clone()
        scores = decoder_out.output_scores.clone()
        mask = tokens.eq(self.mask_id) & ~partial_mask
        if mask.any():
            logits = self(tokens, partial_mask, **forward_kwargs).clone()
            logits[..., self.mask_id] = -torch.inf
            new_scores, predictions = logits.float().log_softmax(-1).max(-1)
            tokens[mask] = predictions[mask]
            scores[mask] = new_scores[mask].to(scores)
        remask_logits = self(tokens, partial_mask, remask_only=True, **forward_kwargs)
        eligible = eligible_positions(
            tokens, self.mask_id, self.remask_special_ids(), partial_mask
        )
        eligible &= decoder_out.non_fixed_sym_masks
        revisions = (remask_logits.sigmoid() >= self.args.remask_threshold) & eligible
        step = decoder_out.step + 1
        if step < decoder_out.max_step:
            tokens.masked_fill_(revisions, self.mask_id)
            scores.masked_fill_(revisions, -torch.inf)
        # On budget exhaustion return the filled candidate, never a newly masked canvas.
        history = (
            None
            if decoder_out.history is None
            else decoder_out.history + [tokens.clone()]
        )
        return decoder_out._replace(
            output_tokens=tokens,
            output_scores=scores,
            output_masks=revisions,
            step=step,
            history=history,
        )
