"""Speculative decoding with rejection correction and request-local rollback.

Backends expose context, vocabulary, prefill(ids), consume(ids, cache), and
truncate(cache, length). Prefill returns next logits; consume returns one next
logit row per consumed token. Cache state never belongs to the shared backend.
"""
from dataclasses import asdict, dataclass
import numpy as np


@dataclass
class Stats:
    target_calls: int = 0
    draft_calls: int = 0
    rounds: int = 0
    proposed: int = 0
    accepted: int = 0
    rejected_rounds: int = 0
    bonus_tokens: int = 0
    boundary_tokens: int = 0


def probabilities(logits, temperature=1.):
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("finite one-dimensional logits and positive temperature required")
    values = (values-values.max())/temperature
    result = np.exp(values)
    return result/result.sum()


def correction_distribution(p, q):
    residual = np.maximum(np.asarray(p, np.float64)-np.asarray(q, np.float64), 0.)
    total = residual.sum()
    if not np.isfinite(total) or total <= 0:
        raise ArithmeticError("rejection has no positive residual mass")
    return residual/total


def inputs(ids, model, count, mode, temperature):
    values = np.asarray(ids)
    if values.ndim != 1 or values.size == 0 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("nonempty one-dimensional integer token IDs required")
    if np.any(values < 0) or np.any(values >= model.vocabulary):
        raise ValueError("token outside vocabulary")
    if type(count) is not int or count < 0 or mode not in {"greedy", "sample"}:
        raise ValueError("nonnegative integer count and greedy/sample mode required")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("positive temperature required")
    return values.astype(int).tolist()


def draw(logits, mode, temperature, rng):
    return int(np.argmax(logits)) if mode == "greedy" else int(rng.choice(len(logits), p=probabilities(logits, temperature)))


def reference(target, ids, count, mode="greedy", temperature=1., seed=123):
    """Reference cached target generation; re-prefill at the context boundary."""
    prefix = inputs(ids, target, count, mode, temperature)
    result, stats = [], Stats()
    if not count:
        return result, asdict(stats)
    rng = np.random.default_rng(seed)
    logits, cache = target.prefill(prefix[-target.context:])
    stats.target_calls += 1
    stats.boundary_tokens = int(len(prefix) >= target.context)
    for step in range(count):
        token = draw(logits, mode, temperature, rng)
        result.append(token)
        prefix.append(token)
        if step+1 < count:
            if len(prefix) > target.context:
                logits, cache = target.prefill(prefix[-target.context:])
                stats.boundary_tokens += 1
            else:
                rows, cache = target.consume([token], cache)
                logits = rows[-1]
            stats.target_calls += 1
    return result, asdict(stats)


def generate(target, draft, ids, count, gamma=4, mode="greedy", temperature=1., seed=123):
    """Preserve greedy IDs or, with residual correction, the target distribution.

    A correction/bonus token stays pending in the target cache and is consumed
    with the next verification block, avoiding an extra target call per round.
    Once a sliding window is full, use the reference semantics one token at a
    time. No stale hidden states survive a shifted window.
    """
    prefix = inputs(ids, target, count, mode, temperature)
    if draft.vocabulary != target.vocabulary or draft.context != target.context:
        raise ValueError("target and draft need identical vocabulary IDs and context")
    if type(gamma) is not int or gamma < 1:
        raise ValueError("positive integer draft block size required")
    result, stats = [], Stats()
    if not count:
        return result, asdict(stats)
    if len(prefix) >= target.context:
        return reference(target, prefix, count, mode, temperature, seed)
    rng = np.random.default_rng(seed)
    prefix = prefix[-target.context:]
    p_next, target_cache = target.prefill(prefix)
    stats.target_calls += 1
    q_next, draft_cache = draft.prefill(prefix)
    stats.draft_calls += 1
    pending = None
    while len(result) < count:
        remaining = count-len(result)
        if len(prefix) >= target.context:
            # Prefill the exact current window; old deeper states are invalid.
            p_next, target_cache = target.prefill(prefix[-target.context:])
            stats.target_calls += 1
            token = draw(p_next, mode, temperature, rng)
            result.append(token)
            prefix = (prefix+[token])[-target.context:]
            stats.boundary_tokens += 1
            pending = None
            continue
        size = min(gamma, target.context-len(prefix), remaining)
        proposed, proposal_distributions = [], []
        base_length = len(prefix)
        for _ in range(size):
            distribution = probabilities(q_next, temperature) if mode == "sample" else None
            token = int(rng.choice(target.vocabulary, p=distribution)) if mode == "sample" else int(np.argmax(q_next))
            proposed.append(token)
            proposal_distributions.append(distribution)
            rows, draft_cache = draft.consume([token], draft_cache)
            stats.draft_calls += 1
            q_next = rows[-1]
        block = ([pending] if pending is not None else [])+proposed
        rows, verified_cache = target.consume(block, target_cache)
        stats.target_calls += 1
        if pending is None:
            target_logits = [p_next, *rows]
        else:
            target_logits = list(rows)
        stats.rounds += 1
        stats.proposed += size
        accepted = 0
        rejected = False
        for index, token in enumerate(proposed):
            if mode == "greedy":
                keep = token == int(np.argmax(target_logits[index]))
            else:
                p = probabilities(target_logits[index], temperature)
                q = proposal_distributions[index]
                keep = rng.random() < min(1., p[token]/q[token])
            if not keep:
                rejected = True
                break
            accepted += 1
        stats.accepted += accepted
        stats.rejected_rounds += int(rejected)
        committed = proposed[:accepted]
        if accepted < remaining:
            if rejected and mode == "sample":
                corrected = correction_distribution(probabilities(target_logits[accepted], temperature), proposal_distributions[accepted])
                token = int(rng.choice(target.vocabulary, p=corrected))
            else:
                token = draw(target_logits[accepted], mode, temperature, rng)
            committed.append(token)
            stats.bonus_tokens += int(not rejected)
            pending = token
        else:
            pending = None
        target_cache = target.truncate(verified_cache, base_length+accepted)
        result.extend(committed)
        prefix.extend(committed)
        if len(result) < count and len(prefix) < target.context:
            draft_cache = draft.truncate(draft_cache, base_length+accepted)
            rows, draft_cache = draft.consume([pending], draft_cache)
            stats.draft_calls += 1
            q_next = rows[-1]
        # At/full beyond context, the next round uses fresh-window target only.
    return result, asdict(stats)
