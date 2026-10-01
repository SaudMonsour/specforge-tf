"""Audit the sampler on a history-dependent model with an exact joint law."""
import itertools
import json
from pathlib import Path
import numpy as np
from specforge import generate, correction_distribution


class TableBackend:
    def __init__(self, table):
        self.table = np.asarray(table, float)
        self.context, self.vocabulary = 64, self.table.shape[1]

    def prefill(self, ids):
        return np.log(self.table[ids[-1]]), tuple(ids)

    def consume(self, ids, cache):
        rows = []
        for token in ids:
            cache = cache+(token,)
            rows.append(np.log(self.table[token]))
        return np.asarray(rows), cache

    @staticmethod
    def truncate(cache, length):
        return cache[:length]


def audit(samples=12000):
    p = np.array([[.2, .8], [.7, .3]])
    q = np.array([[.8, .2], [.25, .75]])
    sequences = list(itertools.product(range(2), repeat=3))
    expected = np.array([p[0, a]*p[a, b]*p[b, c] for a, b, c in sequences])
    counts = np.zeros(len(sequences), dtype=int)
    target, draft = TableBackend(p), TableBackend(q)
    for seed in range(samples):
        tokens, _ = generate(target, draft, [0], 3, gamma=2, mode="sample", seed=seed)
        counts[sequences.index(tuple(tokens))] += 1
    observed = counts/samples
    bound = 5*np.sqrt(expected*(1-expected)/samples)+1/samples
    assert np.all(np.abs(observed-expected) < bound)
    closed_form = []
    for row_p, row_q in zip(p, q):
        accepted = np.minimum(row_p, row_q)
        corrected = accepted+(1-accepted.sum())*correction_distribution(row_p, row_q)
        np.testing.assert_allclose(corrected, row_p, atol=1e-15, rtol=0)
        closed_form.append(float(np.max(np.abs(corrected-row_p))))
    return {"target_transition_matrix": p.tolist(), "draft_transition_matrix": q.tolist(),
            "prompt": [0], "gamma": 2, "generated_length": 3,
            "samples": samples, "seeds": [0, samples-1],
            "sequences": [list(x) for x in sequences], "counts": counts.tolist(),
            "expected_joint_probabilities": expected.tolist(), "observed_joint_frequencies": observed.tolist(),
            "absolute_error": np.abs(expected-observed).tolist(), "five_sigma_bounds": bound.tolist(),
            "maximum_absolute_frequency_error": float(np.max(np.abs(expected-observed))),
            "closed_form_maximum_absolute_error": max(closed_form), "passed": True,
            "scope": "Exact one-step mass identity plus finite empirical three-token sequence-law check. This is not a proof by sampling or a neural language-quality metric."}


if __name__ == "__main__":
    result = audit()
    path = Path(__file__).resolve().parent/"runs/sampler_check.json"
    path.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))
