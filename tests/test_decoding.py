import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import itertools
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from specforge import generate, reference, probabilities, correction_distribution
from specforge.model import Backend, Config, Decoder, load_checkpoint


class TableBackend:
    """A history-dependent finite model makes exact sequence laws inspectable."""
    def __init__(self, table, context=64):
        self.table = np.asarray(table, float)
        self.context, self.vocabulary = context, self.table.shape[1]
        self.calls = 0

    def prefill(self, ids):
        self.calls += 1
        return np.log(self.table[ids[-1]]), tuple(ids)

    def consume(self, ids, cache):
        self.calls += 1
        rows = []
        for token in ids:
            cache = cache+(token,)
            rows.append(np.log(self.table[token]))
        if len(cache) > self.context:
            raise ValueError("context exceeded")
        return np.array(rows), cache

    @staticmethod
    def truncate(cache, length):
        return cache[:length]


class DecodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.experimental.enable_op_determinism()

    def model(self, seed, layers=2):
        tf.keras.utils.set_random_seed(seed)
        model = Decoder(Config(vocabulary=7, context=12, width=16, heads=2, layers=layers, hidden=24))
        model(tf.zeros([1, 1], tf.int32))
        return model

    def test_causal_mask_prevents_future_leakage(self):
        model = self.model(42)
        a, b = model(tf.constant([[0, 1, 2, 3]])), model(tf.constant([[0, 1, 5, 6]]))
        np.testing.assert_allclose(a.numpy()[:, :2], b.numpy()[:, :2], atol=1e-6)

    def test_block_cache_matches_full_prefix_logits(self):
        model = self.model(42)
        engine = Backend(model)
        _, caches = engine.prefill([0, 1, 2])
        rows, _ = engine.consume([3, 4, 5], caches)
        full = model(tf.constant([[0, 1, 2, 3, 4, 5]])).numpy()[0]
        np.testing.assert_allclose(rows, full[3:], atol=2e-5, rtol=2e-5)

    def test_rollback_discards_rejected_hidden_states(self):
        model = self.model(42)
        engine = Backend(model)
        _, cache = engine.prefill([0, 1])
        _, speculative = engine.consume([2, 3, 4], cache)
        rows, _ = engine.consume([6], engine.truncate(speculative, 3))
        np.testing.assert_allclose(rows[-1], model(tf.constant([[0, 1, 2, 6]])).numpy()[0, -1], atol=2e-5, rtol=2e-5)

    def test_tensorflow_greedy_ids_match_through_window_resets(self):
        target, draft = Backend(self.model(42)), Backend(self.model(43, layers=1))
        expected, _ = reference(target, [0, 1, 2], 24)
        for gamma in (1, 2, 4, 8):
            actual, stats = generate(target, draft, [0, 1, 2], 24, gamma=gamma)
            self.assertEqual(actual, expected)
            self.assertGreater(stats["boundary_tokens"], 0)

    def test_closed_form_correction_recovers_target_mass(self):
        for p, q in (([.1, .3, .6], [.6, .3, .1]), ([0., .4, .6], [.5, .5, 0.])):
            p, q = np.array(p), np.array(q)
            accepted = np.minimum(p, q)
            law = accepted+(1-accepted.sum())*correction_distribution(p, q)
            np.testing.assert_allclose(law, p, atol=1e-15, rtol=0)

    def test_sample_joint_distribution_matches_history_dependent_target(self):
        p = np.array([[.2, .8], [.7, .3]])
        q = np.array([[.8, .2], [.25, .75]])
        target, draft = TableBackend(p), TableBackend(q)
        sequences = list(itertools.product(range(2), repeat=3))
        expected = np.array([p[0, a]*p[a, b]*p[b, c] for a, b, c in sequences])
        counts = np.zeros(8)
        for seed in range(12000):
            tokens, _ = generate(target, draft, [0], 3, gamma=2, mode="sample", seed=seed)
            counts[sequences.index(tuple(tokens))] += 1
        # Fixed sample count/seeds and five-sigma bound; no repeated seed search.
        np.testing.assert_array_less(np.abs(counts/12000-expected), 5*np.sqrt(expected*(1-expected)/12000)+1/12000)

    def test_identical_draft_all_accepts_and_uses_bonus(self):
        table = [[.2, .8], [.7, .3]]
        tokens, stats = generate(TableBackend(table), TableBackend(table), [0], 12, gamma=4, mode="sample", seed=42)
        self.assertEqual(len(tokens), 12)
        self.assertEqual(stats["accepted"], stats["proposed"])
        self.assertEqual(stats["rejected_rounds"], 0)
        self.assertGreater(stats["bonus_tokens"], 0)

    def test_first_rejection_and_partial_acceptance_match_target(self):
        target = TableBackend([[.1, .9], [.9, .1]])
        draft = TableBackend([[.9, .1], [.9, .1]])
        expected, _ = reference(target, [0], 11)
        actual, stats = generate(target, draft, [0], 11, gamma=4)
        self.assertEqual(actual, expected)
        self.assertGreater(stats["rejected_rounds"], 0)
        self.assertGreater(stats["accepted"], 0)

    def test_budget_never_overshoots_and_zero_budget_has_no_calls(self):
        for count in (0, 1, 2, 3, 4, 5, 9):
            target, draft = TableBackend([[.3, .7], [.8, .2]]), TableBackend([[.3, .7], [.8, .2]])
            tokens, stats = generate(target, draft, [0], count, gamma=4)
            self.assertEqual(len(tokens), count)
            if not count:
                self.assertEqual(target.calls+draft.calls, 0)

    def test_full_window_fallback_uses_no_draft(self):
        target, draft = TableBackend([[.1, .9], [.8, .2]], context=4), TableBackend([[.9, .1], [.2, .8]], context=4)
        actual, stats = generate(target, draft, [0, 1, 0, 1], 5)
        self.assertEqual(actual, reference(target, [0, 1, 0, 1], 5)[0])
        self.assertEqual(stats["draft_calls"], 0)
        self.assertEqual(stats["target_calls"], 5)

    def test_request_state_is_isolated(self):
        target, draft = Backend(self.model(42)), Backend(self.model(43, layers=1))
        first, _ = generate(target, draft, [0, 1], 6)
        generate(target, draft, [6, 5, 4], 10)
        again, _ = generate(target, draft, [0, 1], 6)
        self.assertEqual(first, again)

    def test_checkpoint_reload_preserves_logits(self):
        model = self.model(42)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"config.json").write_text(json.dumps(asdict(model.config)))
            model.save_weights(root/"model.weights.h5")
            reloaded = load_checkpoint(root)
            np.testing.assert_array_equal(model(tf.constant([[0, 1, 2]])).numpy(), reloaded(tf.constant([[0, 1, 2]])).numpy())

    def test_gradients_are_finite_and_reach_attention(self):
        model = self.model(42)
        with tf.GradientTape() as tape:
            logits = model(tf.constant([[0, 1, 2, 3]]))
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[[1, 2, 3, 4]], logits=logits))
        gradients = tape.gradient(loss, model.trainable_variables)
        self.assertTrue(all(g is not None and np.isfinite(tf.convert_to_tensor(g).numpy()).all() for g in gradients))
        attention_variables = {id(v) for block in model.blocks for v in block.attention.qkv.trainable_variables}
        self.assertTrue(any(id(variable) in attention_variables and np.linalg.norm(tf.convert_to_tensor(g).numpy()) > 0
                            for variable, g in zip(model.trainable_variables, gradients)))

    def test_graphs_do_not_retrace_for_block_and_cache_lengths(self):
        backend = Backend(self.model(42))
        for length in (1, 2, 4):
            _, cache = backend.prefill([0]*length)
            backend.consume([1], cache)
            backend.consume([1, 2], cache)
        self.assertEqual(backend.traces(), {"prefill": 1, "consume": 1})

    def test_invalid_inputs_and_incompatible_backends_fail(self):
        backend = TableBackend([[.2, .8], [.7, .3]])
        for ids in ([], [2], [-1], [0.5], [[0]]):
            with self.assertRaises(ValueError):
                generate(backend, backend, ids, 1)
        for kwargs in ({"gamma": 0}, {"count": -1}, {"temperature": 0}, {"mode": "invalid"}):
            options = {"count": 1, **kwargs}
            with self.assertRaises(ValueError):
                generate(backend, backend, [0], **options)
        with self.assertRaises(ValueError):
            generate(backend, TableBackend([[.2, .8], [.7, .3]], context=8), [0], 1)
        with self.assertRaises(ArithmeticError):
            correction_distribution([.5, .5], [.5, .5])


if __name__ == "__main__":
    unittest.main()
