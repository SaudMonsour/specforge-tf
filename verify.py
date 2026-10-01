"""Replay saved scores, output IDs, source bytes and publication hashes."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
import h5py
import numpy as np
import tensorflow as tf
from prepare import read_data, body_from_source, sha256
from study import evaluate, windows
from specforge import generate, reference
from specforge.model import Backend, load_checkpoint
from sampler_check import audit as sampler_audit
from artifacts import files

ROOT = Path(__file__).resolve().parent


def close(actual, expected, atol=1e-6):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=0)


def main():
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    assert os.environ["TF_ENABLE_ONEDNN_OPTS"] == "0"
    manifest, (train, validation, test) = read_data()
    with urlopen(manifest["download_url"], timeout=30) as response:
        fresh = response.read(500001)
    assert fresh == (ROOT/"data/source.txt").read_bytes()
    assert sha256(fresh) == manifest["source_sha256"]
    assert body_from_source(fresh) == (ROOT/"data/body.bin").read_bytes()
    assert sum(map(len, (train, validation, test))) == manifest["body_bytes"]
    metrics = json.loads((ROOT/"runs/metrics.json").read_text())
    benchmark = json.loads((ROOT/"runs/benchmark.json").read_text())
    for result in (metrics, benchmark):
        assert result["source_sha256"] == manifest["source_sha256"]
        assert result["body_sha256"] == manifest["body_sha256"]
    schedule = np.load(ROOT/"runs/training_starts.npy", allow_pickle=False)
    expected = np.random.default_rng(metrics["seed"]).integers(0, len(train)-96, size=(metrics["steps"], metrics["batch"]))
    np.testing.assert_array_equal(schedule, expected)
    with (ROOT/"runs/training_history.csv").open() as handle:
        history = list(csv.DictReader(handle))
    engines, replayed_scores = {}, {}
    for role in ("draft", "target"):
        directory = ROOT/"runs"/role
        with h5py.File(directory/"model.weights.h5", "r") as archive:
            def finite(name, value):
                if isinstance(value, h5py.Dataset):
                    assert np.isfinite(value[()]).all(), name
            archive.visititems(finite)
        model = load_checkpoint(directory)
        assert model.count_params() == metrics["models"][role]["parameters"]
        rows = [r for r in history if r["model"] == role]
        selected = min(rows, key=lambda r: float(r["validation_nll"]))
        assert int(selected["step"]) == metrics["models"][role]["selected_step"]
        replayed_scores[role] = {}
        for split, piece in (("validation", validation), ("test", test)):
            actual, losses = evaluate(model, piece)
            for key, value in actual.items():
                close(value, metrics["models"][role][split][key])
            replayed_scores[role][split] = actual
            if split == "test":
                close(losses, np.load(directory/"test_window_losses.npy", allow_pickle=False))
        engine = Backend(model)
        engines[role] = engine
        tokens, _ = reference(engine, list(b"Alice was "), 240, mode="sample", temperature=.8, seed=123)
        raw = b"Alice was "+bytes(tokens)
        assert raw == (directory/"sample.bin").read_bytes()
        assert raw.decode("utf-8", errors="replace")+"\n" == (directory/"sample.txt").read_text()
    _, labels = windows(test)
    counts = np.bincount(train, minlength=256).astype(float)+1.
    baseline = float(-np.log((counts/counts.sum())[labels]).mean())
    close(baseline, metrics["unigram"]["test_nll"])
    close(np.exp(baseline), metrics["unigram"]["test_perplexity"])
    errors = json.loads((ROOT/"runs/error_analysis.json").read_text())
    losses = np.load(ROOT/"runs/target/test_window_losses.npy", allow_pickle=False)
    assert [r["window"] for r in errors] == np.argsort(losses)[-5:][::-1].tolist()
    for row in errors:
        start = row["window"]*96
        raw = bytes(test[start:start+96].astype(np.uint8))
        assert raw.hex() == row["input_hex"]
        assert row["body_byte_offset"] == manifest["split_offsets_bytes"][2]+start
        close(row["nll"], losses[row["window"]])
    target, draft = engines["target"], engines["draft"]
    protocol = benchmark["protocol"]
    assert protocol["gammas"] == [1, 2, 4, 8]
    assert protocol["prompt_test_offsets"] == [0, 256, 512, 768]
    assert protocol["generated_bytes"] == 48 and protocol["prompt_bytes"] == 16
    prompts = [test[offset:offset+16].tolist() for offset in protocol["prompt_test_offsets"]]
    assert prompts == benchmark["prompts"]
    for key, saved in benchmark["outputs"].items():
        mode, index, variant = key.split("/")
        prompt = prompts[int(index)]
        common = dict(mode=mode, temperature=protocol["temperature"], seed=protocol["seed"])
        if variant == "reference":
            actual, stats = reference(target, prompt, 48, **common)
        else:
            actual, stats = generate(target, draft, prompt, 48, gamma=int(variant.split("-")[1]), **common)
        assert actual == saved["tokens"] and stats == saved["stats"], key
    extra_checks = []
    for index, prompt in enumerate(prompts):
        expected, _ = reference(target, prompt, 112)
        for gamma in protocol["gammas"]:
            actual, stats = generate(target, draft, prompt, 112, gamma=gamma)
            assert actual == expected and stats["boundary_tokens"] > 0
            extra_checks.append({"prompt_index": index, "gamma": gamma, "equal_tokens": len(actual),
                                 "boundary_tokens": stats["boundary_tokens"]})
    for row in benchmark["boundary_checks"]:
        actual, stats = generate(target, draft, row["prompt"], 16, gamma=4, mode=row["mode"], temperature=1., seed=123)
        assert actual == row["tokens"] and stats == row["stats"]
        assert stats["draft_calls"] == 0
    # Independently compare a trained target's block logits with its full prefix.
    prompt = prompts[0]
    _, cache = target.prefill(prompt)
    block = benchmark["outputs"]["greedy/0/reference"]["tokens"][:8]
    cached, _ = target.consume(block, cache)
    full = target.model(tf.constant([prompt+block], tf.int32)).numpy()[0, len(prompt):]
    close(cached, full, atol=2e-5)
    cache_max_error = float(np.max(np.abs(cached-full)))
    variants = ["reference", "gamma-1", "gamma-2", "gamma-4", "gamma-8"]
    assert len(benchmark["observations"]) == 2*4*5*protocol["trials"]
    for mode in ("greedy", "sample"):
        rows = [r for r in benchmark["observations"] if r["mode"] == mode]
        baseline_ms = float(np.median([r["milliseconds"] for r in rows if r["variant"] == "reference"]))
        for summary in [r for r in benchmark["summaries"] if r["mode"] == mode]:
            selected = [r for r in rows if r["variant"] == summary["variant"]]
            assert len(selected) == 4*protocol["trials"] == summary["requests"]
            assert len({(r["trial"], r["prompt_index"]) for r in selected}) == len(selected)
            for row in selected:
                shift = (row["trial"]+row["prompt_index"]) % 5
                order = variants[shift:]+variants[:shift]
                if row["trial"] % 2:
                    order = order[::-1]
                assert order[row["order_position"]] == row["variant"]
                stats = benchmark["outputs"][f"{mode}/{row['prompt_index']}/{row['variant']}"]["stats"]
                assert all(row[k] == value for k, value in stats.items())
                assert np.isfinite(row["milliseconds"]) and row["milliseconds"] > 0
            times = [r["milliseconds"] for r in selected]
            median = float(np.median(times))
            close(median, summary["median_ms"])
            close(np.percentile(times, 95), summary["p95_ms"])
            close(48*1000/median, summary["median_bytes_per_second"])
            close(baseline_ms/median, summary["speedup_vs_cached_target"])
            proposed = sum(r["proposed"] for r in selected)
            if proposed:
                close(sum(r["accepted"] for r in selected)/proposed, summary["acceptance_fraction"])
            else:
                assert summary["acceptance_fraction"] is None
            for metric, field in (("mean_target_calls", "target_calls"), ("mean_draft_calls", "draft_calls"),
                                  ("mean_verification_rounds", "rounds"), ("mean_rejected_rounds", "rejected_rounds")):
                close(np.mean([r[field] for r in selected]), summary[metric])
    assert benchmark["matched_greedy_tokens"] == 768
    assert benchmark["graph_traces_after_benchmark"] == {"target": {"prefill": 1, "consume": 1}, "draft": {"prefill": 1, "consume": 1}}
    sampler = sampler_audit()
    assert sampler == json.loads((ROOT/"runs/sampler_check.json").read_text())
    readme = (ROOT/"README.md").read_text()
    assert readme.startswith("# SpecForge\n") and "## Reproduce" not in readme
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme)
    assert len(images) == 6
    assert all(name.startswith("figures/") and (ROOT/name).is_file() for name in images)
    metadata = json.loads((ROOT/"repository.json").read_text())
    assert len(metadata["description"]) < 160 and metadata["corpus_redistributed"] is False
    manifest_files = {}
    for path in files():
        content = path.read_bytes()
        relative = path.relative_to(ROOT).as_posix()
        git_sha = hashlib.sha1(b"blob "+str(len(content)).encode()+b"\0"+content).hexdigest()
        manifest_files[relative] = {"bytes": len(content), "sha256": sha256(content), "git_blob_sha1": git_sha}
    result = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "status": "passed",
              "full_public_source_byte_match": True, "source_bytes": len(fresh), "corpus_bytes": manifest["body_bytes"],
              "replayed_scores": replayed_scores, "training_schedule_recreated": True,
              "checkpoint_samples_replayed": True, "benchmark_saved_ids_and_stats_replayed": True,
              "benchmark_observations": len(benchmark["observations"]), "timing_summary_recomputed": True,
              "wall_clock_timings_rerun": False, "trained_cache_vs_full_prefix_max_abs_logit_error": cache_max_error,
              "additional_greedy_boundary_checks": extra_checks,
              "additional_matched_greedy_tokens": sum(r["equal_tokens"] for r in extra_checks),
              "sampler_check_replayed": True, "valid_readme_images": images,
              "tests": {"count": 15, "result": "passed", "command": "python -m unittest discover -s tests -v",
                        "scope": "Executed before model training; timing results measured separately."},
              "files": manifest_files,
              "notes": "Artifact hashes exclude this audit itself, downloaded corpus bytes, Python caches and partial results. This audit replays evaluation and generation; it does not retrain or promise matching wall-clock measurements."}
    (ROOT/"audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"status": "passed", "files": len(manifest_files), "benchmark_observations": len(benchmark["observations"]),
                      "additional_matched_greedy_tokens": result["additional_matched_greedy_tokens"], "cache_max_abs_error": cache_max_error}))


if __name__ == "__main__":
    main()
