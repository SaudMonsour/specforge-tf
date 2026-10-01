"""Counterbalanced request latency against a cached target-only baseline."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import tensorflow as tf
from prepare import read_data
from specforge import generate, reference
from specforge.model import Backend, load_checkpoint

ROOT = Path(__file__).resolve().parent
GAMMAS = (1, 2, 4, 8)
OFFSETS = (0, 256, 512, 768)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=12)
    args = parser.parse_args()
    if args.trials < 5:
        parser.error("at least five measured trials required")
    path = ROOT/"runs/benchmark.json"
    if path.exists():
        parser.error("benchmark exists; preserve the original timing observations")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    manifest, (_, _, test) = read_data()
    target = Backend(load_checkpoint(ROOT/"runs/target"))
    draft = Backend(load_checkpoint(ROOT/"runs/draft"))
    prompts = [test[offset:offset+16].tolist() for offset in OFFSETS]
    variants = ["reference", *[f"gamma-{g}" for g in GAMMAS]]
    temperature, seed, count = 1., 123, 48
    def request(name, prompt, mode, budget=count):
        if name == "reference":
            return reference(target, prompt, budget, mode, temperature, seed)
        return generate(target, draft, prompt, budget, gamma=int(name.split("-")[1]),
                        mode=mode, temperature=temperature, seed=seed)
    observations, outputs, checks = [], {}, []
    started = datetime.now(timezone.utc).isoformat()
    for mode in ("greedy", "sample"):
        for prompt_index, prompt in enumerate(prompts):
            expected = request("reference", prompt, mode)[0] if mode == "greedy" else None
            for variant in variants:
                for _ in range(3):
                    tokens, stats = request(variant, prompt, mode)
                if expected is not None:
                    assert tokens == expected, (variant, prompt_index)
                    if variant != "reference":
                        checks.append({"mode": mode, "prompt_index": prompt_index, "variant": variant,
                                       "equal_tokens": count})
                outputs[f"{mode}/{prompt_index}/{variant}"] = {"tokens": tokens, "stats": stats}
        for trial in range(args.trials):
            for prompt_index, prompt in enumerate(prompts):
                shift = (trial+prompt_index) % len(variants)
                order = variants[shift:]+variants[:shift]
                if trial % 2:
                    order = order[::-1]
                for position, variant in enumerate(order):
                    began = time.perf_counter_ns()
                    tokens, stats = request(variant, prompt, mode)
                    elapsed = (time.perf_counter_ns()-began)/1e6
                    assert tokens == outputs[f"{mode}/{prompt_index}/{variant}"]["tokens"]
                    observations.append({"mode": mode, "prompt_index": prompt_index, "trial": trial,
                                         "order_position": position, "variant": variant, "milliseconds": elapsed,
                                         **stats})
        print(json.dumps({"completed_mode": mode, "requests": len(observations)}), flush=True)
    summaries = []
    for mode in ("greedy", "sample"):
        rows = [r for r in observations if r["mode"] == mode]
        baseline = np.median([r["milliseconds"] for r in rows if r["variant"] == "reference"])
        for variant in variants:
            selected = [r for r in rows if r["variant"] == variant]
            latencies = np.array([r["milliseconds"] for r in selected])
            proposed = sum(r["proposed"] for r in selected)
            accepted = sum(r["accepted"] for r in selected)
            summaries.append({"mode": mode, "variant": variant, "requests": len(selected),
                              "median_ms": float(np.median(latencies)), "p95_ms": float(np.percentile(latencies, 95)),
                              "median_bytes_per_second": float(count*1000/np.median(latencies)),
                              "speedup_vs_cached_target": float(baseline/np.median(latencies)),
                              "acceptance_fraction": accepted/proposed if proposed else None,
                              "mean_target_calls": float(np.mean([r["target_calls"] for r in selected])),
                              "mean_draft_calls": float(np.mean([r["draft_calls"] for r in selected])),
                              "mean_verification_rounds": float(np.mean([r["rounds"] for r in selected])),
                              "mean_rejected_rounds": float(np.mean([r["rejected_rounds"] for r in selected]))})
    boundary = []
    prompt = test[1024:1120].tolist()
    for mode in ("greedy", "sample"):
        expected, ref_stats = request("reference", prompt, mode, 16)
        actual, stats = request("gamma-4", prompt, mode, 16)
        assert expected == actual and stats["draft_calls"] == 0
        boundary.append({"mode": mode, "prompt": prompt, "tokens": actual, "stats": stats,
                         "reference_stats": ref_stats, "equal_tokens": 16})
    result = {"started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": manifest["source_sha256"], "body_sha256": manifest["body_sha256"],
              "protocol": {"prompt_test_offsets": list(OFFSETS), "prompt_bytes": 16, "generated_bytes": count,
                           "gammas": list(GAMMAS), "temperature": temperature, "seed": seed, "trials": args.trials,
                           "warmup_requests_per_prompt_mode_variant": 3, "timed_load_or_trace": False,
                           "baseline": "Cached target-only generation, not repeated full-prefix decoding.",
                           "order": "Rotate by trial+prompt; reverse on odd trials; same four prompts in each variant.",
                           "timing_scope": "Complete request including target/draft prefill, Python sampling, verification, rollback and logit materialization. Checkpoint loading and warmup excluded.",
                           "acceptance_denominator": "All proposed tokens, including later proposals discarded at an earlier rejection."},
              "runtime": {"python": platform.python_version(), "tensorflow": tf.__version__, "platform": platform.platform(),
                          "intra_threads": 2, "inter_threads": 2, "oneDNN": False, "devices": [str(d) for d in tf.config.list_physical_devices()]},
              "prompts": prompts, "outputs": outputs, "summaries": summaries, "observations": observations,
              "greedy_checks": checks, "matched_greedy_tokens": sum(r["equal_tokens"] for r in checks),
              "boundary_checks": boundary, "graph_traces_after_benchmark": {"target": target.traces(), "draft": draft.traces()}}
    path.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"summaries": summaries, "matched_greedy_tokens": result["matched_greedy_tokens"],
                      "graph_traces": result["graph_traces_after_benchmark"]}), flush=True)


if __name__ == "__main__":
    main()
