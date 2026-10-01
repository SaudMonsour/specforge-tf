"""Train fixed small and larger byte decoders on the same sampled windows."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import argparse
import csv
import json
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import tensorflow as tf
from prepare import read_data
from specforge.model import Config, Decoder, Backend, load_checkpoint
from specforge import reference

ROOT = Path(__file__).resolve().parent


def windows(piece, context=96):
    starts = np.arange(0, len(piece)-context, context)
    indices = starts[:, None]+np.arange(context)[None]
    return piece[indices], piece[indices+1]


def evaluate(model, piece):
    inputs, targets = windows(piece, model.config.context)
    @tf.function(input_signature=[tf.TensorSpec([None, model.config.context], tf.int32)])
    def forward(ids):
        return model(ids, training=False)
    losses, accuracies = [], []
    for start in range(0, len(inputs), 16):
        logits = forward(inputs[start:start+16])
        labels = targets[start:start+16]
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        losses.extend(tf.reduce_mean(loss, -1).numpy().tolist())
        accuracies.extend(np.mean(logits.numpy().argmax(-1)==labels, -1).tolist())
    nll = float(np.mean(losses))
    return {"nll_nats_per_byte": nll, "bits_per_byte": nll/np.log(2), "perplexity": float(np.exp(nll)),
            "next_byte_accuracy": float(np.mean(accuracies)), "windows": len(inputs),
            "evaluated_bytes": int(inputs.size), "window_nll_sd": float(np.std(losses, ddof=1))}, np.asarray(losses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.steps, args.batch) < 1:
        parser.error("positive steps/batch required")
    output = ROOT/"runs"
    if output.exists():
        parser.error("runs already exists; preserve previous measurements")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    manifest, (train, validation, test) = read_data()
    output.mkdir()
    started = datetime.now(timezone.utc).isoformat()
    schedule = np.random.default_rng(args.seed).integers(0, len(train)-96, size=(args.steps, args.batch))
    np.save(output/"training_starts.npy", schedule)
    models, history = {}, []
    configurations = {"draft": Config(width=48, heads=4, layers=1, hidden=96), "target": Config()}
    for name, config in configurations.items():
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(args.seed)
        model = Decoder(config)
        model(tf.zeros([1, config.context], tf.int32))
        directory = output/name
        directory.mkdir()
        (directory/"config.json").write_text(json.dumps(asdict(config), indent=2)+"\n")
        optimizer = tf.keras.optimizers.Adam(learning_rate=.001, global_clipnorm=1.)
        optimizer.build(model.trainable_variables)
        @tf.function(input_signature=[tf.TensorSpec([args.batch, 96], tf.int32), tf.TensorSpec([args.batch, 96], tf.int32)])
        def train_step(ids, labels):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=model(ids)))
            optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
            return loss
        best, best_step = float("inf"), 0
        began = time.perf_counter()
        for step, starts in enumerate(schedule, 1):
            indices = starts[:, None]+np.arange(96)[None]
            loss = float(train_step(train[indices], train[indices+1]))
            if step == 1 or step % 100 == 0 or step == args.steps:
                val, _ = evaluate(model, validation)
                row = {"model": name, "step": step, "train_nll": loss, "validation_nll": val["nll_nats_per_byte"], "elapsed_seconds": time.perf_counter()-began}
                history.append(row)
                print(json.dumps(row), flush=True)
                if val["nll_nats_per_byte"] < best:
                    best, best_step = val["nll_nats_per_byte"], step
                    model.save_weights(directory/"model.weights.h5")
        model.load_weights(directory/"model.weights.h5")
        val, _ = evaluate(model, validation)
        models[name] = {"parameters": model.count_params(), "selected_step": best_step, "validation": val,
                        "training_seconds_including_validation_and_saving": time.perf_counter()-began}
        (output/"results.partial.json").write_text(json.dumps(models, indent=2)+"\n")
    errors = []
    for name in models:
        model = load_checkpoint(output/name)
        heldout, losses = evaluate(model, test)
        models[name]["test"] = heldout
        np.save(output/name/"test_window_losses.npy", losses)
        generated, _ = reference(Backend(model), list(b"Alice was "), 240, mode="sample", temperature=.8, seed=123)
        (output/name/"sample.bin").write_bytes(b"Alice was "+bytes(generated))
        (output/name/"sample.txt").write_text((b"Alice was "+bytes(generated)).decode("utf-8", errors="replace")+"\n")
        if name == "target":
            for index in np.argsort(losses)[-5:][::-1]:
                start = int(index)*96
                errors.append({"window": int(index), "body_byte_offset": manifest["split_offsets_bytes"][2]+start,
                               "nll": float(losses[index]), "input_hex": bytes(test[start:start+96].astype(np.uint8)).hex(),
                               "decoded_preview": bytes(test[start:start+96].astype(np.uint8)).decode("utf-8", errors="replace")})
    _, labels = windows(test)
    counts = np.bincount(train, minlength=256).astype(float)+1.
    baseline = float(-np.log((counts/counts.sum())[labels]).mean())
    result = {"started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
              "steps": args.steps, "batch": args.batch, "context": 96, "sampled_byte_targets_per_model": args.steps*args.batch*96,
              "source_sha256": manifest["source_sha256"], "body_sha256": manifest["body_sha256"],
              "models": models, "unigram": {"test_nll": baseline, "test_perplexity": float(np.exp(baseline))},
              "selection": "Independent best validation NLL checkpoints; draft/target roles fixed before evaluation. No test-driven size/seed/block selection.",
              "controls": "Identical sampled windows/order, seed, optimizer, learning rate and steps; widths/layers/parameter counts differ. No distillation or pretrained weights.",
              "runtime": {"python": platform.python_version(), "tensorflow": tf.__version__, "platform": platform.platform(), "intra_threads": 2, "inter_threads": 2, "oneDNN": False, "devices": [str(d) for d in tf.config.list_physical_devices()]}}
    (output/"metrics.json").write_text(json.dumps(result, indent=2)+"\n")
    (output/"error_analysis.json").write_text(json.dumps(errors, indent=2)+"\n")
    with (output/"training_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
