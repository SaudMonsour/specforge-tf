"""Generate bytes with a target alone or a speculative draft/target pair."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import argparse
import json
import sys
from pathlib import Path
import tensorflow as tf
from specforge import generate, reference
from specforge.model import Backend, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--target", type=Path, default=root/"runs/target")
    parser.add_argument("--draft", type=Path, default=root/"runs/draft")
    parser.add_argument("--prompt", default="Alice was ")
    parser.add_argument("--bytes", type=int, default=128)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--mode", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--temperature", type=float, default=.8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    target = Backend(load_checkpoint(args.target))
    if args.reference:
        tokens, stats = reference(target, list(args.prompt.encode("utf-8")), args.bytes, mode=args.mode, temperature=args.temperature, seed=args.seed)
    else:
        draft = Backend(load_checkpoint(args.draft))
        tokens, stats = generate(target, draft, list(args.prompt.encode("utf-8")), args.bytes, gamma=args.gamma, mode=args.mode, temperature=args.temperature, seed=args.seed)
    print((args.prompt.encode("utf-8")+bytes(tokens)).decode("utf-8", errors="replace"))
    print(json.dumps(stats), file=sys.stderr)


if __name__ == "__main__":
    main()
