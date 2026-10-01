# Using SpecForge

The included checkpoints work without downloading the corpus. Python 3.12 and the measured dependency versions are recorded in `environment.lock.txt`.

## Generate text

Install the study dependencies from the repository root:

```bash
pip install -r requirements.txt
python generate.py --prompt "Alice was " --bytes 48 --gamma 4 --mode greedy
python generate.py --prompt "Alice was " --bytes 48 --reference --mode greedy
python generate.py --prompt "Alice was " --bytes 48 --gamma 4 --mode sample --temperature 0.8 --seed 123
```

Generation statistics go to stderr; decoded text goes to stdout. One token is one UTF-8 byte, so `--bytes` is a byte budget, not a word count. Individual outputs can contain invalid UTF-8; the CLI displays replacement characters without changing the sampled IDs. Use the Python API to retain exact IDs.

The two greedy commands should emit identical byte IDs. The sampled command and sampled target-only command do not generally emit identical sequences at the same seed: they consume random numbers differently. Rejection correction preserves the target conditional distribution in ideal arithmetic.

The fixed context is 96 bytes. Once a request reaches that limit, the engine re-prefills the latest 96 bytes and uses target-only generation. This preserves the reference decoder's position-reset semantics and discards deeper cached states that depend on removed context. It also ends the speculative acceleration for the rest of that request.

## Integrate the engine

The sampler module depends on NumPy, independently of TensorFlow. To install the library with its TensorFlow adapter:

```bash
pip install -e '.[tensorflow]'
```

```python
from specforge import generate, reference
from specforge.model import Backend, load_checkpoint

target = Backend(load_checkpoint("runs/target"))
draft = Backend(load_checkpoint("runs/draft"))
tokens, stats = generate(target, draft, list(b"Alice was "), 48,
                         gamma=4, mode="greedy")
print(bytes(tokens))
print(stats)
```

Backends must implement these operations:

| Member | Contract |
| :--- | :--- |
| `context`, `vocabulary` | Identical draft/target limits and exact token-ID mapping. |
| `prefill(ids)` | Return next-token logits shaped `[vocabulary]` and a new request-local cache. |
| `consume(ids, cache)` | Return logits shaped `[len(ids), vocabulary]` and the extended cache. Row `i` predicts the token **after** `ids[i]`. |
| `truncate(cache, length)` | Return a cache containing exactly the committed prefix. Discard all rejected suffix states. |

Caches belong to requests. A shared backend must not keep mutable request state. The included adapter supports batch one and float32. Other frameworks can implement the same interface, but adapters, token mappings, cache semantics and numerics must be tested independently; this repository does not supply pretrained-model adapters or continuous batching.

`generate` returns only newly generated IDs and a statistics dictionary. `accepted / proposed` counts every draft proposal in its denominator, including suffix proposals discarded after an earlier rejection. `target_calls` includes the initial prefill and verification/fallback calls; `rounds` counts only speculative verification rounds.

## Inspect and extend the study

`prepare.py` downloads the complete public source and checks the manifest hashes before preparing contiguous splits. `study.py` trains both roles from scratch using the saved seed and fixed byte-window schedule, and refuses to overwrite an existing `runs/` directory. `benchmark.py` likewise preserves its original timing results. Use a separate checkout or output directory when making a new experiment; keep the published observations intact.

`plots.py` renders the figures from saved measurements. `verify.py` checks the source, checkpoints, held-out scores, raw benchmark observations, generated IDs and artifact hashes. It does not retrain the models or claim to reproduce wall-clock latency exactly. A Gutenberg header or text update can change the source hash; the original manifest remains the record of the source used for this run.

Run the focused correctness suite with:

```bash
python -m unittest discover -s tests -v
```

Useful next experiments include a distilled draft, shared tokenizer adapters, vectorized draft proposal generation and a sliding-position model that can retain valid KV states beyond this short learned-position context. Measure each against its own cached target baseline.
