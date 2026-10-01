"""Fetch and hash a new public-domain corpus; retain raw bytes locally only."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
URL = "https://www.gutenberg.org/cache/epub/11/pg11.txt"


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def body_from_source(raw):
    text = raw.decode("utf-8-sig").replace("\r\n", "\n")
    start = "*** START OF THE PROJECT GUTENBERG EBOOK ALICE'S ADVENTURES IN WONDERLAND ***"
    end = "*** END OF THE PROJECT GUTENBERG EBOOK ALICE'S ADVENTURES IN WONDERLAND ***"
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError("source boundary markers changed")
    return (text.split(start)[1].split(end)[0].strip() + "\n").encode("utf-8")


def read_data():
    import numpy as np
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    raw = (ROOT / "data/source.txt").read_bytes()
    body = body_from_source(raw)
    if sha256(raw) != manifest["source_sha256"] or sha256(body) != manifest["body_sha256"]:
        raise ValueError("source differs from the executed study")
    if (ROOT / "data/body.bin").read_bytes() != body:
        raise ValueError("local corpus differs from source transformation")
    values = np.frombuffer(body, dtype=np.uint8).astype(np.int32)
    offsets = manifest["split_offsets_bytes"]
    return manifest, tuple(values[a:b] for a, b in zip(offsets[:-1], offsets[1:]))


def main():
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    with urlopen(URL, timeout=30) as response:
        raw = response.read(500_001)
        final_url = response.url
        modified = response.headers.get("Last-Modified")
    if len(raw) > 500_000:
        raise ValueError("unexpected source size")
    body = body_from_source(raw)
    if (data / "manifest.json").exists():
        existing = json.loads((data / "manifest.json").read_text())
        if sha256(raw) != existing["source_sha256"] or sha256(body) != existing["body_sha256"]:
            raise ValueError("public source changed; preserve the manifest and locate matching source bytes")
    else:
        n = len(body)
        manifest = {"dataset": "Alice's Adventures in Wonderland", "author": "Lewis Carroll",
                    "source": "https://www.gutenberg.org/ebooks/11", "download_url": final_url,
                    "fetched_at_utc": datetime.now(timezone.utc).isoformat(), "last_modified": modified,
                    "source_sha256": sha256(raw), "body_sha256": sha256(body),
                    "source_bytes": len(raw), "body_bytes": n, "vocabulary": 256,
                    "tokenization": "UTF-8 bytes, IDs 0..255; no fitted vocabulary or unknown tokens",
                    "split_offsets_bytes": [0, int(n*.8), int(n*.9), n],
                    "transformation": "Decode UTF-8 BOM if present; CRLF to LF; keep text between exact Gutenberg markers; strip outer whitespace; append one LF; encode UTF-8.",
                    "license": "Project Gutenberg identifies this book as public domain in the USA; source referenced, corpus not redistributed.",
                    "split_note": "Contiguous 80/10/10 byte splits. No input/target window crosses a split; recurring phrases are not deduplicated."}
        (data / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (data / "source.txt").write_bytes(raw)
    (data / "body.bin").write_bytes(body)
    read_data()
    print(json.dumps({"verified_source_bytes": len(raw), "body_bytes": len(body), "source_sha256": sha256(raw), "body_sha256": sha256(body)}))


if __name__ == "__main__":
    main()
