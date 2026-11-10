"""Fetch the three papers' LaTeX sources from arXiv at exactly the versions the brief names.

  python -m corpus.fetch

Each source goes to data/raw/<id>v<N>/ (git-ignored): the downloaded archive as-is, plus an extracted src/ folder.
The archive is untrusted input. Members are extracted only if they resolve inside src/ and are regular files or
directories; links, devices and absolute or '..' paths are skipped. Nothing extracted is ever executed.
A manifest (data/raw/manifest.json) records URL, byte size and SHA-256 so the exact inputs can be checked.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

PAPERS = {
    "AIAYN": ("1706.03762", 7, "Attention Is All You Need"),
    "MOSHI": ("2410.00037", 1, "Moshi: a speech-text foundation model for real-time dialogue"),
    "HNET": ("2507.07955", 2, "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"),
}


def safe_extract(blob: bytes, dest: Path) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    written = []
    try:
        tf = tarfile.open(fileobj=io.BytesIO(blob), mode="r:*")
    except tarfile.ReadError:
        # Not a tar: arXiv serves a single gzipped .tex for one-file submissions.
        data = gzip.decompress(blob) if blob[:2] == b"\x1f\x8b" else blob
        (dest / "main.tex").write_bytes(data)
        return ["main.tex"]
    with tf:
        for m in tf.getmembers():
            target = (dest / m.name).resolve()
            if not str(target).startswith(str(root)) or not (m.isfile() or m.isdir()):
                print(f"   skipped unsafe member: {m.name!r}")
                continue
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            f = tf.extractfile(m)
            if f is not None:
                target.write_bytes(f.read())
                written.append(m.name)
    return written


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with httpx.Client(timeout=60, follow_redirects=True,
                      headers={"User-Agent": "relay-takehome-corpus-fetch/0.1"}) as c:
        for key, (aid, ver, title) in PAPERS.items():
            url = f"https://arxiv.org/e-print/{aid}v{ver}"
            folder = RAW / f"{aid}v{ver}"
            archive = folder / "source.bin"
            if archive.exists():
                blob = archive.read_bytes()
                print(f"{key}: using cached {archive.relative_to(ROOT)}")
            else:
                r = c.get(url)
                r.raise_for_status()
                blob = r.content
                folder.mkdir(parents=True, exist_ok=True)
                archive.write_bytes(blob)
                print(f"{key}: downloaded {len(blob):,} bytes from {url}")
            files = safe_extract(blob, folder / "src")
            manifest[key] = {"arxiv_id": aid, "version": ver, "title": title, "url": url, "bytes": len(blob),
                             "sha256": hashlib.sha256(blob).hexdigest(), "files": sorted(files),
                             "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            print(f"   {len(files)} files: {', '.join(sorted(f for f in files if f.endswith('.tex')))}")
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
