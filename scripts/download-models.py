#!/usr/bin/env python3
"""Explicitly download the pinned public model artifacts and verify their SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'models.json').read_text())
    args.directory.mkdir(parents=True, exist_ok=True)
    for name, expected in manifest['files'].items():
        path = args.directory / name
        if path.exists():
            if digest(path) != expected:
                raise RuntimeError(f'Existing artifact differs; leaving it untouched: {path}')
            print(f'Verified existing {name}', flush=True)
            continue
        url = f"https://huggingface.co/{manifest['repository']}/resolve/{manifest['revision']}/{name}"
        partial = path.with_name(path.name + '.part')
        print(f'Downloading {name}', flush=True)
        # Exclusive creation preserves any interrupted download for inspection.
        with partial.open('xb') as output, urllib.request.urlopen(url, timeout=120) as source:
            checksum = hashlib.sha256()
            while chunk := source.read(8 * 1024 * 1024):
                output.write(chunk)
                checksum.update(chunk)
        if checksum.hexdigest() != expected:
            raise RuntimeError(f'Checksum mismatch; evidence retained at {partial}')
        partial.rename(path)
        print(f'Verified {name}', flush=True)


if __name__ == '__main__':
    main()
