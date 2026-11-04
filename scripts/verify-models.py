#!/usr/bin/env python3
"""Verify existing model artifacts; never download or replace them."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=Path)
args = parser.parse_args()
manifest = json.loads((Path(__file__).resolve().parents[1] / 'models.json').read_text())
for name, expected in manifest['files'].items():
    path = args.directory / name
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected:
        raise SystemExit(f'FAIL {name}: expected {expected}, got {actual}')
    print(f'OK {name}')
