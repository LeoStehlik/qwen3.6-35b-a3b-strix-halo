#!/usr/bin/env python3
"""Reconstruct a pinned source profile without touching any running service."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def main():
    config = json.loads((ROOT / 'source.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=config['profiles'], default='prefill-stage2')
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--upstream', default=config['repository'], help='Git URL or an existing local clone for offline reconstruction')
    args = parser.parse_args()
    dest = (args.destination or ROOT / 'work' / args.profile).resolve()
    if dest.exists():
        parser.error(f'{dest} already exists; choose a new destination, preserving any local work')
    manifest = {m['patch']: m for m in json.loads((ROOT / 'patches/manifest.json').read_text())}
    patches = [ROOT / 'patches' / name for name in config['profiles'][args.profile]]
    for patch in patches:
        expected = manifest[str(patch.relative_to(ROOT))]['sha256']
        if hashlib.sha256(patch.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Patch checksum mismatch: {patch}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'clone', '--no-checkout', '--filter=blob:none', args.upstream, str(dest)], check=True)
    subprocess.run(['git', '-C', str(dest), 'checkout', '--detach', config['commit']], check=True)
    expected_files = {}
    for patch in patches:
        subprocess.run(['git', '-C', str(dest), 'apply', '--check', str(patch)], check=True)
        subprocess.run(['git', '-C', str(dest), 'apply', str(patch)], check=True)
        for item in manifest[str(patch.relative_to(ROOT))]['files']:
            expected_files[item['path']] = item['sha256_after']
    for rel, expected in expected_files.items():
        path = dest / rel
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        if actual != expected:
            raise RuntimeError(f'Reconstructed source differs from tested source: {rel}')
    subprocess.run(['git', '-C', str(dest), 'diff', '--check'], check=True)
    print(json.dumps({'profile': args.profile, 'source': str(dest), 'base': config['commit'],
                      'verified_changed_files': len(expected_files)}, indent=2))

if __name__ == '__main__':
    main()
