#!/usr/bin/env python3
"""Validate portable repository metadata without building or running a model."""

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    json_files = sorted(ROOT.rglob("*.json"))
    for path in json_files:
        load_json(path)

    python_files = sorted(ROOT.rglob("*.py"))
    for path in python_files:
        ast.parse(path.read_text(), filename=str(path.relative_to(ROOT)))

    source = load_json(ROOT / "source.json")
    patch_manifest = load_json(ROOT / "patches/manifest.json")
    manifest_by_patch = {entry["patch"]: entry for entry in patch_manifest}
    if len(manifest_by_patch) != len(patch_manifest):
        raise RuntimeError("Duplicate patch path in patches/manifest.json")

    referenced_patches = {
        f"patches/{patch}"
        for patches in source["profiles"].values()
        for patch in patches
    }
    if referenced_patches != set(manifest_by_patch):
        missing = sorted(referenced_patches - set(manifest_by_patch))
        unreferenced = sorted(set(manifest_by_patch) - referenced_patches)
        raise RuntimeError(
            f"Patch/profile mismatch: missing={missing}, unreferenced={unreferenced}"
        )

    for relative, entry in manifest_by_patch.items():
        path = ROOT / relative
        if sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Patch checksum mismatch: {relative}")
        for output in entry["files"]:
            digest = output["sha256_after"]
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise RuntimeError(f"Invalid reconstructed-file checksum: {relative}")

    fixture_manifest = load_json(ROOT / "benchmarks/fixtures/manifest.json")
    for relative, expected in fixture_manifest["files"].items():
        path = ROOT / "benchmarks/fixtures" / relative
        if sha256(path) != expected:
            raise RuntimeError(f"Fixture checksum mismatch: {relative}")

    tokens = load_json(ROOT / "benchmarks/fixtures/prompt-tokens.json")
    if len(tokens) != fixture_manifest["prompt_tokens"]:
        raise RuntimeError(
            f"Fixture token count mismatch: {len(tokens)} != "
            f"{fixture_manifest['prompt_tokens']}"
        )

    models = load_json(ROOT / "models.json")
    for filename, digest in models["files"].items():
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise RuntimeError(f"Invalid model checksum: {filename}")

    required_docs = (
        "README.md",
        "THIRD_PARTY_NOTICES.md",
        "handoff.md",
        "docs/BENCHMARKS.md",
        "docs/EXPERIMENTS.md",
        "docs/VALIDATION.md",
    )
    for relative in required_docs:
        if not (ROOT / relative).is_file():
            raise RuntimeError(f"Missing required documentation: {relative}")

    print(
        json.dumps(
            {
                "status": "passed",
                "json_files": len(json_files),
                "python_files": len(python_files),
                "patches": len(patch_manifest),
                "profiles": sorted(source["profiles"]),
                "fixture_tokens": len(tokens),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
