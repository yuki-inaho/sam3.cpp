#!/usr/bin/env python3
"""Losslessly convert the SAM 3.1 ConvRot INT8 checkpoint to typed GGUF v3."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from sam31_gguf.format import GGUFFile, SafeTensorFile, compare_stores, convert


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true", help="Compare every tensor byte after conversion")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-test-fixture", action="store_true", help="Permit explicitly marked synthetic tensors")
    args = parser.parse_args()
    try:
        result = convert(args.input, args.output, allow_test_fixture=args.allow_test_fixture,
                         overwrite=args.overwrite)
        if args.verify:
            result["roundtrip"] = compare_stores(SafeTensorFile(args.input), GGUFFile(args.output))
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError) as exc:
        print(f"Conversion failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
