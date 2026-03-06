"""REVE embedding extraction CLI entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from shared.config import load_config
from reve.extract import load_model, process_edf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract REVE embeddings from EDF files",
    )
    parser.add_argument("input_path", type=Path, help="EDF file or directory")
    parser.add_argument("output_path", type=Path, help="Output directory for .zarr files")
    parser.add_argument("--config", type=Path, default=None, help="YAML config file")
    parser.add_argument(
        "--override", action="append", default=[],
        help="Config override in key=value format (e.g., model.batch_size=4)",
    )
    parser.add_argument("--hf-token", default=None, help="HuggingFace token (overrides config/env)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Build CLI overrides dict
    cli_overrides = {}
    for override in args.override:
        if "=" not in override:
            logger.error("Invalid override format (expected key=value): %s", override)
            sys.exit(1)
        key, value = override.split("=", 1)
        cli_overrides[key] = value

    # HF token from CLI takes highest priority
    if args.hf_token:
        cli_overrides["model.hf_token"] = args.hf_token

    # Load config
    config = load_config(yaml_path=args.config, cli_overrides=cli_overrides or None)

    # Check for HF token
    if not config.model.hf_token:
        logger.warning(
            "No HuggingFace token found. REVE is a gated model - "
            "provide a token via --hf-token, HF_TOKEN env var, or config file."
        )

    # Discover EDF files
    input_path = args.input_path
    if input_path.is_file():
        edf_files = [input_path]
    elif input_path.is_dir():
        edf_files = sorted(input_path.glob("**/*.edf"))
        if not edf_files:
            logger.error("No .edf files found in %s", input_path)
            sys.exit(1)
    else:
        logger.error("Input path does not exist: %s", input_path)
        sys.exit(1)

    logger.info("Found %d EDF file(s) to process", len(edf_files))

    # Ensure output directory exists
    output_dir = args.output_path
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model once
    model, pos_bank = load_model(
        size=config.model.size,
        hf_token=config.model.hf_token,
        device=config.model.device,
    )

    # Process each EDF
    successes = 0
    failures = 0

    for edf_path in edf_files:
        logger.info("Processing: %s", edf_path.name)
        result = process_edf(edf_path, output_dir, config, model=model, pos_bank=pos_bank)
        if result is not None:
            successes += 1
            logger.info("  -> %s", result)
        else:
            failures += 1

    # Summary
    logger.info(
        "Complete: %d succeeded, %d failed out of %d files",
        successes, failures, len(edf_files),
    )

    if failures > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
