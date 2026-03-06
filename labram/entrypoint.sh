#!/bin/bash
set -euo pipefail

# LaBraM EEG Embedding Extraction
#
# Usage:
#   docker run --gpus all -v /data:/data labram-embeddings /data/input /data/output
#   docker run --gpus all -v /data:/data labram-embeddings /data/input/file.edf /data/output
#   docker run --gpus all -v /data:/data labram-embeddings --config /data/config.yaml /data/input /data/output
#
# Options:
#   --backend braindecode|original   (default: braindecode)
#   --batch-size N                   (default: 4)
#   --window-size SEC                (default: 10)
#   --skip-start SEC                 (default: 0)
#   --max-duration SEC               (default: 0, meaning use all)
#   --notch-freq HZ                  (default: 50)
#   --bandpass-low HZ                (default: 0.1)
#   --bandpass-high HZ               (default: 75)
#   --config PATH                    (YAML config file)
#   --verbose                        (print progress)

CONFIG_FILE=""
CLI_OVERRIDES=""
VERBOSE=""
INPUT_PATH=""
OUTPUT_PATH=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --backend)
            CLI_OVERRIDES="${CLI_OVERRIDES} model.backend=$2"
            shift 2
            ;;
        --batch-size)
            CLI_OVERRIDES="${CLI_OVERRIDES} model.batch_size=$2"
            shift 2
            ;;
        --window-size)
            CLI_OVERRIDES="${CLI_OVERRIDES} preprocessing.window_size_sec=$2"
            shift 2
            ;;
        --skip-start)
            CLI_OVERRIDES="${CLI_OVERRIDES} preprocessing.skip_start_sec=$2"
            shift 2
            ;;
        --max-duration)
            CLI_OVERRIDES="${CLI_OVERRIDES} preprocessing.max_duration_sec=$2"
            shift 2
            ;;
        --notch-freq)
            CLI_OVERRIDES="${CLI_OVERRIDES} preprocessing.notch_freq=$2"
            shift 2
            ;;
        --bandpass-low)
            CLI_OVERRIDES="${CLI_OVERRIDES} preprocessing.bandpass_low=$2"
            shift 2
            ;;
        --bandpass-high)
            CLI_OVERRIDES="${CLI_OVERRIDES} preprocessing.bandpass_high=$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            if [[ -z "$INPUT_PATH" ]]; then
                INPUT_PATH="$1"
            elif [[ -z "$OUTPUT_PATH" ]]; then
                OUTPUT_PATH="$1"
            else
                echo "Unexpected argument: $1" >&2
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate required arguments
if [[ -z "$INPUT_PATH" || -z "$OUTPUT_PATH" ]]; then
    echo "Usage: $0 [OPTIONS] INPUT_PATH OUTPUT_PATH" >&2
    echo "" >&2
    echo "INPUT_PATH can be a single .edf file or a directory containing .edf files." >&2
    echo "OUTPUT_PATH is the directory where .zarr files will be written." >&2
    exit 1
fi

# Build Python command
PYTHON_ARGS=""

if [[ -n "$CONFIG_FILE" ]]; then
    PYTHON_ARGS="--config ${CONFIG_FILE}"
fi

for override in $CLI_OVERRIDES; do
    PYTHON_ARGS="${PYTHON_ARGS} --override ${override}"
done

if [[ -n "$VERBOSE" ]]; then
    PYTHON_ARGS="${PYTHON_ARGS} --verbose"
fi

exec python3 -m labram.main ${PYTHON_ARGS} "${INPUT_PATH}" "${OUTPUT_PATH}"
