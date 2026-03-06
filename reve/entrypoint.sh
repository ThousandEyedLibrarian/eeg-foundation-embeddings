#!/bin/bash
set -euo pipefail

# REVE EEG Embedding Extraction
#
# Usage:
#   docker run --gpus all -e HF_TOKEN=hf_xxx -v /data:/data reve-embeddings /data/input /data/output
#   docker run --gpus all -e HF_TOKEN=hf_xxx -v /data:/data reve-embeddings --config /data/config.yaml /data/input /data/output
#
# Options:
#   --hf-token TOKEN                 (HuggingFace token, overrides config/env)
#   --model-size base|large          (default: base, 512-dim vs 1250-dim)
#   --batch-size N                   (default: 2)
#   --window-size SEC                (default: 10)
#   --skip-start SEC                 (default: 0)
#   --max-duration SEC               (default: 0, meaning use all)
#   --notch-freq HZ                  (default: 50)
#   --bandpass-low HZ                (default: 0.5)
#   --bandpass-high HZ               (default: 99.5)
#   --normalisation zscore|robust|none (default: zscore)
#   --clip-std N                     (default: 15)
#   --config PATH                    (YAML config file)
#   --verbose                        (print progress)

CONFIG_FILE=""
CLI_OVERRIDES=""
HF_TOKEN_ARG=""
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
        --hf-token)
            HF_TOKEN_ARG="$2"
            shift 2
            ;;
        --model-size)
            CLI_OVERRIDES="${CLI_OVERRIDES} model.size=$2"
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
        --normalisation)
            CLI_OVERRIDES="${CLI_OVERRIDES} preprocessing.normalisation=$2"
            shift 2
            ;;
        --clip-std)
            CLI_OVERRIDES="${CLI_OVERRIDES} preprocessing.clip_std=$2"
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
    echo "" >&2
    echo "REVE is a gated model. Provide your HuggingFace token via:" >&2
    echo "  --hf-token TOKEN, HF_TOKEN env var, or model.hf_token in config YAML." >&2
    exit 1
fi

# Build Python command
PYTHON_ARGS=""

if [[ -n "$CONFIG_FILE" ]]; then
    PYTHON_ARGS="--config ${CONFIG_FILE}"
fi

if [[ -n "$HF_TOKEN_ARG" ]]; then
    PYTHON_ARGS="${PYTHON_ARGS} --hf-token ${HF_TOKEN_ARG}"
fi

for override in $CLI_OVERRIDES; do
    PYTHON_ARGS="${PYTHON_ARGS} --override ${override}"
done

if [[ -n "$VERBOSE" ]]; then
    PYTHON_ARGS="${PYTHON_ARGS} --verbose"
fi

exec python3 -m reve.main ${PYTHON_ARGS} "${INPUT_PATH}" "${OUTPUT_PATH}"
