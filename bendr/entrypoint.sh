#!/bin/bash
set -euo pipefail

usage() {
    cat >&2 <<EOF
Usage: $0 [OPTIONS] INPUT_PATH OUTPUT_PATH

Extract BENDR embeddings from EDF files into Zarr stores.

INPUT_PATH   Single .edf file or directory of .edf files
OUTPUT_PATH  Directory for .zarr output

Options:
  --help                           Show this help
  --dry-run                        List files that would be processed, then exit
  --batch-size N                   Inference batch size (default: 4)
  --window-size SEC                EEG window length (default: 10)
  --skip-start SEC                 Skip N seconds from start (default: 0)
  --max-duration SEC               Max seconds to use, 0=all (default: 0)
  --notch-freq HZ                  Notch filter frequency (default: 50)
  --bandpass-low HZ                Bandpass low cutoff (default: 0.1)
  --bandpass-high HZ               Bandpass high cutoff (default: 75)
  --config PATH                    YAML config file
  --verbose                        Debug logging

Examples:
  $0 /data/input/recording.edf /data/output
  $0 --batch-size 8 /data/input /data/output
  $0 --config /data/config.yaml /data/input /data/output
EOF
}

CONFIG_FILE=""
CLI_OVERRIDES=""
VERBOSE=""
DRY_RUN=false
INPUT_PATH=""
OUTPUT_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            usage
            exit 0
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
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
            echo "Try '$0 --help' for usage." >&2
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

if [[ -z "$INPUT_PATH" || -z "$OUTPUT_PATH" ]]; then
    usage
    exit 1
fi

# Dry run: list EDF files and exit
if [[ "$DRY_RUN" == true ]]; then
    echo "Dry run - files that would be processed:"
    if [[ -f "$INPUT_PATH" ]]; then
        echo "  $INPUT_PATH"
    elif [[ -d "$INPUT_PATH" ]]; then
        count=0
        while IFS= read -r -d '' f; do
            echo "  $f"
            count=$((count + 1))
        done < <(find "$INPUT_PATH" -name '*.edf' -print0 | sort -z)
        echo "Total: $count file(s)"
    else
        echo "Input path does not exist: $INPUT_PATH" >&2
        exit 1
    fi
    echo "Output directory: $OUTPUT_PATH"
    [[ -n "$CONFIG_FILE" ]] && echo "Config: $CONFIG_FILE"
    exit 0
fi

PYTHON_ARGS=""

# Default to bundled config if none specified
if [[ -z "$CONFIG_FILE" ]]; then
    CONFIG_FILE="/app/bendr/config.yaml"
fi
PYTHON_ARGS="--config ${CONFIG_FILE}"

for override in $CLI_OVERRIDES; do
    PYTHON_ARGS="${PYTHON_ARGS} --override ${override}"
done

if [[ -n "$VERBOSE" ]]; then
    PYTHON_ARGS="${PYTHON_ARGS} --verbose"
fi

exec python3 -m bendr.main ${PYTHON_ARGS} "${INPUT_PATH}" "${OUTPUT_PATH}"
