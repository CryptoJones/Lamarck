#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
#
# Run the full Lamarck eval suite end-to-end: Tier 1 -> 2 -> 3 ->
# aggregate. Each tier writes its TierResult JSON to <out-dir>/tier{N}.json;
# the L17 aggregate CLI then produces <out-dir>/final.json.
#
# Usage:
#   scripts/eval/run-all.sh <model_id> <base_url> <curriculum_jsonl> \
#       [--out-dir DIR] [--mock]
#
# Examples:
#   # Real run on a served pod:
#   scripts/eval/run-all.sh lamarck-g1 http://localhost:8000/v1 ./cur.jsonl
#
#   # Smoke run for CI - no torch / lm_eval / network needed:
#   scripts/eval/run-all.sh lamarck-g1 http://stub /dev/null --mock --out-dir /tmp/out
#
# Exit codes:
#   0  - all three tiers succeeded + aggregate written
#   2  - argument error
#   10 - Tier 1 failed
#   20 - Tier 2 failed
#   30 - Tier 3 failed
#   40 - Aggregate failed
#
# Honours $PYTHON env var for the interpreter (default: python3).

set -euo pipefail

PYTHON="${PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat >&2 <<EOF
usage: run-all.sh <model_id> <base_url> <curriculum_jsonl> [--out-dir DIR] [--mock]

  <model_id>          The served model identifier (e.g. lamarck-g1)
  <base_url>          OpenAI-compatible endpoint root (e.g. http://localhost:8000/v1)
  <curriculum_jsonl>  G_N's curriculum file for Tier 3 (a path; use /dev/null with --mock)

Flags:
  --out-dir DIR       Write per-tier and final JSON here (default: ./eval-out)
  --mock              Skip real runners; emit fixed mock TierResults
  -h, --help          Show this message
EOF
}

# ---- Argument parsing ------------------------------------------------------

if [[ $# -lt 1 ]] || [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
    usage
    exit 2
fi

if [[ $# -lt 3 ]]; then
    echo "run-all.sh: need <model_id> <base_url> <curriculum_jsonl>" >&2
    usage
    exit 2
fi

MODEL_ID="$1"
BASE_URL="$2"
CURRICULUM="$3"
shift 3

OUT_DIR="./eval-out"
MOCK_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out-dir)
            shift
            if [[ $# -eq 0 ]]; then
                echo "run-all.sh: --out-dir requires a value" >&2
                exit 2
            fi
            OUT_DIR="$1"
            ;;
        --mock)
            MOCK_FLAG="--mock"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "run-all.sh: unknown flag: $1" >&2
            usage
            exit 2
            ;;
    esac
    shift
done

mkdir -p "$OUT_DIR"

T1_OUT="$OUT_DIR/tier1.json"
T2_OUT="$OUT_DIR/tier2.json"
T3_OUT="$OUT_DIR/tier3.json"
FINAL_OUT="$OUT_DIR/final.json"

echo "[run-all] model_id=$MODEL_ID base_url=$BASE_URL"
echo "[run-all] curriculum=$CURRICULUM out_dir=$OUT_DIR mock=${MOCK_FLAG:-no}"

# ---- Tier 1 ---------------------------------------------------------------

echo "[run-all] Tier 1: external sanity benchmarks..."
if ! "$PYTHON" "$SCRIPT_DIR/run_tier.py" \
        --tier 1 --model-id "$MODEL_ID" --base-url "$BASE_URL" \
        --out "$T1_OUT" $MOCK_FLAG; then
    echo "[run-all] Tier 1 FAILED" >&2
    exit 10
fi
echo "[run-all]   -> $T1_OUT"

# ---- Tier 2 ---------------------------------------------------------------

echo "[run-all] Tier 2: ML-engineering tasks..."
if ! "$PYTHON" "$SCRIPT_DIR/run_tier.py" \
        --tier 2 --model-id "$MODEL_ID" --base-url "$BASE_URL" \
        --out "$T2_OUT" $MOCK_FLAG; then
    echo "[run-all] Tier 2 FAILED" >&2
    exit 20
fi
echo "[run-all]   -> $T2_OUT"

# ---- Tier 3 ---------------------------------------------------------------

echo "[run-all] Tier 3: grounded teaching..."
T3_ARGS=(--tier 3 --model-id "$MODEL_ID" --base-url "$BASE_URL"
         --curriculum-jsonl "$CURRICULUM" --out "$T3_OUT")
if [[ -n "$MOCK_FLAG" ]]; then
    T3_ARGS+=("$MOCK_FLAG")
fi
if ! "$PYTHON" "$SCRIPT_DIR/run_tier.py" "${T3_ARGS[@]}"; then
    echo "[run-all] Tier 3 FAILED" >&2
    exit 30
fi
echo "[run-all]   -> $T3_OUT"

# ---- Aggregate ------------------------------------------------------------

echo "[run-all] Aggregating..."
if ! "$PYTHON" -m lamarck.eval.aggregate_cli \
        --tier1 "$T1_OUT" --tier2 "$T2_OUT" --tier3 "$T3_OUT" \
        --out "$FINAL_OUT"; then
    echo "[run-all] Aggregate FAILED" >&2
    exit 40
fi

# ---- Summary --------------------------------------------------------------

# Extract final_score for the operator's eye - jq is optional; fall back to
# a python one-liner.
if command -v jq >/dev/null 2>&1; then
    FINAL_SCORE=$(jq -r '.final_score' "$FINAL_OUT")
else
    FINAL_SCORE=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['final_score'])" "$FINAL_OUT")
fi

echo "[run-all] DONE. Final score: $FINAL_SCORE  ($FINAL_OUT)"
exit 0
