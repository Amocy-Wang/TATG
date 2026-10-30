#!/usr/bin/env bash
# End-to-end TATG smoke test against a stub LLM (no API key needed).
#
# Usage:
#   examples/stub-smoke/run_smoke.sh PROJECT_ROOT REL_SOURCE CLASS METHOD CANNED_TEST [extra tatg args...]
#
# Example (commons-csv 1.10.0, existing tests removed):
#   examples/stub-smoke/run_smoke.sh ~/subjects/commons-csv \
#       src/main/java/org/apache/commons/csv/Lexer.java Lexer nextToken \
#       examples/stub-smoke/canned/LexerTest.java --max-iterations 2
#
# Environment: PORT (default 18766), OUT_DIR (default ./artifacts/stub-smoke/<Class>.<method>),
#              PYTHON (default python3).
set -euo pipefail

if [ "$#" -lt 5 ]; then
  sed -n '2,13p' "$0"; exit 2
fi
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PROJECT="$(cd "$1" && pwd)"; SRC="$2"; CLS="$3"; METHOD="$4"; CANNED="$(cd "$(dirname "$5")" && pwd)/$(basename "$5")"
shift 5
PORT="${PORT:-18766}"
PYTHON="${PYTHON:-python3}"
OUT_DIR="${OUT_DIR:-$PWD/artifacts/stub-smoke/$CLS.$METHOD}"
mkdir -p "$OUT_DIR/llm"

"$PYTHON" "$HERE/stub_llm.py" "$PORT" "$OUT_DIR/llm" "$CANNED" > "$OUT_DIR/stub.log" 2>&1 &
STUB_PID=$!
trap 'kill $STUB_PID 2>/dev/null || true' EXIT
sleep 1

# Same Maven + injected JaCoCo command the benchmark batch runner uses.
TEST_COMMAND="$(cd "$REPO/scripts" && "$PYTHON" -c "
import sys
from pathlib import Path
import run_all_hard_methods_batch as r
print(r._make_test_command(sys.argv[1], project_root=Path(sys.argv[2])))
" "$CLS" "$PROJECT" | tail -1)"

TATG_STUB_KEY=stub "$PYTHON" -m tatg.cli generate \
  --scope method \
  --project-root "$PROJECT" \
  --target-file "$PROJECT/$SRC" \
  --class-name "$CLS" \
  --method-name "$METHOD" \
  --api-base "http://127.0.0.1:$PORT/v1" \
  --model stub-model \
  --api-key-env TATG_STUB_KEY \
  --llm-timeout-seconds 60 \
  --test-command "$TEST_COMMAND" \
  --json-out "$OUT_DIR/report.json" \
  "$@"

echo "report: $OUT_DIR/report.json"
echo "llm request log: $OUT_DIR/llm/"
