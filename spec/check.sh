#!/bin/bash
# check.sh — Run TLA+ model checking for RevisionPin (if tlc is available).
# If tlc is not installed, prints a skip message (not a failure).

TLC_JAR="${TLC_JAR:-/usr/local/lib/tla2tools.jar}"

if ! command -v java &>/dev/null; then
    echo "SKIP: java not found (required to run TLC)"
    exit 0
fi

if [ ! -f "$TLC_JAR" ]; then
    echo "SKIP: TLC jar not found at $TLC_JAR"
    echo "Install with: brew install tla-tools  OR set TLC_JAR environment variable"
    exit 0
fi

cd "$(dirname "$0")"

echo "=== Checking RevisionPin (correct spec) ==="
java -jar "$TLC_JAR" -config RevisionPin.cfg RevisionPin.tla 2>&1
CORRECT_EXIT=$?

echo ""
echo "=== Checking RevisionPinWeakened (should violate invariant) ==="
java -jar "$TLC_JAR" -config RevisionPinWeakened.cfg RevisionPinWeakened.tla 2>&1
WEAKENED_EXIT=$?

# For the correct spec: expect exit 0 (no violations)
# For the weakened spec: expect non-zero (violation found)
if [ $CORRECT_EXIT -eq 0 ] && [ $WEAKENED_EXIT -ne 0 ]; then
    echo ""
    echo "PASS: Correct spec holds, weakened spec shows violation (as expected)"
    exit 0
else
    echo ""
    echo "UNEXPECTED: correct_exit=$CORRECT_EXIT, weakened_exit=$WEAKENED_EXIT"
    exit 1
fi
