#!/usr/bin/env bash
# Run the full pipeline end to end, for one line or all lines.
#
# Usage:
#   ./run_all.sh            # runs all six lines
#   ./run_all.sh wkcomp     # runs just workers' compensation
#
# Safe to re-run any time: each stage overwrites its own output files.

set -e  # stop immediately if any step fails, instead of continuing on errors

TARGET="${1:-all}"

echo "=================================================="
echo " Reserve Adequacy Simulation Pipeline"
echo " Target: $TARGET"
echo "=================================================="

echo ""
echo "--- Step 1: Cleaning data ---"
python3 src/clean_data.py "$TARGET"

echo ""
echo "--- Step 2: Running simulation pipeline ---"
python3 src/run_pipeline.py "$TARGET"

echo ""
echo "--- Step 3: Generating figures ---"
python3 src/make_figures.py "$TARGET"

echo ""
echo "=================================================="
echo " Done. Check results/ and results/figures/"
echo "=================================================="
