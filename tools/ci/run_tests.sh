#!/usr/bin/env bash
# Run the CPU unit suite against two checkouts and fail only on regressions —
# tests that pass in BASE_DIR (main) but fail in HEAD_DIR (the PR). This keeps
# the public NeMo base image's pre-existing CPU-only failures from blocking PRs.
#
# Usage:  IMAGE=<image> tools/ci/run_tests.sh <base_dir> <head_dir>
# Local:  IMAGE=nvcr.io/nvidia/nemo:26.06 tools/ci/run_tests.sh ../main .
set -uo pipefail

IMAGE="${IMAGE:?set IMAGE to the test image}"
BASE_DIR="${1:?usage: run_tests.sh <base_dir> <head_dir>}"
HEAD_DIR="${2:?usage: run_tests.sh <base_dir> <head_dir>}"

# Run the suite inside the image against a checkout; print its failing node IDs.
run_suite() {  # $1 = source dir, $2 = raw-log path
  docker run --rm -v "$1:/branch:ro" -w /branch \
    -e PYTHONPATH=/branch/src -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES="" \
    "${IMAGE}" \
    pytest tests/unit_tests --ignore=tests/unit_tests/diffusion -m "not pleasefixme" \
      -p no:cacheprovider -q -rfE >"$2" 2>&1 || true
  grep -E '^(FAILED|ERROR) ' "$2" | awk '{print $2}' | sort -u
}

run_suite "${BASE_DIR}" base.log >base_failures.txt
run_suite "${HEAD_DIR}" head.log >head_failures.txt
# Regressions = failing on head but not on main (comm -13 keeps lines only in the
# 2nd file, dropping main-only failures and failures shared by both).
comm -13 base_failures.txt head_failures.txt >regressions.txt

echo "main failures: $(wc -l <base_failures.txt) | head failures: $(wc -l <head_failures.txt)"
if [ -s regressions.txt ]; then
  echo "::error::PR introduces $(wc -l <regressions.txt) test regression(s) vs main:"
  cat regressions.txt
  exit 1
fi
echo "No regressions vs main (ignored $(wc -l <base_failures.txt) pre-existing base failures)."
