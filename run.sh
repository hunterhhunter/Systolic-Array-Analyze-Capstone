#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  cat <<'EOF'
Usage:
  ./run.sh make env
  ./run.sh make test
  ./run.sh make boundary-sweep
  ./run.sh python -m tools.boundary_sweep --dry-run --mnk 32x32x64 --array 8x8
EOF
  exit 1
fi

# The docker-compose dev service stores its virtualenv in the named /opt/venv volume.
# Give a clear first-run message instead of failing later with a missing python path.
if [ "$#" -ge 2 ] && [ "$1" = "make" ] && [ "$2" != "env" ]; then
  docker compose run --rm dev bash -lc "test -x /opt/venv/bin/python || { echo 'Python environment is missing. Run: ./run.sh make env'; exit 2; }"
fi

docker compose run --rm dev "$@"
