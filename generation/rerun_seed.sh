#!/usr/bin/env bash
# C9 reproduction hook. Contract: hook <episode_id> <seed> [<run_start_seq>], exit 0 on success.
#
# Episode-alone reproduction does not hold (generation/README.md); a run reproduces, bit for
# bit, and so does its prefix. So this reruns the run that made the episode -- from its first
# seq, the episode's run_start_seq (schema §3), up to the episode -- and compares the last
# episode against the published parquet: its step columns, and the episode metadata the
# converter copied out of the sidecar. Without the third argument the run is taken to have
# begun at 0, as every run did before the field existed. An episode of a scenario with a
# conditions table is rerun with the walking policy and behaviour layer it recorded
# (sim_backend.policy in the published row -- that record is compared too); one the rerun cannot
# get back ends this with exit 2 before the simulator boots. A camera distribution (one shipping
# meta/cameras.json) is compared on its box and camera-pose columns too, and on
# a depth/mask pack when that file declares one (none since 2026-09-30), never
# on RGB -- the rerun's camera-canon fingerprint has to be that file's instead
# (compare_published.py); its rerun leaves ~1.4-1.5 GB of media per episode
# in $WORK (seven lossless videos, 2026-10-22) until the trap removes it -- or,
# when a SIGKILL skips the trap, until the validator removes the TMPDIR it
# made for this run (validate_dataset.run_hook).
#
# C9_PUBLISHED_ROOT points at the dataset being validated. It defaults to the
# local conversion output.
set -euo pipefail

EPISODE_ID="${1:?episode_id 가 필요합니다}"
SEED="${2:?seed 가 필요합니다}"
RUN_START="${3-0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PUBLISHED="${C9_PUBLISHED_ROOT:-$REPO_ROOT/.tmp/lerobot/S1a}"
# Say which one, always. The default is a convenience for running this by hand,
# and a silent fallback to it means comparing against a dataset that is not the
# one under test -- which can pass, and then C9 is green about the wrong data.
echo "대조 대상 배포본: $PUBLISHED" >&2
# Under TMPDIR: the validator hands over a directory of its own and removes it once the hook has ended.
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
# A signal ends the script through the EXIT trap above. The validator ends a hook that ran out of
# time with SIGTERM to its whole process group (validate_dataset.run_hook); bash dying of a signal
# it does not handle may skip the EXIT trap, and $WORK is ~1.4-1.5 GB per episode of a camera rerun.
trap 'exit 143' TERM
trap 'exit 130' INT

# All three interpreters are overridable so a test can substitute a recorder and
# see the argv this script actually builds. Without that, a test can only reach
# the seed parsing below -- which is where one stopped, leaving the comparison
# call unprotected: hardcoding --published back to the old default kept it green.
# Defaults are the real ones.
#
# The seq interpreter has to be one of the three: hardcoded, a machine without
# .venv -- CI installs into the system python -- dies at line one of the hook
# with exit 127, so the test that checks the argv cannot run there at all.
#
# The validator strips every C9_* variable except the published root before
# calling this, so the door is open only to a test that runs the hook directly.
# It has to be: left open through the validator, `C9_ISAAC_PYTHON=/bin/true`
# passed C9 on 400 episodes with neither the simulator nor the comparator
# running. The cost is that there is no end-to-end
# validator-through-hook test without Isaac; the wiring is covered instead by
# two tests -- one asserts the argv this script builds, the other asserts what
# the validator puts in the environment.
SEQ_PYTHON="${C9_SEQ_PYTHON:-$REPO_ROOT/.venv/bin/python}"
ISAAC_PYTHON="${C9_ISAAC_PYTHON:-$HOME/isaacsim/python.sh}"
COMPARE_PYTHON="${C9_COMPARE_PYTHON:-$REPO_ROOT/.venv-lerobot/bin/python}"

# The id reaches Python as an argument, never spliced into the source. It comes out of the
# distribution under test -- C9 hands the hook every id when none of them parses -- and
# spliced in as '$EPISODE_ID', a quote in it ran as code (2026-09-28).
SEQ="$(PYTHONPATH="$REPO_ROOT" "$SEQ_PYTHON" -c \
  'import sys; from generation.compare_repro import episode_seq_of; print(episode_seq_of(sys.argv[1]))' \
  "$EPISODE_ID")"
# The scenario is the id's own. run_s1a.py defaults to S1a, and before this an S1d
# episode was regenerated as S1a, so C9 could only say "a different episode".
SCENARIO="$(PYTHONPATH="$REPO_ROOT" "$SEQ_PYTHON" -c \
  'import sys; from generation.compare_repro import episode_scenario_of; print(episode_scenario_of(sys.argv[1]))' \
  "$EPISODE_ID")"

# Refused before the simulator boots: a start that is not a count, or one past the episode --
# a run that began after the episode cannot have made it.
case "$RUN_START" in
  ''|*[!0-9]*)
    echo "run_start_seq 가 0 이상의 정수가 아닙니다: '$RUN_START'" >&2
    exit 2 ;;
esac
RUN_START=$((10#$RUN_START))
if (( RUN_START > SEQ )); then
  echo "run_start_seq $RUN_START 가 에피소드 번호 $SEQ 보다 큽니다 — 그 실행은 이 에피소드를 만들 수 없습니다" >&2
  exit 2
fi

# The walking policy and behaviour layer the episode recorded, as run_s1a.py flags one per line: none for a
# scenario without a conditions table (the distribution is not read -- S1a and S1d rerun without them), the
# recorded ones for a table scenario, read off the published row (generation/rerun_policy.py). One the rerun
# cannot get back -- a checkpoint outside the repository or with other bytes, another published task, a
# behaviour module that is not a file of the repository or has bytecode or an extension module beside it, a
# checkpoint or behaviour file that git does not track by the path it is named by or whose bytes are not HEAD's
# (the repository's history is what vouches for a file, not the hash the distribution records), a repository
# that is not the top level of a git checkout, no such row -- ends the hook here with exit 2, before the
# simulator boots. The flags reach run_s1a.py as arguments, never spliced into a command line.
POLICY_LINES="$(PYTHONPATH="$REPO_ROOT" "$SEQ_PYTHON" "$REPO_ROOT/generation/rerun_policy.py" \
  --published "$PUBLISHED" --episode-id "$EPISODE_ID")" || exit 2
POLICY_FLAGS=()
if [[ -n "$POLICY_LINES" ]]; then
  mapfile -t POLICY_FLAGS <<< "$POLICY_LINES"
fi

"$ISAAC_PYTHON" "$REPO_ROOT/generation/envs/run_s1a.py" --scenario "$SCENARIO" \
  --seq "$RUN_START" --episodes "$((SEQ - RUN_START + 1))" --max-steps 1000 --out "$WORK" \
  ${POLICY_FLAGS[@]+"${POLICY_FLAGS[@]}"} >/dev/null

# stdout from here on is the comparison's alone: C9 reports its last line on success and
# its tail, before this script's stderr, on failure (validate_dataset.repro_hook_finding).
# Print nothing else to stdout.
PYTHONPATH="$REPO_ROOT" "$COMPARE_PYTHON" \
  "$REPO_ROOT/generation/compare_published.py" \
  --episode-id "$EPISODE_ID" --seed "$SEED" --run-start-seq "$RUN_START" \
  --regenerated "$WORK" --published "$PUBLISHED"
