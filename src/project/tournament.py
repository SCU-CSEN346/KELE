"""
Tournament system for comparing multiple LLMs on the KELE Socratic Teaching benchmark.

Sequential evaluation (one model per 32 GB VRAM slot at a time):
  round N: boot model → run n=50 eval → kill server → record state_accuracy
  eliminate: drop worst performer(s) until 3 finalists remain
  finalize:  run survivors to n=681

Crash-safe: state persists to results/tournament/state.json after each model.

CLI:
  uv run tournament run [--n N] [--unified]
  uv run tournament status
  uv run tournament eliminate [N]
  uv run tournament finalize [--unified]
  uv run tournament reset
  uv run tournament download
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ── Model registry ────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_FILE = REPO_ROOT / "results" / "tournament" / "state.json"
PORT = int(os.environ.get("PORT", "8080"))
LLAMA_URL = f"http://localhost:{PORT}"
LLAMA_SERVER = Path(
    os.environ.get(
        "LLAMA_SERVER",
        str(Path.home() / "Documents" / "models" / "llama.cpp" / "build" / "bin" / "llama-server"),
    )
)


@dataclass
class ModelSpec:
    id: str
    name: str
    alias: str
    weight_path: str  # ~ expanded at runtime
    serve_script: str  # relative to repo root
    config_name: str  # configs/<name>.env
    hf_repo: str
    hf_file: str
    on_disk: bool = True


MODEL_REGISTRY: list[ModelSpec] = [
    # ── On disk ──────────────────────────────────────────────────────────────
    ModelSpec(
        id="qwen35-9b",
        name="Qwen3.5-9B Q4",
        alias="Qwen 9B Q4",
        weight_path="~/models/Qwen3.5-9B/Qwen3.5-9B-UD-Q4_K_XL.gguf",
        serve_script="scripts/serve_tournament_qwen35_9b.sh",
        config_name="tournament-qwen35-9b",
        hf_repo="unsloth/Qwen3.5-9B-GGUF",
        hf_file="Qwen3.5-9B-UD-Q4_K_XL.gguf",
        on_disk=True,
    ),
    ModelSpec(
        id="qwen27b",
        name="Qwen3.6-27B Q5",
        alias="Qwen 27B Q5",
        weight_path="~/Documents/models/weights/Qwen3.6-27B-UD-Q5_K_XL.gguf",
        serve_script="scripts/serve_qwen27b_q5.sh",
        config_name="qwen27b-local",
        hf_repo="unsloth/Qwen3.6-27B-GGUF",
        hf_file="Qwen3.6-27B-UD-Q5_K_XL.gguf",
        on_disk=True,
    ),
    ModelSpec(
        id="qwen35b-a3b",
        name="Qwen3.6-35B-A3B Q4 MoE",
        alias="Qwen 35B A3B",
        weight_path="~/Documents/models/weights/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        serve_script="scripts/serve_qwen35b_a3b.sh",
        config_name="qwen35b-a3b-local",
        hf_repo="unsloth/Qwen3.6-35B-A3B-GGUF",
        hf_file="Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        on_disk=True,
    ),
    ModelSpec(
        id="gemma4-31b",
        name="Gemma 4 31B Q5",
        alias="Gemma 4 31B",
        weight_path="~/Documents/models/weights/gemma-4-31B-it-UD-Q5_K_XL.gguf",
        serve_script="scripts/serve_gemma4_31b_q5.sh",
        config_name="gemma4-31b-local",
        hf_repo="unsloth/gemma-4-31b-it-GGUF",
        hf_file="gemma-4-31B-it-UD-Q5_K_XL.gguf",
        on_disk=True,
    ),
    # ── Downloadable (add more as weights arrive) ─────────────────────────────
    ModelSpec(
        id="llama4-scout",
        name="Llama-4-Scout-17B-16E Q4 MoE",
        alias="Llama 4 Scout",
        weight_path="~/Documents/models/weights/Llama-4-Scout-17B-16E-Instruct-UD-Q4_K_XL.gguf",
        serve_script="scripts/serve_llama4_scout.sh",
        config_name="tournament-llama4-scout",
        hf_repo="unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF",
        hf_file="Llama-4-Scout-17B-16E-Instruct-UD-Q4_K_XL.gguf",
        on_disk=False,
    ),
    ModelSpec(
        id="deepseek-r1-14b",
        name="DeepSeek-R1-Distill-Qwen-14B Q5",
        alias="DeepSeek R1 14B",
        weight_path="~/Documents/models/weights/DeepSeek-R1-Distill-Qwen-14B-Q5_K_M.gguf",
        serve_script="scripts/serve_deepseek_r1_14b.sh",
        config_name="tournament-deepseek-r1-14b",
        hf_repo="unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF",
        hf_file="DeepSeek-R1-Distill-Qwen-14B-Q5_K_M.gguf",
        on_disk=False,
    ),
    ModelSpec(
        id="phi4-14b",
        name="Phi-4 14B Q5",
        alias="Phi 4 14B",
        weight_path="~/Documents/models/weights/phi-4-Q5_K_M.gguf",
        serve_script="scripts/serve_phi4_14b.sh",
        config_name="tournament-phi4-14b",
        hf_repo="unsloth/phi-4-GGUF",
        hf_file="phi-4-Q5_K_M.gguf",
        on_disk=False,
    ),
    ModelSpec(
        id="gemma3-27b",
        name="Gemma 3 27B Q4",
        alias="Gemma 3 27B",
        weight_path="~/Documents/models/weights/gemma-3-27b-it-Q4_K_M.gguf",
        serve_script="scripts/serve_gemma3_27b.sh",
        config_name="tournament-gemma3-27b",
        hf_repo="unsloth/gemma-3-27b-it-GGUF",
        hf_file="gemma-3-27b-it-Q4_K_M.gguf",
        on_disk=False,
    ),
    ModelSpec(
        id="mistral-24b",
        name="Mistral-Small-3.2-24B Q4",
        alias="Mistral Small 24B",
        weight_path="~/Documents/models/weights/Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf",
        serve_script="scripts/serve_mistral_24b.sh",
        config_name="tournament-mistral-24b",
        hf_repo="unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
        hf_file="Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf",
        on_disk=False,
    ),
    ModelSpec(
        id="qwen35-14b",
        name="Qwen3.5-14B Q5",
        alias="Qwen 14B Q5",
        weight_path="~/Documents/models/weights/Qwen3.5-14B-Q5_K_M.gguf",
        serve_script="scripts/serve_qwen35_14b.sh",
        config_name="tournament-qwen35-14b",
        hf_repo="unsloth/Qwen3.5-14B-GGUF",
        hf_file="Qwen3.5-14B-Q5_K_M.gguf",
        on_disk=False,
    ),
]

MODEL_BY_ID: dict[str, ModelSpec] = {m.id: m for m in MODEL_REGISTRY}


# ── Tournament state ──────────────────────────────────────────────────────────


@dataclass
class TournamentState:
    round: int = 0
    n_per_round: int = 50
    models_active: list[str] = field(default_factory=list)
    models_eliminated: list[str] = field(default_factory=list)
    # {model_id: {round_N: score}}
    scores: dict[str, dict[str, float]] = field(default_factory=dict)
    # models that completed the current round (crash recovery)
    round_complete: list[str] = field(default_factory=list)
    final_scores: dict[str, float] = field(default_factory=dict)


def load_state() -> TournamentState:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return TournamentState(**json.load(f))
    # Default: 4 on-disk models, n=50
    return TournamentState(
        round=0,
        n_per_round=50,
        models_active=[m.id for m in MODEL_REGISTRY if m.on_disk],
        models_eliminated=[],
        scores={m.id: {} for m in MODEL_REGISTRY if m.on_disk},
    )


def save_state(state: TournamentState) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(asdict(state), f, indent=2)


# ── Server lifecycle ──────────────────────────────────────────────────────────


def _probe_server() -> str | None:  # pragma: no cover
    try:
        import urllib.request

        with urllib.request.urlopen(f"{LLAMA_URL}/v1/models", timeout=3) as r:
            return r.read().decode()
    except Exception:
        return None


def _server_has_alias(response: str, alias: str) -> bool:
    return f'"{alias}"' in response


def _server_ready(alias: str) -> bool:  # pragma: no cover
    resp = _probe_server()
    if not resp:
        return False
    if '"Loading model"' in resp:
        return False
    return _server_has_alias(resp, alias)


def _boot_server(spec: ModelSpec) -> subprocess.Popen[bytes]:  # pragma: no cover
    script = REPO_ROOT / spec.serve_script
    if not script.exists():
        print(f"  ERROR: serve script not found: {script}", file=sys.stderr)
        sys.exit(1)
    weight = Path(spec.weight_path).expanduser()
    if not weight.exists():
        print(f"  ERROR: weight file not found: {weight}", file=sys.stderr)
        sys.exit(1)
    proc = subprocess.Popen(
        ["bash", str(script)],
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def _kill_server(proc: subprocess.Popen[bytes] | None) -> None:  # pragma: no cover
    if proc is None:
        return
    try:
        proc.send_signal(signal.SIGTERM)
        for _ in range(15):
            if proc.poll() is not None:
                return
            time.sleep(1)
        proc.kill()
    except OSError:
        pass


def ensure_server(
    spec: ModelSpec,
) -> tuple[subprocess.Popen[bytes] | None, bool]:  # pragma: no cover
    """Return (proc, we_booted). Boots the server if not already running."""
    resp = _probe_server()
    if resp is not None:
        if _server_has_alias(resp, spec.alias):
            if '"Loading model"' not in resp:
                print(f"  Reusing existing server (alias '{spec.alias}' ready).")
                return None, False
            # Already loading our model — just wait
            print(f"  Existing server loading '{spec.alias}' — waiting...")
            proc = None
        else:
            print(
                f"  ERROR: server is up but serves a different model (want '{spec.alias}').\n"
                f"  Stop the existing server first.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        print(f"  Booting server for {spec.name} (may take 30-120 s)...")
        proc = _boot_server(spec)

    print("  Waiting for server", end="", flush=True)
    for i in range(180):
        if _server_ready(spec.alias):
            print(f" ready (~{(i + 1) * 2}s)")
            return proc, True
        if proc is not None and proc.poll() is not None:
            print(" FAILED — server exited early", file=sys.stderr)
            sys.exit(1)
        print(".", end="", flush=True)
        time.sleep(2)
    print(" TIMEOUT after 360 s", file=sys.stderr)
    _kill_server(proc)
    sys.exit(1)


# ── Eval runner ───────────────────────────────────────────────────────────────


def run_kele(
    spec: ModelSpec, out_dir: Path, n: int, unified: bool
) -> float | None:  # pragma: no cover
    """Run kele test --n N and return state_accuracy, or None on failure."""
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "src.project.kele",
        "--experiment",
        spec.config_name,
        "test",
        "--n",
        str(n),
        "--output",
        str(out_dir),
    ]
    if unified:
        cmd.append("--unified")

    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"  kele exited with code {result.returncode}", file=sys.stderr)
        return None

    metrics_file = out_dir / "metrics_summary.json"
    if not metrics_file.exists():
        print(f"  metrics_summary.json not found at {metrics_file}", file=sys.stderr)
        return None

    with open(metrics_file) as f:
        metrics = json.load(f)
    score = metrics.get("state_accuracy", {}).get("overall")
    if score is None:
        print("  state_accuracy.overall missing from metrics", file=sys.stderr)
    return score


# ── Leaderboard ───────────────────────────────────────────────────────────────


def print_leaderboard(state: TournamentState) -> None:
    round_key = f"round{state.round}"

    rows: list[tuple[str, str, str, str]] = []
    for mid in state.models_active:
        spec = MODEL_BY_ID.get(mid)
        name = spec.name if spec else mid
        latest = state.scores.get(mid, {}).get(round_key)
        score_str = f"{latest:.3f}" if latest is not None else "—"
        history = state.scores.get(mid, {})
        hist_str = "  ".join(f"r{k[5:]}: {v:.3f}" for k, v in sorted(history.items()))
        rows.append((name, score_str, "active", hist_str))

    for mid in state.models_eliminated:
        spec = MODEL_BY_ID.get(mid)
        name = spec.name if spec else mid
        history = state.scores.get(mid, {})
        hist_str = "  ".join(f"r{k[5:]}: {v:.3f}" for k, v in sorted(history.items()))
        rows.append((name, "—", "ELIMINATED", hist_str))

    # Sort active by latest score descending, eliminated last
    active_rows = [(n, s, st, h) for n, s, st, h in rows if st == "active"]
    elim_rows = [(n, s, st, h) for n, s, st, h in rows if st == "ELIMINATED"]
    active_rows.sort(key=lambda r: -float(r[1]) if r[1] != "—" else -99)
    rows = active_rows + elim_rows

    print()
    print(f"{'Model':<36}  {'Score':>7}  {'Status':<12}  History")
    print("─" * 90)
    for name, score, status, hist in rows:
        marker = "  ✓" if status == "active" else "  ✗"
        print(f"  {name:<34}  {score:>7}  {status:<12}  {hist}{marker}")
    print()
    print(
        f"Round: {state.round}  |  n/round: {state.n_per_round}  |  Active: {len(state.models_active)}  |  Eliminated: {len(state.models_eliminated)}"
    )
    print()


# ── Subcommands ───────────────────────────────────────────────────────────────


def cmd_status(_args: argparse.Namespace) -> None:
    state = load_state()
    if state.round == 0 and not state.scores:
        print("No tournament data yet. Run:  uv run tournament run")
        return
    print_leaderboard(state)


def cmd_run(args: argparse.Namespace) -> None:  # pragma: no cover
    state = load_state()
    state.round += 1
    round_key = f"round{state.round}"
    n = args.n if args.n is not None else state.n_per_round
    state.n_per_round = n
    unified = args.unified

    print(f"=== Tournament Round {state.round} ===")
    print(f"n={n}  unified={unified}  active={len(state.models_active)} models")
    print()

    # Warn about time cost
    secs_per_dialogue = 15  # conservative estimate for 32 GB GPU
    est_minutes = (n * secs_per_dialogue * len(state.models_active)) // 60
    print(f"Estimated time: ~{est_minutes} min ({est_minutes // 60}h {est_minutes % 60}m)")
    print()

    for mid in list(state.models_active):
        if mid in state.round_complete:
            score = state.scores.get(mid, {}).get(round_key, "?")
            print(f"[{mid}] already done this round (score={score}) — skipping.")
            continue

        spec = MODEL_BY_ID.get(mid)
        if spec is None:
            print(f"[{mid}] unknown model ID — skipping.", file=sys.stderr)
            continue

        weight = Path(spec.weight_path).expanduser()
        if not weight.exists():
            print(f"[{mid}] weight file not found: {weight} — skipping.")
            continue

        print(f"\n{'─' * 60}")
        print(f"  Model: {spec.name}")
        print(f"  Alias: {spec.alias}")

        proc, _we_booted = ensure_server(spec)

        out_dir = REPO_ROOT / "results" / "tournament" / f"round{state.round}" / mid
        print(f"  Output: results/tournament/round{state.round}/{mid}")

        score = run_kele(spec, out_dir, n, unified)

        _kill_server(proc)

        if score is not None:
            state.scores.setdefault(mid, {})[round_key] = score
            print(f"  Score: {score:.3f}")
        else:
            print("  Score: ERROR (see output above)")

        if mid not in state.round_complete:
            state.round_complete.append(mid)
        save_state(state)

    state.round_complete = []
    save_state(state)

    print(f"\n{'═' * 60}")
    print_leaderboard(state)


def cmd_eliminate(args: argparse.Namespace) -> None:
    state = load_state()
    n = args.n
    round_key = f"round{state.round}"

    if len(state.models_active) <= 3:
        print(f"Only {len(state.models_active)} model(s) remain — none eliminated.")
        print("Run:  uv run tournament finalize")
        return

    scored = [(mid, state.scores.get(mid, {}).get(round_key)) for mid in state.models_active]
    unscored = [mid for mid, s in scored if s is None]
    if unscored:
        print(f"WARNING: these models have no score for {round_key}: {unscored}")
        print("Run the round first:  uv run tournament run")
        return

    ranked = sorted(scored, key=lambda t: t[1] or 0.0)  # ascending (worst first)
    to_drop = min(n, len(ranked) - 3)  # keep at least 3
    if to_drop <= 0:
        print("No models to eliminate (floor: 3 finalists).")
        return

    for mid, score in ranked[:to_drop]:
        print(f"  Eliminated: {MODEL_BY_ID[mid].name} (score={score:.3f})")
        state.models_active.remove(mid)
        state.models_eliminated.append(mid)

    save_state(state)
    print(f"\n{len(state.models_active)} model(s) remain.")
    print_leaderboard(state)


def cmd_finalize(args: argparse.Namespace) -> None:  # pragma: no cover
    state = load_state()
    if len(state.models_active) > 3:
        print(
            f"WARNING: {len(state.models_active)} models still active. Eliminate down to 3 first.\n"
            f"  uv run tournament eliminate {len(state.models_active) - 3}"
        )

    unified = args.unified
    print(f"=== Tournament Finalize ===  (n=681, unified={unified})")
    print(f"Finalists: {state.models_active}")
    print()

    for mid in state.models_active:
        spec = MODEL_BY_ID.get(mid)
        if spec is None:
            continue
        weight = Path(spec.weight_path).expanduser()
        if not weight.exists():
            print(f"[{mid}] weight file not found: {weight} — skipping.")
            continue

        print(f"\n{'─' * 60}")
        print(f"  Model: {spec.name}")

        proc, _ = ensure_server(spec)
        out_dir = REPO_ROOT / "results" / "tournament" / "final" / mid

        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "src.project.kele",
            "--experiment",
            spec.config_name,
            "evaluate",
            "--output",
            str(out_dir),
        ]
        if unified:
            cmd.append("--unified")

        subprocess.run(cmd, cwd=str(REPO_ROOT))
        _kill_server(proc)

        metrics_file = out_dir / "metrics_summary.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            score = metrics.get("state_accuracy", {}).get("overall")
            if score is not None:
                state.final_scores[mid] = score
                print(f"  Final score: {score:.3f}")
        save_state(state)

    print(f"\n{'═' * 60}")
    print("Final results:")
    for mid, score in sorted(state.final_scores.items(), key=lambda t: -t[1]):
        spec = MODEL_BY_ID.get(mid)
        name = spec.name if spec else mid
        print(f"  {name:<36}  {score:.3f}")


def cmd_reset(args: argparse.Namespace) -> None:
    if not args.confirm:
        print("This will wipe results/tournament/state.json.")
        print("Re-run with --confirm to proceed.")
        return
    if STATE_FILE.exists():
        STATE_FILE.unlink()
        print(f"Removed {STATE_FILE}")
    print("State reset. Run:  uv run tournament run")


def cmd_download(_args: argparse.Namespace) -> None:
    pending = [m for m in MODEL_REGISTRY if not m.on_disk]
    if not pending:
        print("All models are on disk.")
        return

    weights_dir = Path.home() / "Documents" / "models" / "weights"
    print(f"Download commands (into {weights_dir}):\n")
    for m in pending:
        print(f"# {m.name}")
        print(f"huggingface-cli download {m.hf_repo} {m.hf_file} --local-dir {weights_dir}")
        print()
    print("After downloading, set on_disk=True in the registry and create serve/config files.")
    print("See existing scripts/serve_qwen27b_q5.sh and configs/qwen27b-local.env as templates.")


# ── CLI entry point ───────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KELE tournament: compare LLMs with elimination rounds."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p = sub.add_parser("run", help="Run one elimination round for all active models")
    p.add_argument("--n", type=int, default=None, help="Dialogues per model (default: 50)")
    p.add_argument("--unified", action="store_true", help="Use fusion architecture")

    # status
    sub.add_parser("status", help="Print current leaderboard")

    # eliminate
    p = sub.add_parser("eliminate", help="Drop the N worst-scoring models")
    p.add_argument("n", type=int, nargs="?", default=1, help="How many to eliminate (default: 1)")

    # finalize
    p = sub.add_parser("finalize", help="Run survivors to n=681 (full evaluation)")
    p.add_argument("--unified", action="store_true")

    # reset
    p = sub.add_parser("reset", help="Wipe tournament state")
    p.add_argument("--confirm", action="store_true")

    # download
    sub.add_parser("download", help="Print huggingface-cli download commands for missing models")

    args = parser.parse_args()

    dispatch = {
        "run": cmd_run,
        "status": cmd_status,
        "eliminate": cmd_eliminate,
        "finalize": cmd_finalize,
        "reset": cmd_reset,
        "download": cmd_download,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
