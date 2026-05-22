"""
KELE Socratic Teaching System — working copy extended from the original.

This module wraps the KELE SocraticTeachingSystem with our config layer
and adds batch evaluation for running against the SocratDataset.
"""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.project.config import load_config
from src.project.socratic_teaching_system import SocraticTeachingSystem

RESOURCES_DIR = Path(__file__).resolve().parents[2] / "references" / "KELE"


def create_system(
    debug: bool | None = None,
    experiment: str | None = None,
    unified: bool = False,
    bert_consultant: str | None = None,
) -> SocraticTeachingSystem:
    """Create a SocraticTeachingSystem from environment config.

    If unified=True, instantiates SocraticTeachingSystemUnified — the
    single-call variant that fuses consultant + teacher into one
    structured-output LLM call. See docs/SOCRATIC_FUSION_PLAN.md.

    If bert_consultant is a path to a trained 34-state classifier checkpoint,
    use SocraticTeachingSystemBertConsultant: BERT predicts the state and
    the LLM only generates the teacher response (two-call style). Mutually
    exclusive with unified=True.
    """
    cfg = load_config(experiment=experiment)
    if bert_consultant and unified:
        raise ValueError("--unified and --bert-consultant are mutually exclusive")

    if bert_consultant:
        from src.project.socratic_teaching_bert_consultant import (
            SocraticTeachingSystemBertConsultant,
        )

        cls: type[SocraticTeachingSystem] = SocraticTeachingSystemBertConsultant
        extra_kwargs: dict = {"bert_ckpt": bert_consultant}
    elif unified:
        from src.project.socratic_teaching_unified import SocraticTeachingSystemUnified

        cls = SocraticTeachingSystemUnified
        extra_kwargs = {}
    else:
        cls = SocraticTeachingSystem
        extra_kwargs = {}
    return cls(
        consultant_api_key=cfg.consultant.api_key,
        consultant_base_url=cfg.consultant.base_url,
        consultant_model_name=cfg.consultant.model_name,
        teacher_api_key=cfg.teacher.api_key,
        teacher_base_url=cfg.teacher.base_url,
        teacher_model_name=cfg.teacher.model_name,
        debug_mode=debug if debug is not None else cfg.debug_mode,
        max_teaching_rounds=cfg.max_teaching_rounds,
        consultant_max_tokens=cfg.consultant.max_tokens,
        consultant_disable_thinking=cfg.consultant.disable_thinking,
        consultant_thinking_budget=cfg.consultant.thinking_budget,
        consultant_num_ctx=cfg.consultant.num_ctx,
        **extra_kwargs,
    )


def load_dataset(
    path: Path | None = None,
    split: str = "test",
    seed: int = 42,
    source: str = "hf",
    hf_repo: str | list[str] = [  # noqa: B006
        "ulises-c/SocratDataset-EN",
        "ulises-c/SocratDataset",
    ],
) -> list[dict]:
    """Load the SocratDataset with train/test split.

    The paper uses a 90/10 train/test split. We evaluate on the test set (~680 dialogues).
    Args:
        split: "test" (10%, for evaluation), "train" (90%), or "all" (full dataset).
        seed: Random seed for reproducible splits.
        source: "hf" to load from HuggingFace (default), "local" to read from a JSON file.
            Passing an explicit path implies source="local" regardless of this argument.
        hf_repo: One or more HuggingFace repo IDs loaded and concatenated when source="hf".
            Each repo must have dialogue records with {id, question, answer, dialogue} fields
            including state annotations (only SocratDataset / SocratDataset-EN qualify).
    """
    if source == "hf" and path is None:
        from datasets import load_dataset as hf_load

        repos = [hf_repo] if isinstance(hf_repo, str) else list(hf_repo)
        data: list[dict] = []
        for repo in repos:
            data.extend(dict(r) for r in hf_load(repo, split="train"))
    else:
        if path is None:
            path = RESOURCES_DIR / "SocratDataset.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

    if split == "all":
        return data

    import random

    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)

    split_point = int(len(data) * 0.9)
    if split == "train":
        return [data[i] for i in sorted(indices[:split_point])]
    else:  # test
        return [data[i] for i in sorted(indices[split_point:])]


def run_single_dialogue(system: SocraticTeachingSystem, item: dict) -> dict:
    """Replay a single dataset item through the system and collect outputs.

    Uses the student turns from the ground-truth dialogue as input,
    but the teacher and consultant responses come from our live system.
    """
    system.reset_session()
    ground_truth = item["dialogue"]
    generated_turns = []
    start = time.time()

    for turn in ground_truth:
        student_input = turn["student"]
        teacher_response = system.process_student_input(student_input)

        turn_record: dict = {
            "student": student_input,
            "state": system.current_state,
            "teacher_response": teacher_response,
            "ground_truth_teacher": turn["teacher"],
            "ground_truth_state": turn["state"],
        }
        thinking = getattr(system, "_last_thinking_content", None)
        if thinking:
            turn_record["thinking_content"] = thinking

        generated_turns.append(turn_record)

        # If we hit the summary stage, stop
        if system.current_state == "e34":
            break

    elapsed = time.time() - start

    return {
        "id": item["id"],
        "question": item["question"],
        "answer": item["answer"],
        "num_turns_ground_truth": len(ground_truth),
        "num_turns_generated": len(generated_turns),
        "dialogue": generated_turns,
        "elapsed_seconds": round(elapsed, 2),
    }


def _format_progress_line(
    pos: int,
    total: int,
    item_id: int,
    result: dict,
    elapsed: float,
    rate: float,
    remaining: float,
) -> str:
    """One-line progress record used by both sequential and parallel paths."""
    if "error" in result:
        status = f"ERROR: {result['error']}"
        turns: str | int = "?"
        secs: float = 0
    else:
        status = "✓"
        turns = result.get("num_turns_generated", "?")
        secs = result.get("elapsed_seconds", 0) or 0
    return (
        f"  {pos:>4}/{total}  id={item_id:04d}"
        f"  {turns:>5} turns  {secs:>4.0f}s"
        f"  {pos / total * 100:>4.1f}%"
        f"  {rate * 3600:>6.1f}  {remaining / 60:>4.0f}m  {status}"
    )


def _write_progress_log(
    progress_log: Path,
    completed: int,
    total: int,
    rate: float,
    remaining: float,
    elapsed: float,
) -> None:
    with open(progress_log, "w") as f:
        f.write(
            f"{completed}/{total} {completed / total * 100:.1f}%"
            f" {rate * 3600:.1f} dlg/hr ETA {remaining / 60:.0f}m elapsed {elapsed / 60:.0f}m\n"
        )


def _run_sequential(
    pending: list[dict],
    dialogues_dir: Path,
    progress_log: Path,
    system: SocraticTeachingSystem,
    total_dataset: int,
    completed_initial: int,
    start_time: float,
) -> int:
    completed = completed_initial
    for item in pending:
        item_id = item["id"]
        out_file = dialogues_dir / f"{item_id:04d}.json"
        try:
            result = run_single_dialogue(system, item)
        except Exception as e:
            result = {"id": item_id, "error": str(e)}
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        completed += 1
        elapsed = time.time() - start_time
        rate = completed / elapsed if elapsed > 0 else 0
        remaining = (total_dataset - completed) / rate if rate > 0 else 0
        print(
            _format_progress_line(
                completed, total_dataset, item_id, result, elapsed, rate, remaining
            ),
            flush=True,
        )
        _write_progress_log(progress_log, completed, total_dataset, rate, remaining, elapsed)
    return completed


def _run_parallel(
    pending: list[dict],
    dialogues_dir: Path,
    progress_log: Path,
    total_dataset: int,
    completed_initial: int,
    start_time: float,
    *,
    workers: int,
    experiment: str | None,
    unified: bool,
    bert_consultant: str | None,
) -> tuple[int, list[SocraticTeachingSystem]]:
    """Run pending dialogues concurrently using a ThreadPoolExecutor.

    Each worker thread owns its own SocraticTeachingSystem (session state
    on the system object makes per-worker isolation the clean choice).
    BERT-consultant variant: each worker loads its own ~100 MB classifier
    onto the GPU — small price for full thread isolation.

    Returns (final completed count, list of worker systems for end-of-run
    metadata aggregation).
    """
    thread_local = threading.local()
    worker_systems: list[SocraticTeachingSystem] = []
    worker_systems_lock = threading.Lock()
    progress_lock = threading.Lock()
    completed_box = [completed_initial]

    def get_or_create_system() -> SocraticTeachingSystem:
        sys_ = getattr(thread_local, "system", None)
        if sys_ is None:
            sys_ = create_system(
                debug=False,
                experiment=experiment,
                unified=unified,
                bert_consultant=bert_consultant,
            )
            thread_local.system = sys_
            with worker_systems_lock:
                worker_systems.append(sys_)
        return sys_

    def process_one(item: dict) -> tuple[dict, dict]:
        item_id = item["id"]
        out_file = dialogues_dir / f"{item_id:04d}.json"
        sys_ = get_or_create_system()
        try:
            result = run_single_dialogue(sys_, item)
        except Exception as e:
            result = {"id": item_id, "error": str(e)}
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return item, result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process_one, item) for item in pending]
        for fut in as_completed(futures):
            item, result = fut.result()
            with progress_lock:
                completed_box[0] += 1
                completed = completed_box[0]
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (total_dataset - completed) / rate if rate > 0 else 0
                print(
                    _format_progress_line(
                        completed, total_dataset, item["id"], result, elapsed, rate, remaining
                    ),
                    flush=True,
                )
                _write_progress_log(
                    progress_log, completed, total_dataset, rate, remaining, elapsed
                )

    return completed_box[0], worker_systems


def run_batch_evaluation(
    output_dir: Path,
    dataset_path: Path | None = None,
    start_id: int = 1,
    limit: int | None = None,
    experiment: str | None = None,
    split: str = "test",
    unified: bool = False,
    bert_consultant: str | None = None,
    workers: int | None = None,
) -> None:
    """Run the full evaluation pipeline on the dataset.

    Saves each dialogue result individually (crash-safe) and writes
    a progress log for monitoring.

    If unified=True, uses the single-call fusion architecture
    (see docs/SOCRATIC_FUSION_PLAN.md).

    If workers > 1 (or env var KELE_PARALLEL_WORKERS > 1), dialogues are
    processed concurrently against the llama.cpp server's parallel KV slots.
    Each worker thread owns its own SocraticTeachingSystem; per-dialogue
    file output remains crash-safe (each dialogue → own JSON file).
    """
    if workers is None:
        workers = int(os.environ.get("KELE_PARALLEL_WORKERS", "1"))
    if workers < 1:
        workers = 1

    dataset = load_dataset(dataset_path, split=split)
    total = len(dataset)

    # Filter to start_id and apply limit
    dataset = [d for d in dataset if d["id"] >= start_id]
    if limit is not None:
        dataset = dataset[:limit]

    # Always create one "probe" system for header info + run config metadata.
    # In sequential mode this is also the workhorse; in parallel mode each
    # worker thread creates its own via thread-local storage.
    system = create_system(
        debug=False, experiment=experiment, unified=unified, bert_consultant=bert_consultant
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    dialogues_dir = output_dir / "dialogues"
    dialogues_dir.mkdir(exist_ok=True)
    progress_log = output_dir / "progress.log"
    start_time = time.time()

    print(f"Starting evaluation: {len(dataset)} dialogues (of {total} in {split} split)")
    print(f"Output: {output_dir}")
    print(f"Teacher model: {system.teacher_model_name}")
    print(f"Consultant model: {system.consultant_model_name}")
    print(f"Parallel workers: {workers}")
    print("-" * 60)
    print(
        f"  {'#':>9}  {'id':<8}  {'turns':>5}  {'time':>5}  {'%':>5}  {'dlg/hr':>6}  {'ETA':>5}  status"
    )
    print("-" * 60)

    # Bucket already-done dialogues into the completed counter; only pending
    # items get dispatched. Same crash-recovery semantics as the prior loop.
    pending: list[dict] = []
    completed = 0
    for item in dataset:
        out_file = dialogues_dir / f"{item['id']:04d}.json"
        if out_file.exists():
            completed += 1
        else:
            pending.append(item)

    if completed and pending:
        print(f"  (resuming: {completed} already on disk, {len(pending)} to run)")

    if workers == 1:
        completed = _run_sequential(
            pending,
            dialogues_dir,
            progress_log,
            system,
            len(dataset),
            completed,
            start_time,
        )
    else:
        completed, worker_systems = _run_parallel(
            pending,
            dialogues_dir,
            progress_log,
            len(dataset),
            completed,
            start_time,
            workers=workers,
            experiment=experiment,
            unified=unified,
            bert_consultant=bert_consultant,
        )
        # Aggregate fallback counts across worker systems for unified mode.
        if unified:
            total_fb = 0
            for ws in [system, *worker_systems]:
                fb = getattr(ws, "_unified_fallback_count", 0)
                if fb:
                    total_fb += fb
            setattr(system, "_unified_fallback_count", total_fb)

    print(f"\nDone. {completed} dialogues saved to {dialogues_dir}")

    # Save run config
    cfg = load_config()
    run_config = {
        "experiment": output_dir.name,
        "teacher_model": cfg.teacher.model_name,
        "teacher_base_url": cfg.teacher.base_url,
        "consultant_model": cfg.consultant.model_name,
        "consultant_base_url": cfg.consultant.base_url,
        "thinking_budget": cfg.consultant.thinking_budget,
        "max_teaching_rounds": cfg.max_teaching_rounds,
        "unified": unified,
        "workers": workers,
        "total_dialogues": len(dataset),
        "completed": completed,
        "total_elapsed_seconds": round(time.time() - start_time, 2),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if unified and hasattr(system, "_unified_fallback_count"):
        run_config["unified_fallback_count"] = system._unified_fallback_count  # type: ignore[reportAttributeAccessIssue]
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # Auto-compute metrics after run completes
    print("\nComputing metrics...")
    from src.project.metrics import compute_all_metrics, format_metrics_table

    metrics = compute_all_metrics(dialogues_dir)
    with open(output_dir / "metrics_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(format_metrics_table(metrics))


def interactive() -> None:
    """Launch an interactive Socratic teaching session."""
    system = create_system()
    system.start_conversation()


def main() -> None:
    """CLI entry point for the KELE runner."""
    import argparse

    parser = argparse.ArgumentParser(description="KELE Socratic Teaching System")
    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        default=None,
        help="Experiment config name (loads configs/<name>.env). E.g.: baseline, gemma4",
    )
    sub = parser.add_subparsers(dest="command")

    # Interactive mode
    sub.add_parser("interactive", help="Start an interactive teaching session")

    # Batch evaluation mode
    eval_parser = sub.add_parser("evaluate", help="Run batch evaluation on the dataset")
    eval_parser.add_argument("--output", type=Path, default=None)
    eval_parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "train", "all"],
        help="Dataset split: test (10%%, default), train (90%%), all",
    )
    eval_parser.add_argument("--start-id", type=int, default=1, help="Resume from this dialogue ID")
    eval_parser.add_argument("--limit", type=int, default=None, help="Max dialogues to process")
    eval_parser.add_argument(
        "--unified",
        action="store_true",
        help="Use single-call fusion architecture (consultant + teacher in one LLM call). "
        "See docs/SOCRATIC_FUSION_PLAN.md.",
    )
    eval_parser.add_argument(
        "--bert-consultant",
        type=str,
        default=None,
        help="Path to a trained 34-state BERT classifier checkpoint dir. "
        "Replaces the LLM consultant with the BERT classifier; LLM only "
        "generates the teacher response (two-call style).",
    )
    eval_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of concurrent dialogue workers. Each worker hits one "
        "of the llama-server's --parallel KV slots. Default 1 (or env "
        "KELE_PARALLEL_WORKERS). Recommended: 4. Server-side -np must be "
        "at least this value.",
    )

    # Quick test mode — run on a handful of dialogues
    test_parser = sub.add_parser("test", help="Quick test with a few dialogues")
    test_parser.add_argument("--n", type=int, default=3, help="Number of dialogues to test")
    test_parser.add_argument("--output", type=Path, default=Path("results/test"))
    test_parser.add_argument(
        "--unified",
        action="store_true",
        help="Use single-call fusion architecture (consultant + teacher in one LLM call). "
        "See docs/SOCRATIC_FUSION_PLAN.md.",
    )
    test_parser.add_argument(
        "--bert-consultant",
        type=str,
        default=None,
        help="Path to a trained 34-state BERT classifier checkpoint dir.",
    )

    args = parser.parse_args()

    if args.command == "interactive":
        interactive()
    elif args.command == "evaluate":
        output = args.output or Path(f"results/{args.experiment or 'baseline'}")
        run_batch_evaluation(
            output,
            start_id=args.start_id,
            limit=args.limit,
            experiment=args.experiment,
            split=args.split,
            unified=args.unified,
            bert_consultant=args.bert_consultant,
            workers=args.workers,
        )
    elif args.command == "test":
        run_batch_evaluation(
            args.output,
            limit=args.n,
            experiment=args.experiment,
            unified=args.unified,
            bert_consultant=args.bert_consultant,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
