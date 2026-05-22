"""
Unified training data loader for SocratTeachLLM v2.

Normalises all datasets to HF `messages` format, ready for
TRL SFTTrainer + apply_chat_template.

Each record:
    {
        "id":                  str,
        "source":              "socrat-zh" | "socrat-en" | "socrateach-multi" | "socrateach-single",
        "messages":            [{"role": "system"|"user"|"assistant", "content": str}, ...],
        "ground_truth_states": list[str] | None,  # only for socrat-zh / socrat-en
    }

Usage:
    from src.project.dataset import load_training_data
    records = load_training_data(["socrat-zh", "socrat-en", "socrateach-multi", "socrateach-single"])
"""

import random

_SOCRAT_ZH_SYSTEM = (
    "你是一位苏格拉底式教师。通过启发性问题引导学生自己发现答案——永远不要直接告诉学生答案。"
)

_SOCRAT_EN_SYSTEM = (
    "You are a Socratic teacher. Guide the student to discover the answer through "
    "heuristic questions — never give away the answer directly."
)

_SOCRATEACH_SYSTEM = (
    "You are a Socratic teacher. Guide the student through this problem using "
    "guiding questions — never give away the answer directly."
)


def _strip_quotes(s: str) -> str:
    """Strip surrounding single or double quotes present in the HF SocratDataset upload."""
    s = s.strip()
    if len(s) >= 2 and s[0] in ("'", '"') and s[-1] == s[0]:
        return s[1:-1]
    return s


# ---------------------------------------------------------------------------
# SocratDataset (original Chinese) — socrat-zh
# ---------------------------------------------------------------------------


def _socrat_zh_to_messages(record: dict) -> dict:
    """Convert one original Chinese SocratDataset record to messages format.

    Same schema as SocratDataset-EN but in Chinese. The local JSON serialises
    state/action/evaluation with extra surrounding quotes (e.g. "'a1'") — these
    are stripped before use. The HF version (ulises-c/SocratDataset) is clean.
    """
    q = record["question"]
    opts = record.get("options") or []
    hint = record.get("newHint") or ""
    knowledge = record.get("newKnowledgePoint") or ""

    system_parts = [_SOCRAT_ZH_SYSTEM, f"问题：{q}"]
    if opts:
        formatted = " ".join(f"({chr(65 + i)}) {o}" for i, o in enumerate(opts))
        system_parts.append(f"选项：{formatted}")
    if hint:
        system_parts.append(f"提示：{hint}")
    if knowledge:
        system_parts.append(f"知识点：{knowledge}")

    messages = [{"role": "system", "content": "\n".join(system_parts)}]
    states: list[str] = []

    for turn in record.get("dialogue", []):
        messages.append({"role": "user", "content": turn["student"]})
        state = _strip_quotes(turn.get("state", ""))
        action = _strip_quotes(turn.get("action", ""))
        teacher_content = turn["teacher"]
        if state or action:
            prefix = f"[State: {state}] [Action: {action}]\n" if state and action else ""
            teacher_content = prefix + teacher_content
        messages.append({"role": "assistant", "content": teacher_content})
        if state:
            states.append(state)

    return {
        "id": str(record["id"]),
        "source": "socrat-zh",
        "messages": messages,
        "ground_truth_states": states if states else None,
    }


def load_socrat_zh(
    split: str = "train",
    seed: int = 42,
    hf_repo: str = "ulises-c/SocratDataset",
) -> list[dict]:
    """Load the original Chinese SocratDataset from HuggingFace and convert to messages format."""
    from datasets import load_dataset as hf_load

    raw = [dict(r) for r in hf_load(hf_repo, split="train")]

    if split == "all":
        return [_socrat_zh_to_messages(r) for r in raw]

    rng = random.Random(seed)
    indices = list(range(len(raw)))
    rng.shuffle(indices)
    split_point = int(len(raw) * 0.9)
    if split == "train":
        subset = [raw[i] for i in sorted(indices[:split_point])]
    else:
        subset = [raw[i] for i in sorted(indices[split_point:])]

    return [_socrat_zh_to_messages(r) for r in subset]


# ---------------------------------------------------------------------------
# SocratDataset-EN
# ---------------------------------------------------------------------------


def _socrat_en_to_messages(record: dict) -> dict:
    """Convert one SocratDataset-EN record to the unified messages format.

    The consultant's evaluation and action are prepended to each teacher turn
    so the fine-tuned model learns state-conditioned response generation.
    At inference time these are supplied by the live consultant agent.
    """
    q = record["question"]
    opts = record.get("options") or []
    hint = record.get("newHint") or ""
    knowledge = record.get("newKnowledgePoint") or ""

    system_parts = [_SOCRAT_EN_SYSTEM, f"Problem: {q}"]
    if opts:
        formatted = " ".join(f"({chr(65 + i)}) {o}" for i, o in enumerate(opts))
        system_parts.append(f"Options: {formatted}")
    if hint:
        system_parts.append(f"Hint: {hint}")
    if knowledge:
        system_parts.append(f"Knowledge: {knowledge}")

    messages = [{"role": "system", "content": "\n".join(system_parts)}]
    states: list[str] = []

    for turn in record.get("dialogue", []):
        messages.append({"role": "user", "content": turn["student"]})
        state = turn.get("state", "")
        action = turn.get("action", "")
        teacher_content = turn["teacher"]
        if state or action:
            prefix = f"[State: {state}] [Action: {action}]\n" if state and action else ""
            teacher_content = prefix + teacher_content
        messages.append({"role": "assistant", "content": teacher_content})
        if state:
            states.append(state)

    return {
        "id": str(record["id"]),
        "source": "socrat-en",
        "messages": messages,
        "ground_truth_states": states if states else None,
    }


def load_socrat_en(
    split: str = "train",
    seed: int = 42,
    hf_repo: str = "ulises-c/SocratDataset-EN",
) -> list[dict]:
    """Load SocratDataset-EN from HuggingFace and convert to messages format.

    Uses the same 90/10 split logic as kele.load_dataset so train/test sets
    are identical between the evaluation and training pipelines.
    """
    from datasets import load_dataset as hf_load

    raw = [dict(r) for r in hf_load(hf_repo, split="train")]

    if split == "all":
        return [_socrat_en_to_messages(r) for r in raw]

    rng = random.Random(seed)
    indices = list(range(len(raw)))
    rng.shuffle(indices)
    split_point = int(len(raw) * 0.9)
    if split == "train":
        subset = [raw[i] for i in sorted(indices[:split_point])]
    else:
        subset = [raw[i] for i in sorted(indices[split_point:])]

    return [_socrat_en_to_messages(r) for r in subset]


# ---------------------------------------------------------------------------
# SocraTeach_Multi
# ---------------------------------------------------------------------------


def _socrateach_multi_record_to_messages(problem: dict, dlg: dict) -> dict:
    """Convert one SocraTeach_Multi dialogue to messages format.

    Each problem has multiple dialogues (one per student persona / path).
    We generate one training record per dialogue, not one per problem.

    Turn structure: teacher asks first (system field), student responds (user field).
    We map: system → assistant (teacher), user → user (student).
    """
    q = problem["question"]
    system_content = f"{_SOCRATEACH_SYSTEM}\nProblem: {q}"

    messages: list[dict] = [{"role": "system", "content": system_content}]
    for turn in dlg.get("turns", []):
        teacher_q = turn.get("system", "")
        student_a = turn.get("user", "")
        if teacher_q:
            messages.append({"role": "assistant", "content": teacher_q})
        if student_a:
            messages.append({"role": "user", "content": student_a})

    return {
        "id": dlg["dialogue_id"],
        "source": "socrateach-multi",
        "messages": messages,
        "ground_truth_states": None,
    }


def load_socrateach_multi(
    split: str = "train",
    seed: int = 42,
    hf_repo: str = "ulises-c/SocraTeach_Multi",
) -> list[dict]:
    """Load SocraTeach_Multi from HuggingFace and convert to messages format.

    Returns one record per (problem, dialogue) pair.
    """
    from datasets import load_dataset as hf_load

    hf_ds = hf_load(hf_repo, split="train")

    records: list[dict] = []
    for problem in hf_ds:
        problem = dict(problem)
        for dlg in problem.get("dialogues", []):
            if isinstance(dlg, dict):
                records.append(_socrateach_multi_record_to_messages(problem, dlg))

    if split == "all":
        return records

    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    split_point = int(len(records) * 0.9)
    if split == "train":
        return [records[i] for i in sorted(indices[:split_point])]
    else:
        return [records[i] for i in sorted(indices[split_point:])]


# ---------------------------------------------------------------------------
# SocraTeach_Single
# ---------------------------------------------------------------------------


def _socrateach_single_to_messages(record: dict) -> dict:
    """Convert one SocraTeach_Single record to messages format.

    History is a list of [user, assistant] pairs. The last exchange
    is (prompt, response). The first history entry often contains the
    system instruction as the user message — we extract it as the system role.
    """
    student_type = record.get("student_type", "")
    history = record.get("history") or []
    prompt = record.get("prompt", "")
    response = record.get("response", "")

    messages: list[dict] = []
    system_parts = [_SOCRATEACH_SYSTEM]
    if student_type:
        system_parts.append(f"Student type: {student_type}")

    # The first history pair typically contains the meta-instruction as the
    # user message (e.g. "You are a Socratic teacher, please guide me...").
    # Treat it as additional system context rather than a training turn.
    start_idx = 0
    if history and isinstance(history[0], (list, tuple)) and len(history[0]) >= 1:
        first_user = history[0][0] if history[0] else ""
        if isinstance(first_user, str) and first_user.startswith("You are a Socratic teacher"):
            system_parts.append(f"Task: {first_user[:300]}")
            start_idx = 1

    messages.append({"role": "system", "content": "\n".join(system_parts)})

    for pair in history[start_idx:]:
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            messages.append({"role": "user", "content": pair[0]})
            messages.append({"role": "assistant", "content": pair[1]})

    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "assistant", "content": response})

    return {
        "id": record["id"],
        "source": "socrateach-single",
        "messages": messages,
        "ground_truth_states": None,
    }


def load_socrateach_single(
    split: str = "train",
    seed: int = 42,
    hf_repo: str = "ulises-c/SocraTeach_Single",
) -> list[dict]:
    """Load SocraTeach_Single from HuggingFace and convert to messages format."""
    from datasets import load_dataset as hf_load

    hf_ds = hf_load(hf_repo, split="train")

    records = [_socrateach_single_to_messages(dict(r)) for r in hf_ds]

    if split == "all":
        return records

    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    split_point = int(len(records) * 0.9)
    if split == "train":
        return [records[i] for i in sorted(indices[:split_point])]
    else:
        return [records[i] for i in sorted(indices[split_point:])]


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

_SOURCE_LOADERS = {
    "socrat-zh": load_socrat_zh,
    "socrat-en": load_socrat_en,
    "socrateach-multi": load_socrateach_multi,
    "socrateach-single": load_socrateach_single,
}


def load_training_data(
    sources: list[str] | None = None,
    split: str = "train",
    seed: int = 42,
) -> list[dict]:
    """Load and combine training records from one or more sources.

    Args:
        sources: List of source keys to include. Defaults to all four.
            Valid values: "socrat-zh", "socrat-en", "socrateach-multi", "socrateach-single"
        split: "train", "test", or "all". Applied independently to each source.
        seed: Random seed for reproducible splits (same seed across all sources).

    Returns:
        Combined list of records in messages format, one record per dialogue.
    """
    if sources is None:
        sources = list(_SOURCE_LOADERS.keys())

    unknown = set(sources) - set(_SOURCE_LOADERS)
    if unknown:
        raise ValueError(f"Unknown sources: {unknown}. Valid: {set(_SOURCE_LOADERS)}")

    all_records: list[dict] = []
    for src in sources:
        loader = _SOURCE_LOADERS[src]
        records = loader(split=split, seed=seed)  # type: ignore[call-arg]
        all_records.extend(records)

    return all_records
