"""Download SocraticMath from GitHub, convert to HF dataset, and upload to HuggingFace.

Usage:
    uv run python scripts/upload_socraticmath_to_hf.py
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

GITHUB_RAW = "https://raw.githubusercontent.com/ECNU-ICALK/SocraticMath/main/data"
HF_REPO = "ulises-c/SocraticMATH"

FILES = [
    ("SocratesMATH_traindata.jsonl", "train"),
    ("SocratesMATH_valdata.jsonl", "val"),
    ("SocratesMATH_testdata.jsonl", "test"),
    ("SocratesMATH_traindata_sol.jsonl", "train"),
    ("SocratesMATH_valdata_sol.jsonl", "val"),
    ("SocratesMATH_testdata_sol.jsonl", "test"),
]


def reshape_test_pairs(text: str) -> list[dict]:
    """Reshape test data from turn pairs to conversation records."""
    lines = [ln for ln in text.strip().split("\n") if ln.strip()]
    convos: list[dict] = []
    current_turns: list[list[str]] = []
    for line in lines:
        data = json.loads(line)
        student, teacher = str(data[0]), str(data[1])
        if "老师：" not in student and current_turns:
            convs = []
            first_student = current_turns[0][0]
            convs.append({"from": "user", "value": first_student})
            for _s, t in current_turns:
                convs.append({"from": "assistant", "value": t})
            convos.append(
                {
                    "id": len(convos) + 1,
                    "conversations": convs,
                }
            )
            current_turns = []
        current_turns.append([student, teacher])

    if current_turns:
        convs = []
        first_student = current_turns[0][0]
        convs.append({"from": "user", "value": first_student})
        for _s, t in current_turns:
            convs.append({"from": "assistant", "value": t})
        convos.append(
            {
                "id": len(convos) + 1,
                "conversations": convs,
            }
        )

    return convos


def fetch_conversations(url: str) -> list[dict]:
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    raw = resp.text
    text = raw.strip()
    if text.startswith("["):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                for r in data:
                    r["id"] = int(r["id"])
                return data
        except json.JSONDecodeError:
            pass
    return reshape_test_pairs(raw)


def main() -> None:
    from datasets import Dataset, DatasetDict
    from huggingface_hub import HfApi

    all_data: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    sol_all_data: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    def load_one(filename: str, split: str) -> tuple[str, str, list[dict]]:
        url = f"{GITHUB_RAW}/{filename}"
        tag = "sol" if "_sol" in filename else "base"
        print(f"  Fetching {filename} ({tag}) …")
        records = fetch_conversations(url)
        print(f"    {len(records)} records")
        return filename, split, records

    with ThreadPoolExecutor(max_workers=6) as pool:
        fut_to_key = {pool.submit(load_one, f, s): (f, s) for f, s in FILES}
        for fut in as_completed(fut_to_key):
            filename, split, records = fut.result()
            is_sol = "_sol" in filename
            target = sol_all_data if is_sol else all_data
            target[split].extend(records)

    # Push base version
    print(f"\n--- Pushing base dataset to {HF_REPO} ---")
    HfApi().create_repo(HF_REPO, repo_type="dataset", exist_ok=True)
    splits = {}
    for split_name in ["train", "val", "test"]:
        records = all_data.get(split_name, [])
        if not records:
            continue
        records.sort(key=lambda r: r["id"])
        clean = [{"id": r["id"], "conversations": r["conversations"]} for r in records]
        splits[split_name] = Dataset.from_list(clean)
        print(f"  {split_name}: {len(clean)} records")

    DatasetDict(splits).push_to_hub(
        HF_REPO,
        commit_message="Initial upload of SocraticMATH from ECNU-ICALK/SocraticMath",
    )
    print(f"  https://huggingface.co/datasets/{HF_REPO}")

    # Push sol version
    sol_repo = f"{HF_REPO}-sol"
    print(f"\n--- Pushing sol dataset to {sol_repo} ---")
    HfApi().create_repo(sol_repo, repo_type="dataset", exist_ok=True)
    sol_splits = {}
    for split_name in ["train", "val", "test"]:
        records = sol_all_data.get(split_name, [])
        if not records:
            continue
        records.sort(key=lambda r: r["id"])
        clean = [{"id": r["id"], "conversations": r["conversations"]} for r in records]
        sol_splits[split_name] = Dataset.from_list(clean)
        print(f"  {split_name}: {len(clean)} records")

    DatasetDict(sol_splits).push_to_hub(
        sol_repo,
        commit_message="Initial upload of SocraticMATH with solutions from ECNU-ICALK/SocraticMath",
    )
    print(f"  https://huggingface.co/datasets/{sol_repo}")


if __name__ == "__main__":
    main()
