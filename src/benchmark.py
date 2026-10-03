from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tabulate import tabulate

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Calculate recall score (0.0 to 1.0) based on expected keywords."""
    if not expected:
        return 1.0
    ans_lower = answer.lower()
    matched = 0
    for exp in expected:
        if exp.lower() in ans_lower:
            matched += 1
    return matched / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight quality heuristic score combining recall, conciseness, and structure."""
    if not answer or not answer.strip():
        return 0.0

    recall = recall_points(answer, expected)
    # Penalize answers stating total lack of context
    if "chưa có thông tin" in answer.lower() or "không có thông tin" in answer.lower():
        return 0.1

    # Bonus for clean structure (e.g. bullets or structured text)
    structure_bonus = 0.2 if ("\n-" in answer or "- " in answer or "\n•" in answer) else 0.1
    # Bonus for reasonable length without excessive verbosity
    length = len(answer)
    length_bonus = 0.1 if 30 <= length <= 500 else 0.05

    quality = (0.7 * recall) + structure_bonus + length_bonus
    return min(1.0, round(quality, 3))


def run_agent_benchmark(
    agent_name: str, agent: Any, conversations: list[dict[str, Any]], config: Any
) -> BenchmarkRow:
    """Evaluate an agent across conversations and fresh recall threads."""
    total_agent_tokens = 0
    total_prompt_tokens = 0
    total_recall = 0.0
    total_quality = 0.0
    question_count = 0
    users_seen: set[str] = set()

    for conv in conversations:
        user_id = conv.get("user_id", "default_user")
        users_seen.add(user_id)
        conv_id = conv.get("id", "conv")
        main_thread_id = f"{conv_id}-main"

        # 1. Feed conversation turns sequentially
        for turn in conv.get("turns", []):
            res = agent.reply(user_id=user_id, thread_id=main_thread_id, message=turn)
            total_agent_tokens += res["tokens"]
            total_prompt_tokens += res["prompt_tokens"]

        # 2. Evaluate recall questions in a FRESH thread (simulating a new session)
        for i, q in enumerate(conv.get("recall_questions", [])):
            recall_thread_id = f"{conv_id}-recall-{i}"
            res = agent.reply(user_id=user_id, thread_id=recall_thread_id, message=q["question"])
            total_agent_tokens += res["tokens"]
            total_prompt_tokens += res["prompt_tokens"]

            r_score = recall_points(res["response"], q.get("expected_contains", []))
            q_score = heuristic_quality(res["response"], q.get("expected_contains", []))

            total_recall += r_score
            total_quality += q_score
            question_count += 1

    avg_recall = (total_recall / question_count) if question_count > 0 else 0.0
    avg_quality = (total_quality / question_count) if question_count > 0 else 0.0

    # Calculate memory growth on disk across all seen users
    memory_growth = 0
    if hasattr(agent, "memory_file_size"):
        for uid in users_seen:
            memory_growth += agent.memory_file_size(uid)

    # Calculate total compactions
    total_compactions = 0
    if hasattr(agent, "compact_memory"):
        for tid in getattr(agent.compact_memory, "state", {}):
            total_compactions += agent.compact_memory.compaction_count(tid)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=memory_growth,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format benchmark rows as a clean markdown table."""
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table_data = []
    for r in rows:
        table_data.append(
            [
                r.agent_name,
                f"{r.agent_tokens_only:,}",
                f"{r.prompt_tokens_processed:,}",
                f"{r.recall_score * 100:.1f}%",
                f"{r.response_quality * 100:.1f}%",
                f"{r.memory_growth_bytes:,} B",
                r.compactions,
            ]
        )
    return tabulate(table_data, headers=headers, tablefmt="github")


def main() -> None:
    """Execute both Standard Benchmark and Long-Context Stress Benchmark."""
    repo_root = Path(__file__).resolve().parent.parent
    config = load_config(repo_root)

    print("=" * 80)
    print("DAY 17: MEMORY SYSTEMS BENCHMARK (Baseline vs. Advanced)")
    print("=" * 80)

    # 1. Standard Benchmark
    conv_path = config.data_dir / "conversations.json"
    if conv_path.exists():
        print("\n--- 1. STANDARD BENCHMARK (10 Conversations, Cross-Session Recall) ---")
        conversations = load_conversations(conv_path)

        baseline_agent = BaselineAgent(config, force_offline=True)
        row_baseline = run_agent_benchmark("Baseline Agent", baseline_agent, conversations, config)

        advanced_agent = AdvancedAgent(config, force_offline=True)
        row_advanced = run_agent_benchmark("Advanced Agent", advanced_agent, conversations, config)

        print(format_rows([row_baseline, row_advanced]))

    # 2. Long-Context Stress Benchmark
    stress_path = config.data_dir / "advanced_long_context.json"
    if stress_path.exists():
        print("\n--- 2. LONG-CONTEXT STRESS BENCHMARK (16 Heavy Turns, Compaction Stress) ---")
        stress_conversations = load_conversations(stress_path)

        baseline_stress = BaselineAgent(config, force_offline=True)
        row_b_stress = run_agent_benchmark("Baseline Agent", baseline_stress, stress_conversations, config)

        advanced_stress = AdvancedAgent(config, force_offline=True)
        row_a_stress = run_agent_benchmark("Advanced Agent", advanced_stress, stress_conversations, config)

        print(format_rows([row_b_stress, row_a_stress]))

    print("\n" + "=" * 80)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
