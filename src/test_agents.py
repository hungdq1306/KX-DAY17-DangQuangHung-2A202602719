from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated configuration for unit testing."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    dummy_model = ProviderConfig(
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.0,
    )

    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=state_dir,
        compact_threshold_tokens=80,  # Small threshold to trigger compaction in tests easily
        compact_keep_messages=2,
        model=dummy_model,
        judge_model=dummy_model,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify User.md can be created, read, edited, and monitored for file size."""
    profiles_dir = tmp_path / "state" / "profiles"
    store = UserProfileStore(profiles_dir)

    user_id = "test_user"
    init_content = "# User Profile: test_user\n\n- **Name:** DũngCT\n- **Location:** Huế\n"

    # 1. Write text
    file_path = store.write_text(user_id, init_content)
    assert file_path.exists()
    assert store.file_size(user_id) > 0

    # 2. Read text
    read_back = store.read_text(user_id)
    assert "DũngCT" in read_back
    assert "Huế" in read_back

    # 3. Edit text (e.g. location correction)
    edited = store.edit_text(user_id, "Location:** Huế", "Location:** Đà Nẵng")
    assert edited is True

    updated_content = store.read_text(user_id)
    assert "Đà Nẵng" in updated_content
    assert "Huế" not in updated_content


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long message threads trigger compaction in AdvancedAgent."""
    cfg = make_config(tmp_path)
    agent = AdvancedAgent(config=cfg, force_offline=True)

    thread_id = "test-compact-thread"
    user_id = "dungct_compact"

    # Each turn has around 40-50 tokens. With threshold=80, 4 turns will easily trigger compaction.
    long_turn_1 = "Đây là thông điệp thứ nhất rất dài về công nghệ AI và hệ thống xử lý ngôn ngữ tự nhiên."
    long_turn_2 = "Đây là thông điệp thứ hai tiếp tục bàn luận chi tiết về kiến trúc memory trong agent production."
    long_turn_3 = "Đây là thông điệp thứ ba về việc đánh giá hiệu năng và giảm chi phí prompt context."
    long_turn_4 = "Đây là thông điệp thứ tư để ép bộ quản lý compact memory phải kích hoạt việc nén ngữ cảnh."

    agent.reply(user_id, thread_id, long_turn_1)
    agent.reply(user_id, thread_id, long_turn_2)
    agent.reply(user_id, thread_id, long_turn_3)
    agent.reply(user_id, thread_id, long_turn_4)

    assert agent.compaction_count(thread_id) > 0

    ctx = agent.compact_memory.context(thread_id)
    assert len(ctx["messages"]) <= cfg.compact_keep_messages + 1  # Recent messages kept
    assert len(str(ctx.get("summary", ""))) > 0  # Summary was created


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify AdvancedAgent remembers across threads/sessions while BaselineAgent does not."""
    cfg = make_config(tmp_path)

    baseline = BaselineAgent(config=cfg, force_offline=True)
    advanced = AdvancedAgent(config=cfg, force_offline=True)

    user_id = "dungct"

    # Session 1 (thread-1): User shares their name and preference
    b_res1 = baseline.reply(user_id, "thread-1", "Chào bạn, mình tên là DũngCT. Đồ uống yêu thích là cà phê sữa đá.")
    a_res1 = advanced.reply(user_id, "thread-1", "Chào bạn, mình tên là DũngCT. Đồ uống yêu thích là cà phê sữa đá.")

    # Session 2 (thread-2, fresh thread): Ask for recall
    b_res2 = baseline.reply(user_id, "thread-2", "Mình tên gì và đồ uống yêu thích là gì?")
    a_res2 = advanced.reply(user_id, "thread-2", "Mình tên gì và đồ uống yêu thích là gì?")

    # Baseline should NOT know cross-session facts in thread-2
    assert "DũngCT" not in b_res2["response"]

    # Advanced MUST know facts via persistent User.md
    assert "DũngCT" in a_res2["response"]
    assert "cà phê sữa đá" in a_res2["response"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Verify compaction reduces prompt context tokens processed compared to baseline on long threads."""
    cfg = make_config(tmp_path)

    baseline = BaselineAgent(config=cfg, force_offline=True)
    advanced = AdvancedAgent(config=cfg, force_offline=True)

    thread_id = "long-eval-thread"
    user_id = "dungct_load"

    turns = [
        f"Lượt trò chuyện số {i}: Chúng ta cùng phân tích chi tiết về hiệu năng bộ nhớ AI agent và các phương pháp nén dữ liệu hội thoại trong môi trường production."
        for i in range(12)
    ]

    for turn in turns:
        baseline.reply(user_id, thread_id, turn)
        advanced.reply(user_id, thread_id, turn)

    baseline_prompt_tokens = baseline.prompt_token_usage(thread_id)
    advanced_prompt_tokens = advanced.prompt_token_usage(thread_id)

    # Advanced agent's compact memory ensures older turns are summarized rather than re-sent in full
    assert advanced_prompt_tokens < baseline_prompt_tokens
    assert advanced.compaction_count(thread_id) > 0
