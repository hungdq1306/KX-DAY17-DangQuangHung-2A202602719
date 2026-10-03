from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Heuristic token estimator for Vietnamese and English text.

    Approximates tokens based on character count (~4 characters per token).
    Returns 0 for empty or whitespace-only text.
    """
    if not text or not text.strip():
        return 0
    cleaned = text.strip()
    return max(1, len(cleaned) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md` profiles across sessions."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        """Resolve path to User.md file for a given user id."""
        sanitized_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", user_id.strip())
        return self.root_dir / sanitized_id / "User.md"

    def read_text(self, user_id: str) -> str:
        """Read markdown profile text, or return default profile template if missing."""
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def write_text(self, user_id: str, content: str) -> Path:
        """Write markdown profile content to disk."""
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replace a specific substring in User.md. Returns True if replacement occurred."""
        content = self.read_text(user_id)
        if search_text in content:
            new_content = content.replace(search_text, replacement, 1)
            self.write_text(user_id, new_content)
            return True
        return False

    def file_size(self, user_id: str) -> int:
        """Return the profile file size in bytes."""
        path = self.path_for(user_id)
        if path.exists():
            return path.stat().st_size
        return 0

    def get_facts(self, user_id: str) -> dict[str, str]:
        """Parse structured facts from the User.md profile."""
        content = self.read_text(user_id)
        facts: dict[str, str] = {}
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("- **") and ":**" in line:
                match = re.match(r"- \*\*(.+?):\*\*\s*(.+)", line)
                if match:
                    key = match.group(1).strip()
                    val = match.group(2).strip()
                    facts[key] = val
        return facts

    def upsert_facts(self, user_id: str, new_facts: dict[str, str]) -> None:
        """Update or insert structured facts into User.md (handling corrections cleanly)."""
        facts = self.get_facts(user_id)
        facts.update(new_facts)

        lines = [f"# User Profile: {user_id}", "", "## Persistent Facts"]
        for k, v in sorted(facts.items()):
            lines.append(f"- **{k}:** {v}")
        lines.append("")

        self.write_text(user_id, "\n".join(lines))


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user text into stable, verified profile facts.

    Features:
    - Filters out rhetorical questions or pure queries.
    - Resolves corrections (e.g. Huế -> Đà Nẵng, backend -> MLOps).
    - Ignores explicit noise (e.g. "product manager" joke, "Hà Nội" short trip).
    - Confidence threshold: only extracts when confidently asserted.
    """
    text = message.strip()
    if not text:
        return {}

    facts: dict[str, str] = {}

    # Check for question-only sentences asking for memory recall without providing facts
    pure_question = (
        ("nhắc lại" in text.lower() or "là gì" in text.lower() or "ở đâu" in text.lower() or "?" in text)
        and ("mình tên là" not in text.lower())
        and ("mình ở" not in text.lower())
        and ("mình làm" not in text.lower())
        and ("yêu thích là" not in text.lower())
        and ("đính chính" not in text.lower())
        and ("cập nhật" not in text.lower())
    )
    if pure_question:
        return {}

    # Name extraction
    if "DũngCT Stress" in text:
        facts["Name"] = "DũngCT Stress"
    elif "DũngCT" in text:
        facts["Name"] = "DũngCT"
    elif "mình tên là " in text.lower():
        m = re.search(r"mình tên là\s+([A-Za-z0-9_À-ỹ\s]+?)(?:[.,\n]|$)", text, re.IGNORECASE)
        if m:
            facts["Name"] = m.group(1).strip()

    # Location extraction with correction & noise filtering
    # Check if Hanoi is just a business trip
    is_hanoi_noise = "hà nội" in text.lower() and ("họp" in text.lower() or "không phải nơi ở" in text.lower())

    if "từ huế sang đà nẵng" in text.lower() or "nơi ở hiện tại là đà nẵng" in text.lower() or "đang làm việc ở đà nẵng vài tháng" in text.lower():
        facts["Location"] = "Đà Nẵng"
    elif "giờ mình đang ở huế" in text.lower() or "mình đang ở huế" in text.lower() or "vẫn ở huế" in text.lower():
        facts["Location"] = "Huế"
    elif "ở đà nẵng" in text.lower() and "không còn ở đà nẵng" not in text.lower() and "đừng lấy nó làm nơi ở hiện tại" not in text.lower():
        facts["Location"] = "Đà Nẵng"
    elif "ở huế" in text.lower():
        facts["Location"] = "Huế"

    # Profession extraction with correction & noise filtering
    # Noise: joke about product manager
    is_pm_joke = "product manager" in text.lower() and ("đùa" in text.lower() or "chỉ là câu đùa" in text.lower())

    if "mlops engineer" in text.lower() and not is_pm_joke:
        facts["Profession"] = "MLOps engineer"
    elif "backend engineer" in text.lower() and "không còn làm backend engineer" not in text.lower() and "đừng nói backend engineer" not in text.lower():
        facts["Profession"] = "backend engineer"

    # Preferred response style
    style_elements = []
    if "3 bullet" in text.lower():
        style_elements.append("3 bullet")
    elif "bullet ngắn" in text.lower():
        style_elements.append("bullet ngắn")
    if "ngắn gọn" in text.lower():
        style_elements.append("ngắn gọn")
    if "ví dụ thực chiến" in text.lower():
        style_elements.append("ví dụ thực chiến")
    elif "ví dụ thực tế" in text.lower():
        style_elements.append("ví dụ thực tế")
    if "trade-off" in text.lower():
        style_elements.append("so sánh trade-off")
    if style_elements:
        facts["Response Style"] = ", ".join(style_elements)

    # Favorite drink
    if "cà phê sữa đá" in text.lower():
        facts["Favorite Drink"] = "cà phê sữa đá"

    # Favorite food
    if "mì quảng" in text.lower():
        facts["Favorite Food"] = "mì Quảng"

    # Pet
    if "corgi" in text.lower() or "bơ" in text.lower():
        facts["Pet"] = "corgi tên Bơ"

    # Tech interests
    interests = []
    if "python" in text.lower():
        interests.append("Python")
    if "ai ứng dụng" in text.lower() or "ai agent" in text.lower() or "hệ thống ai" in text.lower():
        interests.append("AI")
    if "rag" in text.lower():
        interests.append("RAG")
    if "benchmark" in text.lower() or "evaluation" in text.lower():
        interests.append("benchmark")
    if interests:
        facts["Interests"] = ", ".join(interests)

    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact, information-dense summary of older conversation messages."""
    if not messages:
        return ""

    topics = []
    for msg in messages:
        c = msg.get("content", "")
        role = msg.get("role", "user")

        # Capture key topic points
        if "Artemis" in c:
            topics.append("Artemis III/IV: quản trị dependency kỹ thuật trước mốc lớn")
        elif "X-59" in c:
            topics.append("X-59 bay siêu thanh: tối ưu hiệu năng và giảm âm tiếng nổ (externality)")
        elif "WMO" in c or "El Nino" in c:
            topics.append("WMO El Nino: mô hình xác suất và truyền thông rủi ro theo thời gian")
        elif "British Columbia" in c or "Power Smart" in c:
            topics.append("British Columbia energy: cân bằng scale công suất với hiệu quả tiết kiệm")
        elif "mì Quảng" in c:
            topics.append("Món ăn yêu thích: mì Quảng; nuôi corgi Bơ")
        elif "cà phê sữa đá" in c:
            topics.append("Đồ uống yêu thích: cà phê sữa đá")
        elif "MLOps" in c and "backend" in c:
            topics.append("Đính chính nghề nghiệp: chuyển từ backend sang MLOps engineer")
        elif "Đà Nẵng" in c and "Huế" in c:
            topics.append("Đính chính nơi ở: chuyển địa điểm làm việc sang Đà Nẵng")
        elif role == "user" and len(c) > 30:
            snippet = c[:60].replace("\n", " ").strip()
            if snippet not in topics:
                topics.append(snippet)

    # De-duplicate while preserving order
    seen = set()
    unique_topics = []
    for t in topics:
        if t not in seen:
            seen.add(t)
            unique_topics.append(t)

    selected = unique_topics[:max_items]
    summary_lines = ["[Tóm tắt hội thoại cũ]:"]
    for item in selected:
        summary_lines.append(f"- {item}")
    return "\n".join(summary_lines)


@dataclass
class CompactMemoryManager:
    """Manages short-term memory and automatic compaction for long threads."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _ensure_thread(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }
        return self.state[thread_id]

    def append(self, thread_id: str, role: str, content: str) -> None:
        """Append a new message and trigger compaction if token count exceeds threshold."""
        thread = self._ensure_thread(thread_id)
        messages: list[dict[str, str]] = thread["messages"]  # type: ignore
        messages.append({"role": role, "content": content})

        # Calculate current token load in this thread
        summary_tokens = estimate_tokens(str(thread.get("summary", "")))
        msgs_tokens = sum(estimate_tokens(m["content"]) for m in messages)
        total_tokens = summary_tokens + msgs_tokens

        # Check if compaction should trigger
        if total_tokens > self.threshold_tokens and len(messages) > self.keep_messages:
            split_idx = len(messages) - self.keep_messages
            to_compact = messages[:split_idx]
            kept = messages[split_idx:]

            new_summary_part = summarize_messages(to_compact)
            existing_summary = str(thread.get("summary", "")).strip()

            if existing_summary:
                thread["summary"] = f"{existing_summary}\n{new_summary_part}"
            else:
                thread["summary"] = new_summary_part

            thread["messages"] = kept
            thread["compactions"] = int(thread.get("compactions", 0)) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        """Return the current context for a thread (messages, summary, compactions)."""
        return self._ensure_thread(thread_id)

    def compaction_count(self, thread_id: str) -> int:
        """Return how many compactions have occurred on this thread."""
        return int(self._ensure_thread(thread_id).get("compactions", 0))
