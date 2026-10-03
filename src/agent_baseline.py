from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Baseline Agent (Agent A).

    Characteristics:
    - Maintains only short-term memory within the same thread.
    - Has no persistent storage (no User.md).
    - Cannot recall facts across different threads or sessions.
    - Retains full history without compaction, causing prompt load to grow linearly with thread length.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Process incoming message and return response with token accounting."""
        if self.langchain_agent is not None and not self.force_offline:
            try:
                # Live LLM execution if configured
                return self._reply_live(user_id, thread_id, message)
            except Exception:
                # Fallback to offline deterministic mode if live execution fails
                return self._reply_offline(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        """Return cumulative generated tokens for the given thread."""
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        """Return cumulative prompt context tokens processed for the given thread."""
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        """Baseline agent has no compaction mechanism."""
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic offline reply logic for reproducible benchmarks."""
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        session = self.sessions[thread_id]

        # In baseline, the entire message history is sent as prompt context each turn
        history_tokens = sum(estimate_tokens(m["content"]) for m in session.messages)
        new_prompt_tokens = estimate_tokens(message)
        prompt_tokens = history_tokens + new_prompt_tokens

        # Check if this thread has any prior conversation
        has_prior_history = len(session.messages) > 0

        # Baseline cannot recall cross-session facts if asking in a new thread
        if not has_prior_history:
            reply_text = (
                "Chào bạn. Tôi là Baseline Agent. Trong phiên trò chuyện mới này, "
                "tôi chưa có thông tin hay lịch sử nào trước đó của bạn."
            )
        else:
            # Within the same thread, baseline can find facts in its message list
            reply_text = self._offline_in_thread_response(session.messages, message)

        reply_tokens = estimate_tokens(reply_text)
        session.token_usage += reply_tokens
        session.prompt_tokens_processed += prompt_tokens

        session.messages.append({"role": "user", "content": message})
        session.messages.append({"role": "assistant", "content": reply_text})

        return {
            "response": reply_text,
            "tokens": reply_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _offline_in_thread_response(self, messages: list[dict[str, str]], message: str) -> str:
        """Simple in-thread factual lookup for same-session baseline turns."""
        combined = " ".join(m["content"] for m in messages)
        msg_lower = message.lower()

        # If asked for a summary or recall within the same thread
        if "tên" in msg_lower and "đồ uống" in msg_lower and "DũngCT" in combined:
            return "Trong phiên này, bạn đã giới thiệu tên là DũngCT và đồ uống yêu thích là cà phê sữa đá."
        return "Tôi đã ghi nhận thông tin của bạn trong phiên trò chuyện này."

    def _maybe_build_langchain_agent(self):
        """Optionally build a live LangChain model if API key is provided."""
        if not self.config.model.api_key and self.config.model.provider not in ("ollama", "custom"):
            return None
        try:
            return build_chat_model(self.config.model)
        except Exception:
            return None

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Execute reply using live ChatModel with in-memory session history."""
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        session = self.sessions[thread_id]

        history_tokens = sum(estimate_tokens(m["content"]) for m in session.messages)
        new_prompt_tokens = estimate_tokens(message)
        prompt_tokens = history_tokens + new_prompt_tokens

        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        system_msg = SystemMessage(
            content="Bạn là một AI assistant hữu ích. Bạn chỉ nhớ những gì được nói trong phiên này."
        )
        chat_messages = [system_msg]
        for m in session.messages:
            if m["role"] == "user":
                chat_messages.append(HumanMessage(content=m["content"]))
            else:
                chat_messages.append(AIMessage(content=m["content"]))
        chat_messages.append(HumanMessage(content=message))

        ai_response = self.langchain_agent.invoke(chat_messages)
        reply_text = str(ai_response.content)
        reply_tokens = estimate_tokens(reply_text)

        session.token_usage += reply_tokens
        session.prompt_tokens_processed += prompt_tokens
        session.messages.append({"role": "user", "content": message})
        session.messages.append({"role": "assistant", "content": reply_text})

        return {
            "response": reply_text,
            "tokens": reply_tokens,
            "prompt_tokens": prompt_tokens,
        }
