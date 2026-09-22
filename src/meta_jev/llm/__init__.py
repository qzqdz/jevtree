"""Optional LLM helpers (assist only — never used for official scoring)."""

from meta_jev.llm.mimo import chat_completion, load_dotenv_env

__all__ = ["chat_completion", "load_dotenv_env"]
