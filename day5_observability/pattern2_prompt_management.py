import config
from utils.logging import get_logger

logger = get_logger(__name__)
LOCAL_PROMPT_PATH = config.DATA_DIR / "prompts" / "ops_pilot_supervisor.txt"
DEFAULT_SUPERVISOR_PROMPT = LOCAL_PROMPT_PATH.read_text(encoding="utf-8").strip()


@config.trace_operation("day5_prompt_fetch", capture_input=False, capture_output=False)
def fetch_production_system_prompt(prompt_name: str) -> str:
    """Fetch the production prompt, falling back locally if keys or prompt are unavailable."""
    if not config.LANGFUSE_PUBLIC_KEY or not config.LANGFUSE_SECRET_KEY:
        return DEFAULT_SUPERVISOR_PROMPT

    try:
        langfuse = config.get_langfuse_client()
        if langfuse is None:
            return DEFAULT_SUPERVISOR_PROMPT
        prompt = langfuse.get_prompt(name=prompt_name, label="production")
        return prompt.prompt
    except Exception:
        logger.exception(
            "Could not fetch prompt %s at production label; using checked-in fallback.",
            prompt_name,
        )
        return DEFAULT_SUPERVISOR_PROMPT


if __name__ == "__main__":
    system_prompt_text = fetch_production_system_prompt("ops_pilot_supervisor_prompt")
    print("Loaded Production System Prompt:\n", system_prompt_text)
