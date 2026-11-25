import asyncio
import os
import sys
from pathlib import Path

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents import LlmAgent, LoopAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from day4_adk.model_config import get_adk_model

writer_agent = LlmAgent(
    name="draft_writer",
    model=get_adk_model(),
    instruction=(
        "Write a concise incident postmortem based on the user's request. "
        "Include impact, likely cause, and follow-up in at most three bullets. "
        "Incorporate critic feedback if present; do not include reasoning."
    ),
    output_key="postmortem_draft",
)


def stop_loop_when_approved(callback_context: CallbackContext):
    feedback = str(callback_context.state.get("critic_feedback", "")).strip()
    if feedback.upper().startswith("PASSED"):
        callback_context.actions.escalate = True
        return types.Content(role="model", parts=[])


critic_agent = LlmAgent(
    name="draft_critic",
    model=get_adk_model(),
    instruction=(
        "Review 'postmortem_draft' for accuracy, clarity, and completeness. "
        "If adequate, respond with exactly PASSED. Otherwise provide one concise "
        "revision instruction. Do not include reasoning."
    ),
    output_key="critic_feedback",
    after_agent_callback=stop_loop_when_approved,
)

# Bounded critique loop
refinement_loop = LoopAgent(
    name="refinement_loop",
    sub_agents=[writer_agent, critic_agent],
    max_iterations=3,
)


@config.trace_operation("day4_critique_loop_run", flush=True)
async def main():
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="day4_loop", user_id="dev_user", session_id="sess_loop"
    )
    runner = Runner(
        agent=refinement_loop,
        app_name="day4_loop",
        session_service=session_service,
    )
    msg = types.Content(
        role="user",
        parts=[types.Part(text="Write a concise postmortem for a 10-minute Redis cache eviction spike.")],
    )
    print("[CritiqueLoop]: Executing writer-critic refinement loop...")
    async def consume_events():
        async for event in runner.run_async(
            user_id="dev_user",
            session_id=session.id,
            new_message=msg,
        ):
            if event.is_final_response() and event.content:
                for part in event.content.parts:
                    if part.text:
                        print(f"[{event.author}]: {part.text}\n")

    await asyncio.wait_for(consume_events(), timeout=240)


if __name__ == "__main__":
    asyncio.run(main())
