import asyncio

from google.adk.agents import LlmAgent, LoopAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

import config
from day4_adk.model_config import get_adk_model, rate_limit_before_model_callback
from day5_observability.pattern1_dual_observability import (
    configure_if_enabled,
    flush_observability,
)

writer_agent = LlmAgent(
    name="draft_writer",
    model=get_adk_model(),
    instruction=(
        "Write a concise incident postmortem based on the user's request. "
        "Include impact, likely cause, and follow-up in at most three bullets. "
        "Incorporate critic feedback if present; do not include reasoning."
    ),
    output_key="postmortem_draft",
    before_model_callback=rate_limit_before_model_callback,
)


def stop_loop_when_approved(callback_context: CallbackContext):
    iteration = int(callback_context.state.get("critique_iteration", 0)) + 1
    callback_context.state["critique_iteration"] = iteration
    feedback = str(callback_context.state.get("critic_feedback", "")).strip()
    if feedback.upper().startswith("PASSED"):
        config.log_metric(
            "critique_iterations",
            iteration,
            stop_reason="score_passed",
        )
        callback_context.actions.escalate = True
        return types.Content(role="model", parts=[])
    if iteration >= refinement_loop.max_iterations:
        config.log_metric(
            "critique_iterations",
            iteration,
            stop_reason="maximum_iterations",
        )
    return None


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
    before_model_callback=rate_limit_before_model_callback,
)

# Bounded critique loop
refinement_loop = LoopAgent(
    name="refinement_loop",
    sub_agents=[writer_agent, critic_agent],
    max_iterations=3,
)


@config.trace_operation("day4_critique_loop_run", flush=True)
async def main():
    tracer_provider = configure_if_enabled()
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name="day4_loop", user_id="dev_user", session_id="sess_loop")
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
                for part in event.content.parts or []:
                    if part.text:
                        print(f"[{event.author}]: {part.text}\n")

    try:
        await asyncio.wait_for(
            consume_events(),
            timeout=config.ADK_AGENT_RUN_TIMEOUT_SECONDS,
        )
    finally:
        if tracer_provider is not None:
            flush_observability(tracer_provider)


if __name__ == "__main__":
    asyncio.run(main())
