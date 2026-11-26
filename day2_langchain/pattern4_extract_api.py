from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config
from llm import get_chat_model
from utils.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Ticket Extraction API")


class Ticket(BaseModel):
    category: str = Field(description="Issue category e.g., Billing, Infrastructure, Auth")
    priority: str = Field(description="Priority classification: Low, Medium, High, Critical")
    product: str = Field(description="Impacted product or service component")
    sentiment: str = Field(description="Customer sentiment: Positive, Neutral, Negative")
    summary: str = Field(description="One sentence summary of the core issue")


def build_structured_extractor():
    """Create the provider-backed extractor lazily so schema checks stay offline."""
    llm = get_chat_model(
        f"google_genai:{config.LLM_MODEL}",
        temperature=0.0,
    )
    return llm.with_structured_output(Ticket).with_retry(stop_after_attempt=2)


@app.post("/extract_ticket", response_model=Ticket)
@config.trace_operation(
    "day2_ticket_extraction",
    capture_input=False,
    capture_output=False,
    flush=True,
)
async def extract_ticket_endpoint(raw_text: str):
    try:
        structured_extractor = build_structured_extractor()
        invoke_config = config.get_run_config()
        return await structured_extractor.ainvoke(
            f"Analyze the following support ticket and extract structured properties:\n\n{raw_text}",
            config=invoke_config,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to extract structured payload: {str(e)}") from e


if __name__ == "__main__":
    sample_ticket = "Production checkout API has returned 504 gateway timeouts for 15 minutes."
    structured_extractor = build_structured_extractor()
    invoke_config = config.get_run_config()

    try:
        extracted = structured_extractor.invoke(
            f"Analyze the following support ticket and extract structured properties:\n\n{sample_ticket}",
            config=invoke_config,
        )
        config.flush_traces()
        print(extracted.model_dump_json(indent=2))
    except Exception as e:
        logger.warning(f"Extract API test note: {e}")
