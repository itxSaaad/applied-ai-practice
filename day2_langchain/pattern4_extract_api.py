import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from langchain.chat_models import init_chat_model

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TicketExtractionAPI")

app = FastAPI(title="Ticket Extraction API")


class Ticket(BaseModel):
    category: str = Field(description="Issue category e.g., Billing, Infrastructure, Auth")
    priority: str = Field(description="Priority classification: Low, Medium, High, Critical")
    product: str = Field(description="Impacted product or service component")
    sentiment: str = Field(description="Customer sentiment: Positive, Neutral, Negative")
    summary: str = Field(description="One sentence summary of the core issue")


llm = init_chat_model(
    f"google_genai:{os.getenv('LLM_MODEL', config.LLM_MODEL)}",
    temperature=0.0,
)

# Configure structured output with an automated single-retry policy on validation failure
structured_extractor = llm.with_structured_output(Ticket).with_retry(stop_after_attempt=2)


@app.post("/extract_ticket", response_model=Ticket)
@config.trace_operation(
    "day2_ticket_extraction",
    capture_input=False,
    capture_output=False,
    flush=True,
)
async def extract_ticket_endpoint(raw_text: str):
    try:
        callbacks = config.get_callbacks()
        invoke_config = {"callbacks": callbacks} if callbacks else {}
        extracted_ticket = structured_extractor.invoke(
            f"Analyze the following support ticket and extract structured properties:\n\n{raw_text}",
            config=invoke_config,
        )
        return extracted_ticket
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to extract structured payload: {str(e)}")


if __name__ == "__main__":
    sample_ticket = "Production checkout API has returned 504 gateway timeouts for 15 minutes."
    callbacks = config.get_callbacks()
    invoke_config = {"callbacks": callbacks} if callbacks else {}

    try:
        extracted = structured_extractor.invoke(
            f"Analyze the following support ticket and extract structured properties:\n\n{sample_ticket}",
            config=invoke_config,
        )
        config.flush_traces()
        print(extracted.model_dump_json(indent=2))
    except Exception as e:
        logger.warning(f"Extract API test note: {e}")
