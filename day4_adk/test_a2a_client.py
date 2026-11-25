"""Smoke client for discovering and submitting a task to the local A2A server."""

import asyncio

import httpx

BASE_URL = "http://127.0.0.1:8000/a2a/v1"


async def main() -> None:
    async with httpx.AsyncClient(timeout=120) as client:
        card_response = await client.get(f"{BASE_URL}/agent-card")
        card_response.raise_for_status()
        card = card_response.json()
        assert card["endpoints"]["task"] == "/a2a/v1/tasks"
        print("Discovered Agent Card:", card["name"])

        task_response = await client.post(
            f"{BASE_URL}/tasks",
            json={"prompt": "Analyze the auth-service error spike."},
        )
        task_response.raise_for_status()
        task = task_response.json()
        assert task.get("status") in {"completed", "submitted", "working"}
        print("Task submission:", task.get("status"))


if __name__ == "__main__":
    asyncio.run(main())
