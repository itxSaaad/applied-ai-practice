FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --extra dev --no-install-project

COPY . .
RUN uv sync --frozen --extra dev

EXPOSE 8000 8002
CMD ["uv", "run", "--no-sync", "uvicorn", "capstone.main:app", "--host", "0.0.0.0", "--port", "8000"]
