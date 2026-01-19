FROM python:3.13-slim-bookworm

# 1) Install Python + basic build tools + curl (for uv install)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2) Install uv (Astral) as a binary
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Put uv on PATH for login shells and non-login shells
ENV PATH="/root/.local/bin:${PATH}"


# 3) Set working directory
WORKDIR /code

# 4) Copy project metadata first (for better layer caching)
COPY pyproject.toml uv.lock ./

# 5) Create and sync venv using uv (locked)
ENV UV_PROJECT_ENVIRONMENT=.venv
RUN uv sync --locked
RUN uv lock --check

# 6) Copy the actual application code and model 
COPY ./src /code/src
COPY ./artifacts /code/artifacts

# 7) DEBUG: prove they are there at build time
RUN echo "==== AFTER COPY ====" && ls -R /code

ENV PYTHONPATH=/code/src

EXPOSE 8000

# 8) Run FastAPI app via fastapi CLI
# Assuming your app object is in app/main.py
CMD ["/code/.venv/bin/python3", "-m", "uvicorn", "src.service:app", "--host", "0.0.0.0", "--port", "9100"]


### docker build -t flower_model_docker .
### docker run -p 127.0.0.1:9100:9100 flower_model_docker
### docker build --no-cache -t flower_model_docker .
### docker run -it --entrypoint /bin/bash flower_model_docker
### docker stop flower_model_docker
### docker rm flower_model_docker
### docker rmi flower_model_docker