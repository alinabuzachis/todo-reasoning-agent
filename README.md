# To-Do Agent: A Universal Worker-Evaluator Framework

This repository contains a dual-model, agentic AI workflow built in Python using Gradio and the OpenAI SDK. It uses a **Worker-Evaluator** architecture to solve problems, plan tasks, and self-correct its own mistakes before presenting the final answer to the user.

## Architecture

Instead of relying on a single Large Language Model (LLM) to get everything right on the first try, this framework splits the workload:
1. **The Worker**: A fast, instruction-following model that handles tool calling (creating to-do lists, marking them complete) and attempts to solve the user's prompt.
2. **The Evaluator**: A reasoning-focused model (e.g., DeepSeek-R1, GPT-4o) that reviews the Worker's final answer. If the answer is incorrect, poorly formatted, or raw JSON, the Evaluator rejects it and sends feedback back to the Worker for a retry.

## Prerequisites

This project uses [`uv`](https://github.com/astral-sh/uv) for lightning-fast Python dependency and environment management.

1. Install `uv`:
   * **Mac/Linux**: `curl -sSf https://astral.sh/uv/install.sh | sh`
   * **Windows**: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

## Quick Start

1. Clone the repository:
   ```bash
   git clone [https://github.com/alinabuzachis/todo-reasoning-agent.git](https://github.com/alinabuzachis/todo-reasoning-agent.git)
   cd todo-reasoning-agent
   ```

2. Set up your environment variables:

   Use your preferred LLM provider and add it to your `.env` file. For example, if you have Ollama installed and run `ollama pull llama3.2:3b` and `ollama pull deepseek-r1:1.5b`.

   ```bash
   OPENAI_BASE_URL=http://localhost:11434/v1
   OPENAI_API_KEY=ollama
   WORKER_MODEL=llama3.2:3b
   REASONER_MODEL=deepseek-r1:1.5b
   ```

3. Run the app:

   `uv` will automatically create an isolated virtual environment, install all dependencies from the `pyproject.toml` or `uv.lock` file, and launch the Gradio server.

   ```bash
   uv run app.py
   ```
