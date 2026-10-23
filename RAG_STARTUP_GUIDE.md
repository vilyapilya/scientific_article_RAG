# Research RAG Assistant — Startup Guide

This guide explains how to start the complete RAG demo on a Mac and make its Gradio frontend accessible through a Cloudflare Quick Tunnel.

## Prerequisites

- Docker Desktop installed and running
- Ollama installed with the `llama3.2:3b` model
- Python virtual environment (`.venv`) with Gradio and project dependencies
- `cloudflared` installed
- Project directory containing `Dockerfile`, `main.py`, and `app.py`
- install dependencies `python -m pip install -r requirements.txt`

## 1. Start Ollama

Open the Ollama application, or run in **Terminal 1**:

```bash
ollama serve
```

If Ollama is already running, do not start another instance.

Check that the model is available:

```bash
ollama list
```

If needed, download it once:

```bash
ollama pull llama3.2:3b
```

## 2. Start the FastAPI backend in Docker

Open **Terminal 2** and navigate to the project directory:

```bash
cd /path/to/your/project
```

Build the image **only when backend code, Dockerfile, or requirements change** (or on first setup):

```bash
docker build -t capstone-rag .
```

Start the backend:

```bash
docker run --rm \
  -p 127.0.0.1:8000:8000 \
  -e OLLAMA_URL=http://host.docker.internal:11434 \
  capstone-rag
```

Check its health from another terminal:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

Optional: test the streaming endpoint:

```bash
curl -N -X POST http://127.0.0.1:8000/predict/stream \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is retrieval-augmented generation?"}'
```

You should see `data:` events with stages followed by an answer.

## 3. Start the Gradio frontend

Open **Terminal 3**:

```bash
cd /path/to/your/project
source .venv/bin/activate
python app.py
```

Open **http://127.0.0.1:7860** in your browser.

The frontend should use this backend URL in `app.py`:

```python
API_URL = "http://127.0.0.1:8000/predict/stream"
```

## 4. Make the frontend public with Cloudflare Tunnel

Open **Terminal 4**:

```bash
cloudflared tunnel --url http://127.0.0.1:7860
```

Copy the newly generated URL, for example:

```text
https://random-words.trycloudflare.com
```

Open the URL from another browser or device to verify that questions return answers. **Share this URL for live demonstrations**, not the local `127.0.0.1` address.

> Cloudflare Quick Tunnel URLs are temporary and may change each time the tunnel restarts.

## Startup order — quick reference

1. Start **Ollama** (`ollama serve`, unless already running).
2. Start **Docker Desktop**, then the **FastAPI Docker container**.
3. Start **Gradio** (`python app.py`).
4. Start **Cloudflare Tunnel** (`cloudflared tunnel --url http://127.0.0.1:7860`).
5. Test the **new public URL**.

## Stop the services

Press `Ctrl+C` in the Cloudflare, Gradio, and Docker terminals. Stop Ollama using its app, if desired.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `/health` does not respond | Docker Desktop and the backend container are running; port `8000` is available. |
| `/predict/stream` returns 404 | Rebuild the Docker image and restart the container to load the current `main.py`. |
| `Model inference failed` | Check the Docker logs for the Python traceback and verify Ollama is running. |
| Gradio cannot connect to backend | Ensure `API_URL` points to `http://127.0.0.1:8000/predict/stream`. |
| Cloudflare shows 530 | Verify Gradio responds locally on port `7860`; restart the tunnel and use its new URL. |
| Stages appear locally but not publicly | Intermediate Gradio streaming updates may be delayed through the tunnel; the final answer can still work. |
| Requests return 429 | The backend rate limit or single-inference lock is active; wait and retry. |

## Important limitations

- The Mac must stay **powered on, awake, and connected to the internet** during the demo.
- Ollama, the Docker backend, Gradio, and `cloudflared` must remain running.
- Cloudflare Tunnel exposes the Gradio interface; it does **not** host the model.
- The public Gradio link is accessible to anyone with the URL. Keep rate limits enabled and avoid exposing Ollama directly.
