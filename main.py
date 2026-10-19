
import asyncio
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware

from model import rag

app = FastAPI(
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["10/minute"]
)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# Only one inference at a time
inference_lock = asyncio.Lock()

class QueryRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=1000
    )

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
@limiter.limit("3/minute")
async def predict(request: Request, data: QueryRequest):
    if inference_lock.locked():
        raise HTTPException(
            status_code=429,
            detail="Model is busy. Try again later."
        )

    async with inference_lock:
        try:
            result = await asyncio.to_thread(
                rag.run,
                data.question
            )
            return {"answer": result["answer"]}

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail="Model inference failed"
            ) from e
import asyncio
import json
import threading
from fastapi.responses import StreamingResponse



@app.post("/predict/stream")
@limiter.limit("3/minute")
async def predict_stream(request: Request, data: QueryRequest):

    # Reject requests when the model is busy
    if inference_lock.locked():
        raise HTTPException(
            status_code=429,
            detail="Model is busy. Try again later."
        )

    async def events():
        async with inference_lock:
            loop = asyncio.get_running_loop()
            queue = asyncio.Queue()

            def report(stage):
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"stage": stage}
                )

            def worker():
                try:
                    result = rag.run(
                        data.question,
                        on_progress=report
                    )
                    report_result = {
                        "answer": result["answer"]
                    }

                except Exception:
                    import traceback
                    traceback.print_exc()
                    report_result = {
                        "error": "Model inference failed"
                    }

                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    report_result
                )

            thread = threading.Thread(
                target=worker,
                daemon=True
            )
            thread.start()

            while True:
                event = await queue.get()

                yield (
                    f"data: {json.dumps(event)}\n\n"
                )

                if "answer" in event or "error" in event:
                    break

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache"
        }
    )
