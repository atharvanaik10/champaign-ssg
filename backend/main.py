from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.jobs import get_registry
from backend.routers.graph import router as graph_router
from backend.routers.jobs import router as jobs_router
from backend.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(title="ALMA API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs_router)
app.include_router(graph_router)


@app.on_event("startup")
def startup() -> None:
    get_registry(settings.max_workers)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content={"detail": exc.errors()})


@app.exception_handler(Exception)
async def global_exception_handler(_: Request, exc: Exception):
    logging.getLogger(__name__).exception("Unhandled error")
    return JSONResponse(status_code=500, content={"detail": str(exc)})
