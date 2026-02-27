import os
import time
import uuid
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.routers.optimize import router as optimize_router


# ----------------------------
# Logging (simple + useful)
# ----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("api")


# ----------------------------
# App
# ----------------------------
app = FastAPI(
    title="Lieferkette Optimierungsplattform API",
    version="0.2.0",
    description="Supply Chain Optimization & Simulation API",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ----------------------------
# CORS (safe default: allow all; later restrict by env)
# ----------------------------
# برای حالت حرفه‌ای‌تر: بعداً ALLOWED_ORIGINS رو با دامنه‌های واقعی محدود می‌کنیم.
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in allowed_origins] if allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Routers
# ----------------------------
app.include_router(optimize_router)


# ----------------------------
# Middleware: Request-ID + timing + basic access log
# ----------------------------
@app.middleware("http")
async def add_request_id_and_log(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    start = time.time()

    try:
        response = await call_next(request)
    except Exception as e:
        # let the exception handlers format the response
        logger.exception(f"Unhandled error | request_id={request_id} | path={request.url.path}")
        raise e

    duration_ms = (time.time() - start) * 1000
    response.headers["x-request-id"] = request_id

    logger.info(
        f"{request.method} {request.url.path} -> {response.status_code} "
        f"({duration_ms:.1f}ms) | request_id={request_id}"
    )
    return response


# ----------------------------
# Error handlers (clean JSON)
# ----------------------------
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error": "http_exception",
            "detail": exc.detail,
            "path": request.url.path,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error": "internal_server_error",
            "detail": "Unexpected error occurred.",
            "path": request.url.path,
        },
    )


# ----------------------------
# Endpoints
# ----------------------------
@app.get("/", tags=["default"])
def root():
    return {
        "status": "ok",
        "message": "API is running",
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi": "/openapi.json",
    }


@app.get("/health", tags=["default"])
def health():
    return {"ok": True}