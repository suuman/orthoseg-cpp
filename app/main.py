import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from app.api.routes import router
from app.api.management import router as management_router
from app.management.jobs import Management
from app.core.config import load_config
from app.core.logging import configure_logging
from app.ml.predictor import Predictor
from app.ml.store import CaseStore


class RequestLimit:
    """Bound streamed multipart bodies before the multipart parser consumes them."""
    def __init__(self, app, maximum):
        self.app, self.maximum = app, maximum

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        count = 0
        async def limited():
            nonlocal count
            message = await receive()
            count += len(message.get("body", b""))
            if count > self.maximum:
                from starlette.exceptions import HTTPException
                raise HTTPException(413, "Request exceeds maximum size")
            return message
        headers = dict(scope.get("headers", []))
        try:
            size = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await JSONResponse({"detail": "Invalid Content-Length"}, 400)(scope, receive, send)
        if size > self.maximum:
            return await JSONResponse({"detail": "Request exceeds maximum size"}, 413)(scope, receive, send)
        await self.app(scope, limited, send)


def create_app(config=None, predictor=None):
    config = config or load_config()
    @asynccontextmanager
    async def lifespan(app):
        configure_logging()
        try:
            app.state.predictor.load()
        except Exception:
            logging.getLogger(__name__).exception("Production model failed to load; inference unavailable")
        yield
    app = FastAPI(title="Offline Femur/Tibia Segmentation", lifespan=lifespan,
                  docs_url=None, redoc_url=None)
    app.state.config = config
    app.state.predictor = predictor or Predictor(config)
    app.state.store = CaseStore(config)
    app.state.management = Management(config, app.state.predictor)
    app.add_middleware(RequestLimit, maximum=config["limits"]["max_file_bytes"] * 2 + 65536)
    app.add_middleware(CORSMiddleware, allow_origins=config["server"]["cors_origins"],
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
                       expose_headers=["X-Model-Version", "X-Image-Width", "X-Image-Height", "X-Inference-Time-Ms"])
    app.include_router(router)
    app.include_router(management_router)
    return app


app = create_app()
