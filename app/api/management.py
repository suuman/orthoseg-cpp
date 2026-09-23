"""Opt-in local administrative endpoints, separate from annotation submission."""
import ipaddress
import logging
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict

log = logging.getLogger(__name__)


def local_admin(request: Request):
    if not request.app.state.config["management"]["enabled"]:
        raise HTTPException(403, "Model management is disabled in backend configuration")
    try:
        local = ipaddress.ip_address(request.client.host).is_loopback
        host = urlsplit("http://" + request.headers.get("host", "")).hostname
    except (ValueError, AttributeError):
        local = False
        host = None
    # Native Qt sends no Origin. Prevent browser-driven requests and DNS rebinding.
    if not local or host not in ("127.0.0.1", "localhost", "::1") or "origin" in request.headers:
        raise HTTPException(403, "Model management requires a direct localhost client")


router = APIRouter(dependencies=[Depends(local_admin)], tags=["Local administration"])


class DefaultsOnly(BaseModel):
    model_config = ConfigDict(extra="forbid")


@router.get("/management/status")
def management_status(request: Request):
    return request.app.state.management.summary()


@router.post("/training/start", status_code=202)
def start(request: Request, options: DefaultsOnly):
    try:
        return request.app.state.management.start()
    except BlockingIOError as exc:
        raise HTTPException(409, "Training is already active (API or local script)") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except OSError as exc:
        log.exception("Unable to start administrative training job")
        raise HTTPException(503, "Unable to start training; check backend logs") from exc


@router.get("/training/jobs/{job_id}")
def job(request: Request, job_id: str):
    try:
        return request.app.state.management.job(job_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(404, "Unknown training job") from exc


@router.post("/models/{version}/promote")
def promote(request: Request, version: str, options: DefaultsOnly):
    try:
        return request.app.state.management.promote(version)
    except BlockingIOError as exc:
        raise HTTPException(409, "Wait for active training to finish before promotion") from exc
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        log.exception("Candidate promotion failed")
        raise HTTPException(500, "Candidate promotion failed; check backend logs") from exc
