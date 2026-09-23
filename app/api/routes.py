import logging
import json

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from app.ml.predictor import ModelUnavailable
from app.ml.preprocessing import ImageTooLarge, InvalidImage, decode_png, encode_mask

router = APIRouter()
log = logging.getLogger(__name__)


async def read_file(file, config):
    data = await file.read(config["limits"]["max_file_bytes"] + 1)
    if len(data) > config["limits"]["max_file_bytes"]:
        raise HTTPException(413, "File exceeds maximum size")
    return data


@router.get("/health")
def health(request: Request):
    predictor = request.app.state.predictor
    info = predictor.info()
    import torch
    return {"status": "ok", "device": predictor.device.type,
            "gpu": torch.cuda.get_device_name() if predictor.device.type == "cuda" else None,
            **{k: info[k] for k in ("model_loaded", "model_version", "labels")},
            "sam2_model_loaded": request.app.state.sam2.info()["model_loaded"],
            "sam2_model_version": request.app.state.sam2.info()["model_version"],
            "nnunet_model_loaded": request.app.state.nnunet.info()["model_loaded"],
            "nnunet_model_version": request.app.state.nnunet.info()["model_version"]}


@router.get("/model/info")
def model_info(request: Request):
    return {**request.app.state.predictor.info(), "prompted_model": request.app.state.sam2.info(),
            "nnunet_model": request.app.state.nnunet.info()}


@router.post("/segment")
async def segment(request: Request, image: UploadFile = File(...),
                  case_id: str | None = Form(None, max_length=128),
                  filename: str | None = Form(None, max_length=255)):
    data = await read_file(image, request.app.state.config)
    def run():
        arr = decode_png(data, request.app.state.config["limits"])
        mask, version, elapsed = request.app.state.predictor.predict(arr)
        if mask.shape != arr.shape:
            raise RuntimeError("Prediction dimensions do not match original image")
        try:
            encoded = encode_mask(mask)
        except InvalidImage as exc:
            raise RuntimeError("Model produced an invalid class-index mask") from exc
        return Response(encoded, media_type="image/png", headers={
            "X-Model-Version": version, "X-Image-Width": str(arr.shape[1]),
            "X-Image-Height": str(arr.shape[0]), "X-Inference-Time-Ms": f"{elapsed:.3f}"})
    try:
        return await run_in_threadpool(run)
    except ImageTooLarge as exc:
        raise HTTPException(413, str(exc)) from exc
    except InvalidImage as exc:
        raise HTTPException(422, str(exc)) from exc
    except ModelUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        log.exception("Inference failed")
        raise HTTPException(500, "Model inference failed; see local server logs") from exc


@router.post("/segment/nnunet")
async def segment_nnunet(request: Request, image: UploadFile = File(...),
                         case_id: str | None = Form(None, max_length=128)):
    data = await read_file(image, request.app.state.config)
    def run():
        arr = decode_png(data, request.app.state.config["limits"])
        mask, version, elapsed = request.app.state.nnunet.predict(arr)
        if mask.shape != arr.shape:
            raise RuntimeError("nnUNet prediction dimensions do not match original image")
        try:
            encoded = encode_mask(mask)
        except InvalidImage as exc:
            raise RuntimeError("nnUNet produced an invalid class-index mask") from exc
        return Response(encoded, media_type="image/png", headers={
            "X-Model-Version": version, "X-Image-Width": str(arr.shape[1]),
            "X-Image-Height": str(arr.shape[0]), "X-Inference-Time-Ms": f"{elapsed:.3f}"})
    try:
        return await run_in_threadpool(run)
    except ImageTooLarge as exc:
        raise HTTPException(413, str(exc)) from exc
    except InvalidImage as exc:
        raise HTTPException(422, str(exc)) from exc
    except ModelUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        log.exception("nnUNet v2 inference failed")
        raise HTTPException(500, "nnUNet v2 inference failed; see local server logs") from exc


@router.post("/segment/prompted")
async def segment_prompted(request: Request, image: UploadFile = File(...),
                           femur_box: str | None = Form(None, max_length=100),
                           tibia_box: str | None = Form(None, max_length=100)):
    data = await read_file(image, request.app.state.config)
    def run():
        arr = decode_png(data, request.app.state.config["limits"])
        boxes = {}
        for label, raw in ((1, femur_box), (2, tibia_box)):
            if raw is not None:
                try:
                    boxes[label] = json.loads(raw)
                except ValueError as exc:
                    raise InvalidImage("SAM2 boxes must be JSON arrays [x0,y0,x1,y1]") from exc
        mask, version, elapsed = request.app.state.sam2.predict(arr, boxes)
        if mask.shape != arr.shape:
            raise RuntimeError("SAM2 prediction dimensions do not match original image")
        return Response(encode_mask(mask), media_type="image/png", headers={
            "X-Model-Version": version, "X-Image-Width": str(arr.shape[1]),
            "X-Image-Height": str(arr.shape[0]), "X-Inference-Time-Ms": f"{elapsed:.3f}"})
    try:
        return await run_in_threadpool(run)
    except ImageTooLarge as exc:
        raise HTTPException(413, str(exc)) from exc
    except InvalidImage as exc:
        raise HTTPException(422, str(exc)) from exc
    except ModelUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        log.exception("Prompted SAM2 inference failed")
        raise HTTPException(500, "Prompted SAM2 inference failed; see local server logs") from exc


@router.post("/training/cases")
async def submit(request: Request, image: UploadFile = File(...), mask: UploadFile = File(...),
                 case_id: str | None = Form(None, max_length=128),
                 original_filename: str | None = Form(None, max_length=255),
                 model_version: str | None = Form(None, max_length=128),
                 annotator: str | None = Form(None, max_length=256),
                 notes: str | None = Form(None, max_length=8192)):
    config = request.app.state.config
    image_data, mask_data = await read_file(image, config), await read_file(mask, config)
    try:
        return await run_in_threadpool(request.app.state.store.submit, image_data, mask_data,
            case_id=case_id, original_filename=original_filename, model_version=model_version,
            annotator=annotator, notes=notes)
    except ImageTooLarge as exc:
        raise HTTPException(413, str(exc)) from exc
    except InvalidImage as exc:
        raise HTTPException(422, str(exc)) from exc
    except OSError as exc:
        log.exception("Case storage failed")
        raise HTTPException(507, "Unable to persist case; see local server logs") from exc


@router.get("/training/status")
def training_status(request: Request):
    return {**request.app.state.store.status(),
            "production_model_version": request.app.state.predictor.info()["model_version"]}
