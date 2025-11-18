from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import cv2
import numpy as np
import base64
from inference import YOLOInference 
import os
import tempfile
import uuid
import logging
import time

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("yolo_api.log"),  
        logging.StreamHandler()              
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI(title="YOLO Inference API")

model_path = r"D:\ML\Tasks\CNN\Yolo_results\weapons_train2\weights\last.pt"
yolo_inf = YOLOInference(model_path=model_path, device="cuda", imgsz=640, conf=0.25)

def frame_to_base64(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    logger.info(f"Received image upload: {file.filename}")
    start_time = time.time()
    try:
        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        annotated, result = yolo_inf.process_image(frame=frame)
        img_base64 = frame_to_base64(annotated)
        logger.info(f"Processed image {file.filename} successfully")
        elapsed_time = time.time() - start_time
        logger.info(f"Processing time: {elapsed_time:.2f} seconds")
        return JSONResponse({"image": img_base64})
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(f"Error processing image {file.filename}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/predict/video/full")
async def predict_video(file: UploadFile = File(...)):
    input_path = os.path.join(tempfile.gettempdir(), f"input_{uuid.uuid4().hex}.mp4")
    output_path = os.path.join(tempfile.gettempdir(), f"output_{uuid.uuid4().hex}.mp4")
    logging.info(f"Received video upload: {file.filename}")
    start_time = time.time()
    try:
        contents = await file.read()
        with open(input_path, "wb") as f:
            f.write(contents)
        frames = yolo_inf.get_full_video(input_path)
        if not frames:
            raise HTTPException(status_code=400, detail="No frames found in video")
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()
        logging.info(f"Processed video {file.filename} successfully")
        elapsed_time = time.time() - start_time
        logging.info(f"Processing time: {elapsed_time:.2f} seconds")
        return FileResponse(output_path, media_type="video/mp4", filename="processed_video.mp4")
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.exception(f"Error processing video {file.filename}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

@app.websocket("/ws/video/stream")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    logging.info("WebSocket connection accepted for video streaming")
    start_time = time.time()
    try:
        while True:
            try:
                data = await websocket.receive_text()
                frame_bytes = base64.b64decode(data)
                npimg = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                annotated, _ = yolo_inf.process_image(frame=frame)
                out_base64 = frame_to_base64(annotated)
                await websocket.send_text(out_base64)
            except Exception as e:
                logging.exception("Error processing video frame")
                await websocket.send_text(f"ERROR: {str(e)}")
    except WebSocketDisconnect:
        logging.info("WebSocket connection closed for video streaming")
        elapsed_time = time.time() - start_time
        logging.info(f"Total streaming time: {elapsed_time:.2f} seconds")
    except Exception as e:
        logging.exception("WebSocket error")
        raise HTTPException(status_code=500, detail="WebSocket processing error")

@app.get("/")
def read_root():
    return {"message": "YOLOInference API running"}

@app.post("/predict/image/json")
async def predict_image_json(file: UploadFile = File(...)):
    try:
        logging.info(f"Received image upload for JSON prediction: {file.filename}")
        start_time = time.time()
        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        detections = yolo_inf.process_image_json(frame=frame)
        elapsed_time = time.time() - start_time
        logging.info(f"Processed image {file.filename} successfully in {elapsed_time:.2f}s")
        return JSONResponse({"detections": detections})
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.exception(f"Error processing image {file.filename}")
        raise HTTPException(status_code=500, detail="Internal server error")
