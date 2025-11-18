from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
from inference import YOLOInference 

app = FastAPI(title="YOLO Inference API")


model_path = r"D:\ML\Tasks\CNN\Yolo_results\weapons_train2\weights\last.pt"
yolo_inf = YOLOInference(model_path=model_path, device="cuda", imgsz=640, conf=0.25)


def frame_to_base64(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    annotated, result = yolo_inf.process_image(frame=frame)
    img_base64 = frame_to_base64(annotated)

    return JSONResponse({"image": img_base64})


from fastapi.responses import FileResponse
import os

@app.post("/predict/video/full")
async def predict_video(file: UploadFile = File(...)):
    contents = await file.read()
    input_path = "temp_input_video.mp4"
    with open(input_path, "wb") as f:
        f.write(contents)

    frames = yolo_inf.get_full_video(input_path)

    output_path = "processed_video.mp4"

    height, width, _ = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))  

    for frame in frames:
        out.write(frame)
    out.release()

    os.remove(input_path)

    return FileResponse(output_path, media_type="video/mp4", filename="processed_video.mp4")


@app.websocket("/ws/video/stream")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            frame_bytes = base64.b64decode(data)
            npimg = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            annotated, _ = yolo_inf.process_image(frame=frame)

            out_base64 = frame_to_base64(annotated)
            await websocket.send_text(out_base64)

    except WebSocketDisconnect:
        print("Client disconnected")


@app.get("/")
def read_root():
    return {"message": "YOLOInference API running"}


@app.post("/predict/image/json")
async def predict_image_json(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    detections = yolo_inf.process_image_json(frame=frame)

    return JSONResponse({"detections": detections})