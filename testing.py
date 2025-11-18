import requests
import base64
import cv2
import numpy as np
import websockets
import asyncio



class YOLOClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url


    def decode_b64_image(self, b64_string):
        img_bytes = base64.b64decode(b64_string)
        npimg = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(npimg, cv2.IMREAD_COLOR)


    def test_video_full(self, video_path):
        url = f"{self.base_url}/predict/video/full"

        with open(video_path, "rb") as f:
            video_bytes = f.read()

        response = requests.post(
            url,
            files={"file": ("video.mp4", video_bytes, "video/mp4")}
        )

        output_path = "processed_video.mp4"
        with open(output_path, "wb") as f:
            f.write(response.content)

        print(f"Processed video saved to {output_path}")

        cap = cv2.VideoCapture(output_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Video Prediction", frame)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    async def test_stream(self, video_path):
        ws_url = self.base_url.replace("http", "ws") + "/ws/video/stream"

        cap = cv2.VideoCapture(video_path)

        async with websockets.connect(ws_url) as ws:
            print("Connected to WebSocket.")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                _, buffer = cv2.imencode(".jpg", frame)
                frame_b64 = base64.b64encode(buffer).decode("utf-8")

                await ws.send(frame_b64)

                processed_b64 = await ws.recv()
                processed_frame = self.decode_b64_image(processed_b64)

                cv2.imshow("Streaming Inference", processed_frame)
                if cv2.waitKey(1) == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":

    client = YOLOClient()

IMAGE_PATH = r"D:\ML\Tasks\CNN\All-weapons-data-1\test\images\test_0002.jpg"
VIDEO_PATH = r"D:\ML\Tasks\CNN\weapons.mp4"

# client.test_image(IMAGE_PATH)

client.test_video_full(VIDEO_PATH)

# asyncio.run(client.test_stream(VIDEO_PATH))