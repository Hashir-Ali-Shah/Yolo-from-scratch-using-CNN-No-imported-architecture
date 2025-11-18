import requests
import base64
import cv2
import numpy as np

API_URL = "http://localhost:8000/predict/image"
IMAGE_PATH = r"D:\ML\Tasks\CNN\All-weapons-data-1\test\images\test_0002.jpg"


with open(IMAGE_PATH, "rb") as f:
    image_bytes = f.read()


response = requests.post(
    API_URL,
    files={"file": ("test.jpg", image_bytes, "image/jpeg")}
)

# Parse JSON
data = response.json()
image_b64 = data["image"]

# Decode base64 back to image
img_bytes = base64.b64decode(image_b64)
npimg = np.frombuffer(img_bytes, np.uint8)
frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

# Show prediction
cv2.imshow("Result", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
