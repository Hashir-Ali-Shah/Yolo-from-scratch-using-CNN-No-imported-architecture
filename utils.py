import cv2
import numpy as np

def letterbox(image, new_size=(640, 640), color=(114, 114, 114)):
    """
    Resize image to fit into new_size while keeping aspect ratio, pad remaining space.
    """
    h, w = image.shape[:2]
    new_w, new_h = new_size

    scale = min(new_w / w, new_h / h)
    resized_w, resized_h = int(w * scale), int(h * scale)

    resized_image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((new_h, new_w, 3), color, dtype=np.uint8)

    top = (new_h - resized_h) // 2
    left = (new_w - resized_w) // 2

    canvas[top:top+resized_h, left:left+resized_w] = resized_image

    return canvas


def read_image(image_path):
    """
    Read an image from a file path.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return image

def show_image(image):
    """
    Display an image in a window.
    """
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()