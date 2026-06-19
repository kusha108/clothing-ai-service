import cv2
import numpy as np

def get_skin_region(image):
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Skin color range (HSV)
    lower = np.array([0, 48, 80], dtype=np.uint8)
    upper = np.array([20, 255, 255], dtype=np.uint8)

    # Mask skin
    mask = cv2.inRange(hsv, lower, upper)

    # Clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Blur mask
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    # Apply mask
    skin = cv2.bitwise_and(image, image, mask=mask)

    return skin


def get_skin_tone(image):
    skin = get_skin_region(image)

    # Get only non-zero pixels
    pixels = skin.reshape(-1, 3)
    pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]

    if len(pixels) == 0:
        return "medium"  # fallback

    # Average color
    avg_color = np.mean(pixels, axis=0)

    # Convert BGR → brightness
    brightness = np.mean(avg_color)

    if brightness > 180:
        return "light"
    elif brightness > 120:
        return "medium"
    else:
        return "dark"