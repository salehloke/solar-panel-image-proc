import cv2
import time
import numpy as np
from pathlib import Path

class CameraService:
    """
    Service to handle image capture from Pi Camera or USB Webcam.
    """
    def __init__(self):
        self.save_dir = Path("data/captured")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def capture_image(self):
        """
        Captures an image using OpenCV. 
        Note: On Raspberry Pi, ensure the camera is enabled via raspi-config.
        """
        # 0 is usually the default camera (Pi Camera or USB)
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise Exception("Could not open camera device")

        # Allow camera to warm up
        time.sleep(2)
        
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise Exception("Failed to grab frame from camera")

        # Save locally for reference
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        file_path = self.save_dir / f"capture_{timestamp}.jpg"
        cv2.imwrite(str(file_path), frame)

        return frame, str(file_path)
