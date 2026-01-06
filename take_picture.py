import pyzed.sl as sl
import cv2
import os
from datetime import datetime

OUTPUT_DIR = "img"

zed = sl.Camera()

init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 30
init_params.depth_mode = sl.DEPTH_MODE.NEURAL_LIGHT

status = zed.open(init_params)

if status != sl.ERROR_CODE.SUCCESS:
    print(f"Opening ZED failed: {status}")

runtime_params = sl.RuntimeParameters()
image = sl.Mat()

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
print(timestamp)

try:
    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)

        img = image.get_data()
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        path = os.path.join(OUTPUT_DIR, f"left_{timestamp}.png") 
        cv2.imwrite(path, bgr)
        
        print(f"Saved image to {path}")

finally:
    zed.close()
    print("Done.")