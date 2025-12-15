import pyzed.sl as sl
import cv2

zed = sl.Camera()

init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 30
init_params.depth_mode = sl.DEPTH_MODE.NEURAL_LIGHT
init_params.coordinate_units = sl.UNIT.METER
init_params.sdk_verbose = 0

status = zed.open(init_params)

if status != sl.ERROR_CODE.SUCCESS:
    print(f"ZED open failed: {status}")

runtime_params = sl.RuntimeParameters()

image = sl.Mat()
window_name = "ZED 2 - Live (Left RGB)"

try:
    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)

            frame = image.get_data()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            cv2.imshow(window_name, frame_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

finally:
    cv2.destroyAllWindows()
    zed.close() 