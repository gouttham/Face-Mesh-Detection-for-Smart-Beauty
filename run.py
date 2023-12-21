import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import utils




w = 960
h = 540
settings = utils.init_settings(w,h)


base_options = python.BaseOptions(model_asset_path='./face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,num_faces=10)
detector = vision.FaceLandmarker.create_from_options(options)

cam = cv2.VideoCapture(0)


def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("clicked", x, y)
        for ech_circle in settings:
            dist = utils.distanceCalculate(ech_circle.centre, (x, y))
            if dist <= 10:
                ech_circle.toggle_status()



while cam.isOpened():
    _,frame = cam.read()

    frame = cv2.resize(frame,(w,h))

    utils.display_settings(settings,frame)

    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    detection_result = detector.detect(mp_image)
    annotated_image = utils.draw_landmarks_on_image(mp_image.numpy_view(), detection_result, settings)

    frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    cv2.imshow("myface", frame)

    cv2.setMouseCallback('myface', mouse_click)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()