import os.path
import shutil
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

dir = "./temp"
sv = "./videos"
if not os.path.exists(sv):
    os.makedirs(sv)
def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("clicked", x, y)
        for ech_circle in settings:
            dist = utils.distanceCalculate(ech_circle.centre, (x, y))
            if dist <= 10:
                ech_circle.toggle_status()


try:
    while cam.isOpened():
        _,frame = cam.read()

        frame = cv2.resize(frame,(w,h))

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        detection_result = detector.detect(mp_image)

        set_dict = utils.get_settings_dict(settings)
        if set_dict[utils.SKETCH]:
            inp = (mp_image.numpy_view()*0)
            annotated_image = utils.draw_landmarks_on_image(inp, detection_result, settings)
            utils.display_settings(settings, annotated_image,True)
        else:
            annotated_image = utils.draw_landmarks_on_image(mp_image.numpy_view(), detection_result, settings)
            utils.display_settings(settings, annotated_image)

        frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        if set_dict[utils.RECORD]:
            if not os.path.exists(dir):
                os.makedirs(dir)
            new_name = str(len(os.listdir(dir))+1)
            while len(new_name)<7:
                new_name = "0"+new_name
            frame_pth = os.path.join(dir,new_name+".png")
            cv2.imwrite(frame_pth,frame)
        else:
            if os.path.exists(dir):
                vid_dir = os.path.join(sv,str(len(os.listdir(sv))+1)+".mp4")
                video = cv2.VideoWriter(vid_dir, cv2.VideoWriter_fourcc(*'MP4V') , 20, (w, h))
                dir_list = sorted(os.listdir(dir))
                for echframe in dir_list:
                    video.write(cv2.imread(os.path.join(dir,echframe)))
                video.release()
                shutil.rmtree(dir)



        cv2.imshow("myface", frame)

        cv2.setMouseCallback('myface', mouse_click)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            raise Exception()
except:
    print("Progam Terminated")
    cam.release()
    cv2.destroyAllWindows()