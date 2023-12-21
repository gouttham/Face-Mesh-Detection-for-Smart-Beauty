import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2


MESH = 'Mesh'
IRIS = 'Iris'
EYES = 'Eyes'
EYE_BROWS = 'Eyebrows'
LIPS = 'Lips'
OUTLINE = 'Outline'
SKETCH = 'Sketch'
RECORD = 'Record'

def get_settings_dict(settings):
    set_dict = {}
    for ech_setting in settings:
        set_dict[ech_setting.name] = ech_setting.status
    return set_dict

def draw_landmarks_on_image(rgb_image, detection_result,settings):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    set_dict = get_settings_dict(settings)
    # set_dict = {}
    # for ech_setting in settings:
    #     set_dict[ech_setting.name] = ech_setting.status

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks])

        if set_dict[MESH]:
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
        if set_dict[IRIS]:
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())
        if set_dict[EYES]:
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_LEFT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())

            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_RIGHT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
        if set_dict[EYE_BROWS]:
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_LEFT_EYEBROW,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())

            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_RIGHT_EYEBROW,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
        if set_dict[LIPS]:
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_LIPS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
        if set_dict[OUTLINE]:
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_FACE_OVAL,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())



    return annotated_image


def distanceCalculate(p1, p2):
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return dis

class circles:
    def __init__(self,name,centre,status):
        self.name = name
        self.centre = centre
        self.status = status
    def toggle_status(self):
        self.status = not self.status


def display_settings(circle_list,frame,invert=False):
    radius = 7

    clr = (0,0,0) if not invert else (255,255,255)
    for ech_circle in circle_list:
        cv2.circle(frame, ech_circle.centre, radius, clr, 2)

        if ech_circle.status:
            if ech_circle.name == RECORD:
                cv2.circle(frame, ech_circle.centre, 5, (255, 0, 0), -1)
            else:
                cv2.circle(frame, ech_circle.centre, 5, (0, 255, 0), -1)
        w,h = ech_circle.centre
        w = w+radius
        h = h+radius
        cv2.putText(frame, " "+ech_circle.name, (w,h),cv2.FONT_HERSHEY_SIMPLEX ,0.75,clr,2)

    cv2.putText(frame, "Press q to exit" , (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, clr, 2)

    return frame

def init_settings(w, h):
    init_width = (w - int(w / 5))
    init_height = int(h / 10)
    space = 30
    settings = [
        circles(MESH, (init_width, init_height),True),
        circles(IRIS, (init_width, init_height + space),True),
        circles(EYES, (init_width, init_height + space * 2),True),
        circles(EYE_BROWS, (init_width, init_height + space * 3),True),
        circles(LIPS, (init_width, init_height + space * 4),True),
        circles(OUTLINE, (init_width, init_height + space * 5),True),
        circles(SKETCH, (init_width, init_height + space * 6),False),
        circles(RECORD, (init_width, init_height + space * 7),False),

    ]
    return settings