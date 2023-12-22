import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import math
from typing import Tuple, Union


MESH = 'Mesh'
IRIS = 'Iris'
EYES = 'Eyes'
EYE_BROWS = 'Eyebrows'
LIPS = 'Lips'
OUTLINE = 'Outline'
SKETCH = 'Sketch'
IRIS_D = 'Iris_dist'
RECORD = 'Record'

def get_settings_dict(settings):
    set_dict = {}
    for ech_setting in settings:
        set_dict[ech_setting.name] = ech_setting.status
    return set_dict


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

def get_coods(connections,idx_to_coordinates):
    iris_coords = []
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
            pts = idx_to_coordinates[start_idx]
            iris_coords.append(pts)
    return iris_coords
def blink_detection(rgb_image, detection_result):
    try:
        face_landmarks_list = detection_result.face_landmarks
        image_rows, image_cols, _ = rgb_image.shape
        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks])



            idx_to_coordinates = {}
            for idx, landmark in enumerate(face_landmarks_proto.landmark):
                if ((landmark.HasField('visibility') and landmark.visibility < 0.5) or (landmark.HasField('presence') and landmark.presence < 0.5)):
                    continue
                landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                               image_cols, image_rows)
                if landmark_px:
                    idx_to_coordinates[idx] = landmark_px

            left_connections = frozenset([(476, 477),(474, 475)])
            ll,lr = get_coods(left_connections,idx_to_coordinates)

            right_connections = frozenset([(471, 472), (469, 470)])
            rl, rr = get_coods(right_connections, idx_to_coordinates)

            left_eye_mid = (int((ll[0] + lr[0])/2),int((ll[1] + lr[1])/2))
            right_eye_mid = (int((rl[0] + rr[0]) / 2), int((rl[1] + rr[1]) / 2))

            line_mid = ( int((left_eye_mid[0]+right_eye_mid[0])/2)-20,int((left_eye_mid[1]+right_eye_mid[1])/2)-20 )

            iris_dist = int(distanceCalculate(left_eye_mid,right_eye_mid))

            cv2.circle(rgb_image, left_eye_mid, 2, (255, 0, 0), 2)
            cv2.circle(rgb_image, right_eye_mid, 2, (255, 0, 0), 2)

            cv2.line(rgb_image, left_eye_mid,right_eye_mid, (0,255,0),2)
            cv2.putText(rgb_image, str(iris_dist), line_mid, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

    except:
        print("")
    return rgb_image


def draw_landmarks_on_image(rgb_image, detection_result,settings):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    set_dict = get_settings_dict(settings)

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
        circles(IRIS_D, (init_width, init_height + space * 6), True),
        circles(SKETCH, (init_width, init_height + space * 7),False),
        circles(RECORD, (init_width, init_height + space * 8),False),

    ]
    return settings