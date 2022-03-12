import json

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# For adding new landmarks based on default predicted landmarks
def add_extra_points(landmark_list):
    left_shoulder = landmark_list[11]
    right_shoulder = landmark_list[12]
    left_hip = landmark_list[23]
    right_hip = landmark_list[24]

    # Calculating hip position and visibility
    hip = {
          'x': (left_hip['x'] + right_hip['x']) / 2.0,
          'y': (left_hip['y'] + right_hip['y']) / 2.0,
          'z': (left_hip['z'] + right_hip['z']) / 2.0,
          'visibility': (left_hip['visibility'] + right_hip['visibility']) / 2.0
        }
    landmark_list.append(hip)

    # Calculating spine position and visibility
    spine = {
          'x': (left_hip['x'] + right_hip['x'] + right_shoulder['x'] + left_shoulder['x']) / 4.0,
          'y': (left_hip['y'] + right_hip['y'] + right_shoulder['y'] + left_shoulder['y']) / 4.0,
          'z': (left_hip['z'] + right_hip['z'] + right_shoulder['z'] + left_shoulder['z']) / 4.0,
          'visibility': (left_hip['visibility'] + right_hip['visibility'] + right_shoulder['visibility'] + left_shoulder['visibility']) / 4.0
        }
    landmark_list.append(spine)

    left_mouth = landmark_list[9]
    right_mouth = landmark_list[10]
    nose = landmark_list[0]
    left_ear = landmark_list[7]
    right_ear = landmark_list[8]
    # Calculating neck position and visibility
    neck = {
          'x': (left_mouth['x'] + right_mouth['x'] + right_shoulder['x'] + left_shoulder['x']) / 4.0,
          'y': (left_mouth['y'] + right_mouth['y'] + right_shoulder['y'] + left_shoulder['y']) / 4.0,
          'z': (left_mouth['z'] + right_mouth['z'] + right_shoulder['z'] + left_shoulder['z']) / 4.0,
          'visibility': (left_mouth['visibility'] + right_mouth['visibility'] + right_shoulder['visibility'] + left_shoulder['visibility']) / 4.0
        }
    landmark_list.append(neck)

    # Calculating head position and visibility
    head = {
          'x': (nose['x'] + left_ear['x'] + right_ear['x']) / 3.0,
          'y': (nose['y'] + left_ear['y'] + right_ear['y']) / 3.0,
          'z': (nose['z'] + left_ear['z'] + right_ear['z']) / 3.0,
          'visibility': (nose['visibility'] + left_ear['visibility'] + right_ear['visibility']) / 3.0,
        }
    landmark_list.append(head)


def world_landmarks_list_to_array(landmark_list, image_shape):
    rows, cols, _ = image_shape
    array = []
    for lmk in landmark_list.landmark:
        new_row = {
          'x': lmk.x * cols,
          'y': lmk.y * rows,
          'z': lmk.z * cols,
          'visibility': lmk.visibility
        }
        array.append(new_row)
    return array
    return np.asarray([(lmk.x * cols, lmk.y * rows, lmk.z * cols,lmk.visibility)
                       for lmk in landmark_list.landmark])


def landmarks_list_to_array(landmark_list):

    array = []
    for lmk in landmark_list.landmark:
        new_row = {
          'x': lmk.x,
          'y': lmk.y,
          'z': lmk.z,
          'visibility': lmk.visibility
        }
        array.append(new_row)
    return array
    return np.asarray([(lmk.x, lmk.y, lmk.z, lmk.visibility)
                       for lmk in landmark_list.landmark])


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose



def Write_Json(path, index, pose_landmarks,rows, cols):
    # print(results.pose_landmarks)
    # print(pose_landmarks)
    # print(results.pose_landmarks)
    # Dump actual JSON.
    json_path = path + "" + str(index) + ".json"
    with open(json_path, 'w') as fl:
        dump_data = {
            'predictions': pose_landmarks,
            'height': rows,
            'width': cols
        }
        # np.around(pose_landmarks, 4).tolist()
        fl.write(json.dumps(dump_data, indent=3 , separators=(',', ': ')))
        fl.close()

def Pose_Images():
    # For static images:
    IMAGE_FILES = ["C:\\Danial\\Projects\\Clone\\3DModelGeneratorTest\\pifuhd\\sample_images\\5.jpg"]
    BG_COLOR = (192, 192, 192) # gray
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0) as pose:
      for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
          continue
        print(
            f'Nose coordinates: ('
            f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
            f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
        )

        annotated_image = image.copy()
        # Draw segmentation on the image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        annotated_image = np.where(condition, annotated_image, bg_image)
        # Draw pose landmarks on the image.
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.imwrite('C:/Danial/Projects/Danial/DigiHuman/Backend/output' + str(idx) + '.png', annotated_image)
        # Plot pose world landmarks.
        mp_drawing.plot_landmarks(
            results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

        new_pose = world_landmarks_list_to_array(
                results.pose_world_landmarks)
        pose_landmarks = landmarks_list_to_array(results.pose_landmarks,
                                                       image.shape)
        print(results.pose_landmarks)
        print(pose_landmarks)

        # Dump actual JSON.
        json_path = "C:/Danial/Projects/Danial/DigiHuman/Backend/json/"
        with open(json_path, 'w') as fl:
          dump_data = {
              'predictions': pose_landmarks,
              'predictions_world': new_pose
          }
          #np.around(pose_landmarks, 4).tolist()
          fl.write(json.dumps(dump_data, indent=2, separators=(',', ': ')))

# For webcam input:
json_path = "C:/Danial/Projects/Danial/DigiHuman/Backend/json/"
cap = cv2.VideoCapture("C:/Danial/Projects/Danial/DigiHuman/Backend/Video/Action_with_wiper.mp4")
index = 0
with mp_pose.Pose(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8) as pose:
  while cap.isOpened():
    success, image = cap.read()
    index += 1
    if not success:
      print("Some probelm with video!")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    #cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))

    try:

        pose_landmarks = landmarks_list_to_array(results.pose_world_landmarks) #also can use results.pose_landmarks
       # world_pose_landmarks = world_landmarks_list_to_array(results.pose_world_landmarks, image.shape)

        #add_extra_points(world_pose_landmarks)
        rows, cols, _ = image.shape
        add_extra_points(pose_landmarks)

        Write_Json(json_path, index, pose_landmarks,rows,cols)
    except:
        print("hi")


    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()