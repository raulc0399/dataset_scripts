import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import numpy as np
import cv2
import json
import os

model_path = '/absolute/path/to/gesture_recognizer.task'

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  # Convert RGB to BGR for MediaPipe drawing utilities
  annotated_image = cv2.cvtColor(np.copy(rgb_image), cv2.COLOR_RGB2BGR)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  # Convert back to RGB before returning
  return cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

def landmarks_to_dict(detection_result):
    """Convert hand landmarks to a serializable dictionary format."""
    result = []
    
    for idx in range(len(detection_result.hand_landmarks)):
        hand_landmarks = detection_result.hand_landmarks[idx]
        handedness = detection_result.handedness[idx]
        
        landmarks_list = []
        for landmark in hand_landmarks:
            landmarks_list.append({
                'x': float(landmark.x),
                'y': float(landmark.y),
                # 'z': float(landmark.z)
            })
        
        hand_data = {
            'type': handedness[0].category_name,
            # 'score': float(handedness[0].score),
            'keypoints': landmarks_list
        }
        
        result.append(hand_data)
    
    return result

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='./models/hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# Load the input image.
image_path = "./test_images/input1.png"
image = mp.Image.create_from_file(image_path)

# Detect hand landmarks from the input image.
detection_result = detector.detect(image)

# Save landmarks to JSON file
landmarks_dict = landmarks_to_dict(detection_result)
json_path = os.path.splitext(image_path)[0] + "_landmarks.json"
with open(json_path, 'w') as f:
    json.dump(landmarks_dict, f, indent=2)

print(f"Landmarks saved to {json_path}")

# Process the classification result. In this case, visualize it and save to JSON.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

cv2.imshow('Annotated Image', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
