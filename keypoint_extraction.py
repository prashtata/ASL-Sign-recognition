###################################################################################
# Credit: https://www.kaggle.com/code/dataguy1234567899/sign-language
###################################################################################


import cv2
import mediapipe as mp
import albumentations as A
import os
import numpy as np
import pandas as pd
import json

# Define augmentations
transform = A.Compose([
    A.HorizontalFlip(p=1),
    A.Rotate(limit=20, p=1),
    A.GaussianBlur(p=0.2)
])

# Apply to a video frame
def augment_frame(frame):
    augmented = transform(image=frame)
    return augmented["image"]


# Mediapipe model and utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

def extract_keypoints(results):
    
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    print(pose.shape)
    # face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    print(lh.shape)
    print(rh.shape)
    return np.concatenate([pose, lh, rh])  

#Load the metadata from the JSON file
metadata = {}
with open('/home/prashtata/gradschool/asl/dataset/WLASL_v0.3.json', 'r') as file:
    metadata = json.load(file)

#Extract label maps from the metadata
labelMap = {}
for i in metadata:
    label = i['gloss']
    for instance in i['instances']:
        video_id = int(instance['video_id'])
        frame_start = instance['frame_start']
        frame_end = instance['frame_end']
        fps = instance['fps']
        labelMap[video_id] = [label, frame_start, frame_end, fps]


DATA_PATH = '/home/prashtata/gradschool/asl/dataset/MP_data'
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

# Process the videos
sample_number = 0
video_path = '/home/prashtata/gradschool/asl/dataset/videos'
for video in os.listdir(video_path)[:6486]:
    if video.endswith('.mp4'):
        video_filename = os.path.basename(video)
        video_id = int(os.path.splitext(video_filename)[0])

        # Assign video metadata from the datamap
        label, start_frame, end_frame, fps = labelMap[video_id]
        
        # Load Video
        cap = cv2.VideoCapture(os.path.join(video_path, video))
        cap.set(cv2.CAP_PROP_FPS, fps)

        # holistic pose estimation from the video
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            # Create directories for each gloss
            action_path = os.path.join(DATA_PATH, f'{label}_aug')
            if not os.path.exists(action_path):
                os.makedirs(action_path)

            # Create a subdirectory within the given gloss directory for each video
            video_dir = os.path.join(action_path, str(video_id))
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
                sample_number += 1
            else: continue

            frame_count = 0
            keypoints_data = []

            # Process frames
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break
                frame_count += 1

                # Frame number Error Handling
                if frame_count < start_frame or (end_frame != -1 and frame_count > end_frame):
                    continue

                # Extract landmarks
                image = augment_frame(image)
                image, results = mediapipe_detection(image, holistic)
                keypoints = extract_keypoints(results)
                keypoints_data.append(keypoints)

            print(f"Processing video #{sample_number}")
            # Dump in a numpy array
            # np.save(os.path.join(video_dir, f'{video_id}_aug_keypoints.npy'), np.array(keypoints_data))

        # End video reading
        cap.release()

print(f"Successfully processed dataset. Total {sample_number} videos processed")