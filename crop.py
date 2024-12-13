import cv2
import os
import numpy as np

# Update paths as needed
base_video_dir = r"D:/ml project/pythonProject1/data"
output_frame_dir = r"D:/ml project/pythonProject1/frames_cropped"
categories = ['celeb-real', 'celeb-synthesis', 'youtube-real']
splits = ['train', 'val', 'test']

# Ensure the output directory exists
if not os.path.exists(output_frame_dir):
    os.makedirs(output_frame_dir)

# Load pre-trained DNN model for face detection
net = cv2.dnn.readNetFromCaffe(
    r'D:/ml project/pythonProject1/deploy.prototxt',
    r'D:/ml project/pythonProject1/res10_300x300_ssd_iter_140000.caffemodel'
)


# Function to detect faces using DNN
def detect_face_dnn(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Use confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x2, y2) = box.astype("int")
            faces.append((x, y, x2 - x, y2 - y))

    return faces


# Process frames for a single video
def process_video(video_path, output_dir, skip_frames):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    cropped_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip every Nth frame
        if frame_count % skip_frames != 0:
            frame_count += 1
            continue

        # Detect faces
        faces = detect_face_dnn(frame)
        for (x, y, w, h) in faces:
            cropped_face = frame[y:y + h, x:x + w]
            frame_filename = os.path.join(output_dir, f"frame{cropped_frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, cropped_face)
            cropped_frame_count += 1

        frame_count += 1

    cap.release()
    print(f"Processed {cropped_frame_count} cropped frames from video {os.path.basename(video_path)}.")


# Iterate through all categories and splits
skip_frames = 5  # Skip every 5th frame

for category in categories:
    for split in splits:
        video_dir = os.path.join(base_video_dir, category, split)
        output_dir = os.path.join(output_frame_dir, category, split)
        os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(video_dir):
            for video_file in os.listdir(video_dir):
                video_path = os.path.join(video_dir, video_file)
                video_output_dir = os.path.join(output_dir, os.path.splitext(video_file)[0])
                os.makedirs(video_output_dir, exist_ok=True)

                process_video(video_path, video_output_dir, skip_frames)

print("Face cropping with frame skipping completed.")