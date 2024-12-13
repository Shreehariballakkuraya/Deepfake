import cv2
import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from model import EfficientNetClassifier  # Import your model class

# Paths
input_video_path = "test.mp4"  # Path to the input video
output_video_frames_dir = "temp"  # Directory to save extracted frames
output_cropped_frames_dir = "temp_frames"  # Directory to save cropped face frames
model_path = "best_model.pth"  # Path to the saved model

# Extract video name without extension for folder naming
video_name = os.path.splitext(os.path.basename(input_video_path))[0]

# Create output directories for the video
video_frames_dir = os.path.join(output_video_frames_dir, video_name)
cropped_frames_dir = os.path.join(output_cropped_frames_dir, video_name)
os.makedirs(video_frames_dir, exist_ok=True)
os.makedirs(cropped_frames_dir, exist_ok=True)

# Load pre-trained DNN face detector
dnn_model_path = r"C:/Users/naras/PycharmProjects/DeepFake_Detect/models/deploy.prototxt"
dnn_weights_path = r"C:/Users/naras/PycharmProjects/DeepFake_Detect/models/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(dnn_model_path, dnn_weights_path)

# Initialize video capture
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print("Error: Cannot open the video file.")
    exit()

# Get the frame rate of the video
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_interval = max(1, fps // 5)  # Process every 5th frame

frame_count = 0
processed_frame_count = 0

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 3  # Adjust based on your dataset classes
model = EfficientNetClassifier(num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

fake_frame_count = 0
total_faces_detected = 0

# To store frame analysis results
frame_results = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Process frames based on the interval
    if frame_count % frame_interval != 0:
        continue

    processed_frame_count += 1

    # Save the current frame to the corresponding video folder
    frame_filename = os.path.join(video_frames_dir, f"frame_{processed_frame_count:04d}.jpg")
    cv2.imwrite(frame_filename, frame)

    # Prepare the frame for DNN detection
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Detect faces and crop
    frame_is_fake = False
    frame_fake_scores = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")

            # Ensure bounding box is within the frame dimensions
            x, y = max(0, x), max(0, y)
            x2, y2 = min(w, x2), min(h, y2)

            cropped_face = frame[y:y2, x:x2]
            cropped_face_filename = os.path.join(cropped_frames_dir, f"frame_{processed_frame_count:04d}_face_{i + 1}.jpg")
            cv2.imwrite(cropped_face_filename, cropped_face)

            # Predict whether the face is real or fake
            total_faces_detected += 1
            face_image = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
            input_tensor = transform(face_image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                prediction = torch.argmax(output, dim=1).item()
                fake_score = torch.softmax(output, dim=1)[0, 1].item()  # Probability of "fake"

            if prediction == 1:  # Assuming label 1 corresponds to "fake"
                fake_frame_count += 1
                frame_is_fake = True

            frame_fake_scores.append(fake_score)
            print(f"Frame {processed_frame_count}, Face {i + 1}: {'Fake' if prediction == 1 else 'Real'}, Score: {fake_score:.2f}")

    # Append result for the frame
    frame_results.append((processed_frame_count, frame_is_fake, np.mean(frame_fake_scores) if frame_fake_scores else 0.0))

cap.release()

# Analyze temporal consistency
consistent_fake_frames = 0
temporal_fake_scores = []
for i in range(1, len(frame_results)):
    temporal_fake_scores.append(frame_results[i][2])
    if frame_results[i][1] == frame_results[i - 1][1]:
        consistent_fake_frames += 1

# Compute temporal trends
average_fake_score = np.mean(temporal_fake_scores) if temporal_fake_scores else 0.0

# Final video-level decision using both factors
temporal_threshold = 0.6  # Threshold for average fake score
temporal_consistency_threshold = 0.7  # Threshold for consistent fake frames

fake_decision_by_count = fake_frame_count > total_faces_detected // 2
fake_decision_by_temporal = average_fake_score > temporal_threshold and consistent_fake_frames / max(1, (len(frame_results) - 1)) > temporal_consistency_threshold

if fake_decision_by_count or fake_decision_by_temporal:
    print("The video is classified as FAKE.")
else:
    print("The video is classified as REAL.")

print(f"Consistent frames classified as fake: {consistent_fake_frames} out of {len(frame_results) - 1}")
print(f"Average fake score over the video: {average_fake_score:.2f}")
print("Video processing and temporal analysis complete.")
