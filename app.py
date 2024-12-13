import os
import csv
from flask import Flask, render_template, request, send_from_directory, jsonify
import cv2
import torch
from torchvision import transforms
from PIL import Image
from model import EfficientNetClassifier  # Import your model class
import numpy as np
import matplotlib.pyplot as plt
app = Flask(__name__)

# Paths
output_video_frames_dir = "temp"  # Directory to save extracted frames
output_cropped_frames_dir = "temp_frames"  # Directory to save cropped face frames
model_path = "best_model.pth"  # Path to the saved model
feedback_file = "feedback.csv"  # Path to store feedback

# Load pre-trained DNN face detector
dnn_model_path = r"C:/Users/naras/PycharmProjects/DeepFake_Detect/models/deploy.prototxt"
dnn_weights_path = r"C:/Users/naras/PycharmProjects/DeepFake_Detect/models/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(dnn_model_path, dnn_weights_path)

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

# Ensure feedback file exists
if not os.path.exists(feedback_file):
    with open(feedback_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["video_name", "model_prediction", "user_feedback"])  # Header


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    # Get the uploaded video file
    video_file = request.files['video']

    if video_file:
        # Save the video file to the 'uploads' folder
        video_path = os.path.join('uploads', video_file.filename)
        video_file.save(video_path)

        # Extract the video name (without extension)
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # Process the video
        result = process_video(video_path, video_name)

        # Convert numpy types to Python types for JSON serialization
        result = convert_numpy_types(result)

        # Return the result as a JSON response
        return jsonify(result)


@app.route('/feedback', methods=['POST'])
def feedback():
    # Get the feedback data from the user
    data = request.json
    video_name = data.get('video_name')
    model_prediction = data.get('model_prediction')
    user_feedback = data.get('user_feedback')

    # Append feedback to the CSV file
    with open(feedback_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([video_name, model_prediction, user_feedback])

    return jsonify({"status": "success", "message": "Feedback saved!"})


def convert_numpy_types(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert ndarray to a list
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj  # Return other types unchanged


def process_video(input_video_path, video_name):
    # Create output directories for the video
    video_frames_dir = os.path.join(output_video_frames_dir, video_name)
    cropped_frames_dir = os.path.join(output_cropped_frames_dir, video_name)
    os.makedirs(video_frames_dir, exist_ok=True)
    os.makedirs(cropped_frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        return {"error": "Cannot open the video file."}

    # Get the frame rate of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, fps // 5)  # Process every 5th frame

    frame_count = 0
    processed_frame_count = 0
    fake_frame_count = 0
    total_faces_detected = 0

    frame_results = []
    frame_paths = []  # New list to store frame paths
    face_frame_paths = []  # New list to store face frame paths

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        processed_frame_count += 1

        # Save the current frame to the corresponding video folder
        frame_filename = os.path.join(video_frames_dir, f"frame_{processed_frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_paths.append(f"/temp/{video_name}/frame_{processed_frame_count:04d}.jpg")

        # Prepare the frame for DNN detection
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        # Detect faces and crop
        frame_is_fake = False
        frame_fake_scores = []
        frame_faces = []  # New list to store face frame paths for this frame
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")
                x, y = max(0, x), max(0, y)
                x2, y2 = min(w, x2), min(h, y2)

                cropped_face = frame[y:y2, x:x2]
                cropped_face_filename = os.path.join(cropped_frames_dir,
                                                     f"frame_{processed_frame_count:04d}_face_{i + 1}.jpg")
                cv2.imwrite(cropped_face_filename, cropped_face)
                frame_faces.append(f"/temp_frames/{video_name}/frame_{processed_frame_count:04d}_face_{i + 1}.jpg")

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

        frame_results.append(
            (processed_frame_count, frame_is_fake, np.mean(frame_fake_scores) if frame_fake_scores else 0.0))
        face_frame_paths.append(frame_faces)

    cap.release()

    # Temporal analysis
    consistent_fake_frames = 0
    temporal_fake_scores = []
    for i in range(1, len(frame_results)):
        temporal_fake_scores.append(frame_results[i][2])
        if frame_results[i][1] == frame_results[i - 1][1]:
            consistent_fake_frames += 1

    average_fake_score = np.mean(temporal_fake_scores) if temporal_fake_scores else 0.0

    fake_decision_by_count = fake_frame_count > total_faces_detected // 2
    fake_decision_by_temporal = average_fake_score > 0.6 and consistent_fake_frames / max(1, (
                len(frame_results) - 1)) > 0.7

    result = {
        "fake_decision_by_count": fake_decision_by_count,
        "fake_decision_by_temporal": fake_decision_by_temporal,
        "consistent_fake_frames": consistent_fake_frames,
        "average_fake_score": average_fake_score,
        "frame_paths": frame_paths,
        "face_frame_paths": face_frame_paths,
        "video_name": video_name
    }

    return result


@app.route('/temp/<path:filename>')
def serve_temp_image(filename):
    return send_from_directory(output_video_frames_dir, filename)


@app.route('/temp_frames/<path:filename>')
def serve_temp_frames(filename):
    return send_from_directory(output_cropped_frames_dir, filename)


if __name__ == '__main__':
    app.run(debug=True)
