import cv2
import torch
import numpy as np
import os
import tempfile
import csv
from flask.cli import load_dotenv
from pip._internal.utils import temp_dir

from model import EfficientNetClassifier
from PIL import Image
from torchvision import transforms
import boto3
from botocore.exceptions import NoCredentialsError
from flask import Flask, jsonify, request, send_from_directory, render_template
from dotenv import load_dotenv

app = Flask(__name__, template_folder='.')

# Load environment variables
load_dotenv()
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

# Initialize the S3 client
s3_client = boto3.client('s3',
                         aws_access_key_id=aws_access_key_id,
                         aws_secret_access_key=aws_secret_access_key,
                         region_name='eu-north-1')
BUCKET_NAME = 'deepfakernsit'  # Replace with your actual S3 bucket name

# S3 Prefixes
UPLOADS_S3_PREFIX = 'uploads/'
TEMP_S3_PREFIX = 'temp/'
TEMP_FRAMES_S3_PREFIX = 'temp_frames/'
FEEDBACK_FILE_S3 = 'feedback.csv'

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 3  # Adjust based on your dataset classes
model = EfficientNetClassifier(num_classes=num_classes)
model_path = "best_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to ensure feedback file exists in S3
def ensure_feedback_file():
    try:
        s3_client.head_object(Bucket=BUCKET_NAME, Key=FEEDBACK_FILE_S3)
    except:
        # If the file doesn't exist, create it with the header
        temp_dir = tempfile.mkdtemp()  # Create a temporary directory
        feedback_file_path = os.path.join(temp_dir, 'feedback.csv')
        with open(feedback_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["video_name", "model_prediction", "user_feedback"])
        s3_client.upload_file(feedback_file_path, BUCKET_NAME, FEEDBACK_FILE_S3)

ensure_feedback_file()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    video_file = request.files['video']

    if video_file:
        video_name = video_file.filename
        video_s3_key = os.path.join(UPLOADS_S3_PREFIX, video_name)

        try:
            # Upload file to S3
            s3_client.upload_fileobj(video_file, BUCKET_NAME, video_s3_key)
            result = process_video(video_name)
            return jsonify(result)
        except NoCredentialsError:
            return jsonify({"error": "AWS credentials not available."}), 403
        except Exception as e:
            return jsonify({"error": str(e)}), 500

def save_temp_frame_to_s3(frame, frame_name):
    try:
        # Convert the frame to a byte stream
        _, buffer = cv2.imencode('.jpg', frame)
        byte_data = buffer.tobytes()

        # Upload the frame as a byte stream
        s3_client.put_object(Body=byte_data, Bucket=BUCKET_NAME, Key=os.path.join(TEMP_FRAMES_S3_PREFIX, frame_name))
    except Exception as e:
        print(f"Error saving frame to S3: {e}")

def download_file_from_s3(s3_key, local_filename):
    try:
        s3_client.download_file(BUCKET_NAME, s3_key, local_filename)
    except Exception as e:
        print(f"Error downloading file from S3: {e}")

def process_video(video_name):
    # Temporary download video from S3
    video_s3_key = os.path.join(UPLOADS_S3_PREFIX, video_name)
    temp_dir = tempfile.mkdtemp()  # Create a temporary directory
    local_video_path = os.path.join(temp_dir, video_name)
    download_file_from_s3(video_s3_key, local_video_path)

    cap = cv2.VideoCapture(local_video_path)
    if not cap.isOpened():
        return {"error": "Cannot open the video file."}

    frame_count = 0
    processed_frame_count = 0
    fake_frame_count = 0
    total_faces_detected = 0

    frame_results = []
    frame_paths = []  # New list to store frame paths
    face_frame_paths = []  # New list to store face frame paths

    # Initialize face detector (using a pre-trained model)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 5 != 0:
            continue

        processed_frame_count += 1

        # Detect faces in the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            total_faces_detected += len(faces)
            frame_filename = f"frame_{processed_frame_count:04d}.jpg"
            save_temp_frame_to_s3(frame, frame_filename)
            frame_paths.append(f"/temp/{frame_filename}")

            # Save cropped face frames
            for i, (x, y, w, h) in enumerate(faces):
                face_crop = frame[y:y + h, x:x + w]
                face_filename = f"frame_{processed_frame_count:04d}_face_{i}.jpg"
                save_temp_frame_to_s3(face_crop, face_filename)
                face_frame_paths.append(f"/temp_frames/{face_filename}")

        # Apply model to detect fake faces in frames
        face_crops = []
        for i, (x, y, w, h) in enumerate(faces):
            face_crop = frame[y:y + h, x:x + w]
            face_crops.append(face_crop)

        if face_crops:
            for face_crop in face_crops:
                pil_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                input_tensor = transform(pil_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    _, predicted_class = torch.max(output, 1)
                    if predicted_class.item() == 1:  # Assuming '1' indicates fake face
                        fake_frame_count += 1

    cap.release()

    # Temporal analysis and final decision logic
    fake_video = {
        "fake_decision_by_count": fake_frame_count > total_faces_detected // 2,
        "fake_decision_by_temporal": fake_frame_count > 0,
        "frame_paths": frame_paths,
        "face_frame_paths": face_frame_paths,
        "video_name": video_name
    }

    return fake_video

@app.route('/temp/<path:filename>')
def serve_temp_image(filename):
    return send_from_directory(temp_dir, filename)

@app.route('/temp_frames/<path:filename>')
def serve_temp_frames(filename):
    return send_from_directory(temp_dir, filename)

if __name__ == '__main__':
    app.run(debug=True)
