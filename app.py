import os
import cv2
import torch
import tempfile
import numpy as np
import csv
from flask import Flask, jsonify, request, render_template, send_from_directory
from torchvision import transforms
from PIL import Image
import boto3
from botocore.exceptions import NoCredentialsError


# AWS Credentials and S3 Config
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
s3_client = boto3.client('s3',
                         aws_access_key_id=aws_access_key_id,
                         aws_secret_access_key=aws_secret_access_key,
                         region_name='eu-north-1')
BUCKET_NAME = 'deepfakernsit'
UPLOADS_S3_PREFIX = 'uploads/'
TEMP_S3_PREFIX = 'temp/'
TEMP_FRAMES_S3_PREFIX = 'temp_frames/'
FEEDBACK_FILE_S3 = 'feedback.csv'

# Flask app
app = Flask(__name__, template_folder='.')

# Model loading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 3  # Update according to your dataset
model_path = "best_model.pth"

# Ensure the model file is downloaded from S3
if not os.path.exists(model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    try:
        s3_client.download_file(BUCKET_NAME, "best_model.pth", model_path)
        print("Model downloaded successfully.")
    except NoCredentialsError:
        print("AWS credentials not available.")
    except Exception as e:
        print(f"Error downloading model: {e}")

# Load the model
from model import EfficientNetClassifier
model = EfficientNetClassifier(num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Ensure feedback file exists in S3
def ensure_feedback_file():
    try:
        s3_client.head_object(Bucket=BUCKET_NAME, Key=FEEDBACK_FILE_S3)
    except:
        temp_dir = tempfile.mkdtemp()
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
            # Upload video to S3
            s3_client.upload_fileobj(video_file, BUCKET_NAME, video_s3_key)
            result = process_video(video_name)
            return jsonify(result)
        except NoCredentialsError:
            return jsonify({"error": "AWS credentials not available."}), 403
        except Exception as e:
            return jsonify({"error": str(e)}), 500

def download_file_from_s3(s3_key, local_filename):
    try:
        s3_client.download_file(BUCKET_NAME, s3_key, local_filename)
    except Exception as e:
        print(f"Error downloading file from S3: {e}")

def save_temp_frame_to_s3(frame, frame_name):
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        byte_data = buffer.tobytes()
        s3_client.put_object(Body=byte_data, Bucket=BUCKET_NAME, Key=os.path.join(TEMP_FRAMES_S3_PREFIX, frame_name))
    except Exception as e:
        print(f"Error saving frame to S3: {e}")

def process_video(video_name):
    video_s3_key = os.path.join(UPLOADS_S3_PREFIX, video_name)
    temp_dir = tempfile.mkdtemp()
    local_video_path = os.path.join(temp_dir, video_name)
    download_file_from_s3(video_s3_key, local_video_path)

    cap = cv2.VideoCapture(local_video_path)
    if not cap.isOpened():
        return {"error": "Cannot open the video file."}

    frame_count, processed_frame_count, fake_frame_count, total_faces_detected = 0, 0, 0, 0
    frame_paths, face_frame_paths = [], []

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 5 != 0:
            continue

        processed_frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            total_faces_detected += len(faces)
            frame_filename = f"frame_{processed_frame_count:04d}.jpg"
            save_temp_frame_to_s3(frame, frame_filename)
            frame_paths.append(f"/temp/{frame_filename}")

            for i, (x, y, w, h) in enumerate(faces):
                face_crop = frame[y:y + h, x:x + w]
                face_filename = f"frame_{processed_frame_count:04d}_face_{i}.jpg"
                save_temp_frame_to_s3(face_crop, face_filename)
                face_frame_paths.append(f"/temp_frames/{face_filename}")

                pil_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                input_tensor = transform(pil_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(input_tensor)
                    _, predicted_class = torch.max(output, 1)
                    if predicted_class.item() == 1:
                        fake_frame_count += 1

    cap.release()

    return {
        "fake_decision_by_count": fake_frame_count > total_faces_detected // 2,
        "fake_decision_by_temporal": fake_frame_count > 0,
        "frame_paths": frame_paths,
        "face_frame_paths": face_frame_paths,
        "video_name": video_name
    }

@app.route('/temp/<path:filename>')
def serve_temp_image(filename):
    return send_from_directory(tempfile.gettempdir(), filename)

@app.route('/temp_frames/<path:filename>')
def serve_temp_frames(filename):
    return send_from_directory(tempfile.gettempdir(), filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
