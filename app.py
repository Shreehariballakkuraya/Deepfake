import cv2
import torch
import numpy as np
import os
import io
import tempfile
import csv
from flask import Flask, jsonify, request, render_template
from PIL import Image
from torchvision import transforms
import boto3
from botocore.exceptions import NoCredentialsError

app = Flask(__name__, template_folder='.')

# Load environment variables
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
BUCKET_NAME = os.getenv('AWS_BUCKET_NAME', 'deepfakernsit')  # Replace with your S3 bucket name

# Initialize the S3 client
s3_client = boto3.client('s3',
                         aws_access_key_id=aws_access_key_id,
                         aws_secret_access_key=aws_secret_access_key,
                         region_name='eu-north-1')

# S3 folder prefixes
UPLOADS_S3_PREFIX = 'uploads/'
TEMP_S3_PREFIX = 'temp/'
TEMP_FRAMES_S3_PREFIX = 'temp_frames/'

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 3  # Adjust based on your dataset classes
model = torch.load("best_model.pth", map_location=device)  # Load model
model.eval()

# Define preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    """Handles video uploads, stores in S3, and triggers processing."""
    video_file = request.files['video']
    if video_file:
        video_name = video_file.filename
        video_s3_key = os.path.join(UPLOADS_S3_PREFIX, video_name)

        try:
            # Upload video to S3
            s3_client.upload_fileobj(video_file, BUCKET_NAME, video_s3_key)

            # Process the video
            result = process_video(video_name)
            return jsonify(result)
        except NoCredentialsError:
            return jsonify({"error": "AWS credentials not available."}), 403
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "No video file provided."}), 400


def download_file_from_s3(s3_key):
    """Download a file from S3 as bytes."""
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key)
        return response['Body'].read()
    except Exception as e:
        raise RuntimeError(f"Error downloading file from S3: {str(e)}")


def upload_to_s3(file_bytes, s3_key):
    """Upload a file (bytes) to S3."""
    try:
        s3_client.put_object(Bucket=BUCKET_NAME, Key=s3_key, Body=file_bytes)
    except Exception as e:
        raise RuntimeError(f"Error uploading file to S3: {str(e)}")


def process_video(video_name):
    """Process the video, save frames and cropped faces to S3."""
    video_s3_key = os.path.join(UPLOADS_S3_PREFIX, video_name)

    # Download video from S3
    video_bytes = download_file_from_s3(video_s3_key)
    video_np = np.frombuffer(video_bytes, dtype=np.uint8)

    # Read video
    temp_video_path = tempfile.NamedTemporaryFile(delete=False).name  # Temporary video file path
    with open(temp_video_path, 'wb') as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        return {"error": "Cannot open the video file."}

    frame_count = 0
    processed_frame_count = 0
    fake_frame_count = 0
    total_faces_detected = 0
    frame_paths = []
    face_frame_paths = []

    # Face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 5 != 0:
            continue  # Process every 5th frame

        processed_frame_count += 1

        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            total_faces_detected += len(faces)

            # Save full frame to S3
            _, buffer = cv2.imencode('.jpg', frame)
            frame_key = os.path.join(TEMP_S3_PREFIX, f"frame_{processed_frame_count:04d}.jpg")
            upload_to_s3(buffer.tobytes(), frame_key)
            frame_paths.append(frame_key)

            # Save cropped faces to S3
            for i, (x, y, w, h) in enumerate(faces):
                face_crop = frame[y:y + h, x:x + w]
                _, face_buffer = cv2.imencode('.jpg', face_crop)
                face_key = os.path.join(TEMP_FRAMES_S3_PREFIX, f"frame_{processed_frame_count:04d}_face_{i}.jpg")
                upload_to_s3(face_buffer.tobytes(), face_key)
                face_frame_paths.append(face_key)

                # Predict if the face is fake
                pil_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                input_tensor = transform(pil_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(input_tensor)
                    _, predicted_class = torch.max(output, 1)
                    if predicted_class.item() == 1:  # Assuming '1' indicates fake face
                        fake_frame_count += 1

    cap.release()

    return {
        "video_name": video_name,
        "total_frames_processed": processed_frame_count,
        "total_faces_detected": total_faces_detected,
        "fake_frame_count": fake_frame_count,
        "frame_paths": frame_paths,
        "face_frame_paths": face_frame_paths,
        "is_fake_video": fake_frame_count > total_faces_detected // 2
    }


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
