import cv2
import os

# Update paths as needed
base_video_dir = r"D:/ml project/pythonProject1/data"
output_frame_dir = r"D:/ml project/pythonProject1/frames"
categories = ['celeb-real', 'celeb-synthesis', 'youtube-real']
splits = ['train', 'val', 'test']

if not os.path.exists(output_frame_dir):
    os.makedirs(output_frame_dir)

for category in categories:
    for split in splits:
        video_dir = os.path.join(base_video_dir, category, split)
        frame_output_dir = os.path.join(output_frame_dir, category, split)
        os.makedirs(frame_output_dir, exist_ok=True)

        if os.path.exists(video_dir):
            for video_file in os.listdir(video_dir):
                video_path = os.path.join(video_dir, video_file)
                video_name = os.path.splitext(video_file)[0]
                video_frame_dir = os.path.join(frame_output_dir, video_name)
                os.makedirs(video_frame_dir, exist_ok=True)

                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_filename = os.path.join(video_frame_dir, f"frame{frame_count:04d}.jpg")
                    cv2.imwrite(frame_filename, frame)
                    frame_count += 1
                cap.release()
                print(f"Extracted {frame_count} frames for video {video_file} in {split} of {category}.")