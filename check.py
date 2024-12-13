from PIL import Image

try:
    img = Image.open('frames_cropped/celeb-real/train/id0_0000/frame0000.jpg')
    img.show()
except PermissionError as e:
    print(f"PermissionError: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
