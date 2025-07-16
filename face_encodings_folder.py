import os
import cv2
import torch
import pickle
from ultralytics import YOLO
from inception_resnet_v1 import InceptionResnetV1
from pathlib import Path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = YOLO('model/yolov11n-face.pt')

resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()

input_folder = 'star_train'
face_encodings = {}
if os.path.exists(f'{input_folder}/face_encodings.pkl'):
    with open(f'{input_folder}/face_encodings.pkl', 'rb') as f:
        face_encodings = pickle.load(f)
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_folder, filename)
        print(f"Processing {filename}...")
        name = Path(filename).stem
        results = model.predict(source=img_path, task='detect', save=False, device=device, imgsz=480, verbose=False)
        img = cv2.imread(img_path)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                face_img = img[int(y1):int(y2), int(x1):int(x2)]
                face_img_resized = cv2.resize(face_img, (160, 160))
                face_img_rgb = cv2.cvtColor(face_img_resized, cv2.COLOR_BGR2RGB)
                face_cropped = torch.from_numpy(face_img_rgb).float().permute(2, 0, 1)
                face_cropped = face_cropped / 255.0
                face_cropped = face_cropped.to(device)
                face_embedding = resnet(face_cropped.unsqueeze(0))
                face_embedding = face_embedding.squeeze(0)
                face_embedding_list = face_embedding.detach().cpu().numpy().tolist()
                face_encodings[name] = face_embedding_list
with open(f'{input_folder}/face_encodings.pkl', 'wb') as f:
    pickle.dump(face_encodings, f)
print(f"Face encodings saved to '{input_folder}/face_encodings.pkl'")
