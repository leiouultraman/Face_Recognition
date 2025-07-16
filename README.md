# Face_Recognition
This repository is based on  
facenet-pytorch：https://github.com/timesler/facenet-pytorch  
yolo-face：https://github.com/akanametov/yolo-face  
deepface：https://github.com/serengil/deepface  
using PyTorch, allowing for easy GPU acceleration.
# Download Pretrained models
Download the pretrained models from the following link.  
yolo-face:https://github.com/akanametov/yolo-face.(yolov11n-face.pt,yolov11s-face.pt,yolov11m-face.pt,yolov11l-face.pt)  
facenet-pytorch:https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt or https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180408-102900-casia-webface.pt.  
Then place them in the ​model​ folder.
# Quick start
1.install:
  ```
  conda create -n Face python=3.9
  conda activate Face
  pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
  pip install ultralytics
  pip install Flask
 ```
2.Face detection  
  ```
  python face_detection.py
  ```
  or  
  ```
  import cv2
  from ultralytics import YOLO
  import torch
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # Choose model
  model = YOLO('model/yolov11n-face.pt')
  # Choose picture
  img_path = 'star_test/111.jpg'
  face_locations_results = model.predict(source=img_path, task='detect', save=False, device=device, imgsz=480, verbose=True)
  img = cv2.imread(img_path)
  def draw_face_boxes(image, results):
      for result in results:
          boxes = result.boxes
          for box in boxes:
              x1, y1, x2, y2 = box.xyxy[0].tolist()
              cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
      return image
  img_with_boxes = draw_face_boxes(img, face_locations_results)
  cv2.imshow("Detected Faces", img_with_boxes)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  ```
![image](https://github.com/leiouultraman/Face_Recognition/blob/main/img/111.jpg)
3.Face encodings
  ```
  python face_encodings_folder.py
  ```
  or  
  ```
  import os
  import cv2
  import torch
  import pickle
  from ultralytics import YOLO
  from inception_resnet_v1 import InceptionResnetV1
  from pathlib import Path
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # Choose yolo_face model
  model = YOLO('model/yolov11n-face.pt')
  # Choose facenet model (vggface2 or casia-webface)
  resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
  # Choose folder(Store personnel photos, naming them with the corresponding individuals' names.)
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
  ```
4.Face Recognition
```
  python face_recognition.py
  ```
  or  
  ```
  import os
  import cv2
  import torch
  import pickle
  from ultralytics import YOLO
  from inception_resnet_v1 import InceptionResnetV1
  from pathlib import Path
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # Choose yolo_face model
  model = YOLO('model/yolov11n-face.pt')
  # Choose facenet model (vggface2 or casia-webface)
  resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
  # Choose folder(Store personnel photos, naming them with the corresponding individuals' names.)
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
  ```
