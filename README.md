# Face_Recognition-API
This repository is based on https://github.com/timesler/facenet-pytorch, https://github.com/akanametov/yolo-face and https://github.com/serengil/deepface, using PyTorch, allowing for easy GPU acceleration.
# Quick start
1.install:
  ```
  conda create -n Face python=3.9
  conda activate Face
  pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
  pip install ultralytics
  pip install Flask
 ```
2.Face_detection
  ```
  import cv2
  from ultralytics import YOLO
  import torch
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = YOLO('model/yolov11n-face.pt')
  # Choose a picture
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
  '''

