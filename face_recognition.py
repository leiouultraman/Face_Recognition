import cv2
import pickle
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from model import YOLO
from inception_resnet_v1 import InceptionResnetV1
import torch
from face_distance import find_distance
confidences = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('star_train/face_encodings.pkl', 'rb') as file:
    face_encodings_dict = pickle.load(file)
known_faces = []
known_names = []
for name, encodings in face_encodings_dict.items():
    known_faces.extend([encodings])
    known_names.extend([name])
img_path = 'star_test/111.jpg'
show_unknown = False
unknown_face_encodings = []

torlerance = 0.4

model = YOLO('yolov11n-face.pt')

resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
face_locations_results = model.predict(source=img_path, task='detect', save=False, device=device, imgsz=480, verbose=True)
img = cv2.imread(img_path)
for result in face_locations_results:
    boxes = result.boxes
    for box in boxes:
        face_embedding_list = []
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        face_img = img[int(y1):int(y2), int(x1):int(x2)]
        face_img_resized = cv2.resize(face_img, (160, 160))
        face_img_rgb = cv2.cvtColor(face_img_resized, cv2.COLOR_BGR2RGB)
        face_cropped = torch.from_numpy(face_img_rgb).float().permute(2, 0, 1)
        face_cropped = face_cropped / 255.0
        face_cropped = face_cropped.to(device)
        if face_cropped is not None:
            face_embedding = resnet(face_cropped.unsqueeze(0))
            face_embedding = face_embedding.squeeze(0)
            face_embedding_list = face_embedding.detach().cpu().numpy().tolist()
        unknown_face_encodings.append([face_embedding_list])
def draw_face_boxes(image, results, names, confidences, show_unknown=True):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype("msyh.ttc", 24)
    for result in results:
        boxes = result.boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
            if i < len(names):
                name = names[i]
                confidence = confidences[i]
                text_bbox = draw.textbbox((0, 0), name, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_x = x1 + 6
                text_y = y2 - text_height - 6
                if name != "Unknown" or show_unknown:
                    label = f"{name} ({confidence:.2f}%)"
                    draw.text((text_x, text_y), label, font=font, fill="green")
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return image
def compare_faces_in_image(known_encodings, unknown_encoding):
    alpha_embedding = np.asarray(known_encodings)
    beta_embedding = np.asarray(unknown_encoding)
    compare_results = find_distance(alpha_embedding, beta_embedding, distance_metric="cosine")
    compare_results = list(compare_results <= torlerance)
    compare_results = np.asarray(compare_results)
    return compare_results
def get_face_distance(known_encodings, unknown_encoding):
    return [find_distance([encoding], unknown_encoding, distance_metric='cosine') for encoding in known_encodings]
unknown_names = []
for unknown_encoding in unknown_face_encodings:
    results = compare_faces_in_image(known_faces, unknown_encoding)
    if True in results:
        distances = get_face_distance(known_faces, unknown_encoding)
        min_distance_index = distances.index(min(distances))
        confidences.append((1 - min(distances)[0][0]) * 100)
        unknown_names.append(known_names[min_distance_index])
    else:
        unknown_names.append("Unknown")
        confidences.append(100.0)
name_confidence = {}
for name, confidence in zip(unknown_names, confidences):
    if name not in name_confidence or name_confidence[name] < confidence:
        name_confidence[name] = confidence
final_names = []
final_confidences = []
for name, confidence in zip(unknown_names, confidences):
    if name_confidence[name] == confidence:
        final_names.append(name)
        final_confidences.append(confidence)
    else:
        final_names.append('Unknown')
        final_confidences.append(100)
filtered_names = [name for name in final_names if name != 'Unknown']
filtered_confidences = [confidence for name, confidence in zip(final_names, final_confidences) if name != 'Unknown']
def resize_image(image, target_width, target_height):
    resized_image = cv2.resize(image, (target_width, target_height))
    return resized_image
unknown_image_cv = draw_face_boxes(img, face_locations_results, final_names, final_confidences, show_unknown=show_unknown)
unknown_image_cv = resize_image(unknown_image_cv, 1600, 800)
cv2.imshow("Unknown Face", unknown_image_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()

