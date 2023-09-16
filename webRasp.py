from ultralytics import YOLO
import numpy as np

detection_count = [0, 0, 0, 0]  # Reset detection counts

model = YOLO('weights/base-m.pt')
img = 'static/uploads/blueberry348.jpg'
results = model.predict([img], conf=0.5)
unique_elements, counts = np.unique(results[0].boxes.cls, return_counts=True)
detections = dict(zip(unique_elements, counts))

berries_dict = {0: "Малина", 1: "Черника", 2: "Морошка", 3: "Клубника"}
detections_text = "Ягоды:\n"

for class_id, count in detections.items():
    detection_count[int(class_id)] += count
for class_id, count in enumerate(detection_count):
    if count == 0:
        continue
    detections_text += f"{berries_dict[class_id]}: {detection_count[int(class_id)]}\n"
    print(detections_text)