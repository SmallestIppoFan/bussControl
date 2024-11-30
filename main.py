import cv2
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

cap = cv2.VideoCapture(0)  # Если у вас несколько камер, попробуйте другие индексы (например, 1, 2)

if not cap.isOpened():
    print("Ошибка при подключении к камере!")
    exit()

max_capacity = 10  # Максимальное количество людей, которое может быть в кадре

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось захватить кадр")
        break

    results = model(frame)

    detections = results.xywh[0]

    people_count = sum([1 for det in detections if det[5] == 0])

    occupancy = min((people_count / max_capacity) * 100, 100)

    cv2.putText(frame, f'Occupancy: {occupancy:.1f}%', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Bus Occupancy', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Закрываем все окна
cap.release()
cv2.destroyAllWindows()
