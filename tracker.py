import cv2 as cv
import numpy as np
import csv
import math
import time


# Función para obtener la distancia euclideana etre 2 puntos
def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


# Rutas de los archivos YOLOv4
yolo_cfg = 'yolov3.cfg'
yolo_weights = 'yolov3.weights'
coco_names = 'coco.names'

# Cargar las etiquetas de COCO
with open(coco_names, 'r') as f:
    labels = f.read().strip().split('\n')

# Cargar la red YOLO
net = cv.dnn.readNetFromDarknet(yolo_cfg, yolo_weights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Obtener los nombres de las capas de salida
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# Ruta del video
video = cv.VideoCapture('Ruta/del/video')

object_id = 0
objects = {}

# Coordenadas para las zonas de predicción
zones = {
    "pred_izq": [(300, 250), (400, 300)],
    "pred_der": [(600, 250), (700, 300)],
    "pred_recto": [(500, 200), (550, 250)],
    "real_izq": [(300, 350), (400, 400)],
    "real_der": [(600, 350), (700, 400)],
    "real_recto": [(500, 300), (550, 350)]
}
DIST_THRESHOLD = 50  # Umbral de distancia para considerar el mismo objeto
frame_count = 0

# Variables de trackeo
active_ids = set()


def is_in_zone(point, zone):
    (x1, y1), (x2, y2) = zone
    return x1 <= point[0] <= x2 and y1 <= point[1] <= y2


# Diccionario para el tiempo en cada zona de predicción
pred_time = {"pred_izq": 0, "pred_der": 0, "pred_recto": 0}
start_times = {}

while True:
    isTrue, frame = video.read()
    if not isTrue:
        break
    frame_count += 1

    blob = cv.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(ln)

    boxes = []
    confidences = []
    class_ids = []
    centroids = []
    current_objects = []

    for detection in detections:
        for object_detection in detection:
            scores = object_detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence >= 0.6 and labels[class_id] in ['car', 'truck']:
                center_x = int(object_detection[0] * frame.shape[1])
                center_y = int(object_detection[1] * frame.shape[0])
                width = int(object_detection[2] * frame.shape[1])
                height = int(object_detection[3] * frame.shape[0])
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                centroids.append((center_x, center_y))
                current_objects.append((center_x, center_y))

    indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)

    for (cx, cy) in current_objects:
        matched = False
        for obj_id, (prev_cx, prev_cy) in objects.items():
            distance = euclidean_distance((cx, cy), prev_cx)
            if distance < DIST_THRESHOLD:
                objects[obj_id] = ((cx, cy), frame_count)
                active_ids.add(obj_id)
                matched = True

                # Calcular tiempo en zona de predicción
                for zone_name, zone_coords in zones.items():
                    if "pred" in zone_name and is_in_zone((cx, cy), zone_coords):
                        if obj_id not in start_times:
                            start_times[obj_id] = time.time()
                        else:
                            pred_time[zone_name] += time.time() - start_times[obj_id]
                            start_times[obj_id] = time.time()

                break

        if not matched:
            objects[object_id] = ((cx, cy), frame_count)
            active_ids.add(object_id)
            start_times[object_id] = time.time()
            object_id += 1

    for obj_id in active_ids:
        (cx, cy), _ = objects[obj_id]
        cv.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
        cv.putText(frame, f'ID: {obj_id}', (cx, cy - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Limpiar IDs activos y remover objetos antiguos
    active_ids.clear()
    objects = {obj_id: data for obj_id, data in objects.items() if frame_count - data[1] < 30}

    # Mostrar el video con las detecciones
    cv.imshow('Video', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Predicción basada en tiempo en zona
pred_dominante = max(pred_time, key=pred_time.get)
print(f"Predicción de zona dominante: {pred_dominante} con tiempo {pred_time[pred_dominante]}")

video.release()
cv.destroyAllWindows()

# Crear archivo CSV con resultados
fields = ['Zona de Predicción', 'Tiempo Acumulado']
rows = [[zona, tiempo] for zona, tiempo in pred_time.items()]

filename = "prediccion_resultados.csv"
with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(rows)
