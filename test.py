from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
import dlib

# Inicializa o vídeo
video = cv2.VideoCapture(0)

# Carrega o classificador Haar (já existente)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Inicializa o detector de rostos do dlib (alternativa)
detector = dlib.get_frontal_face_detector()

# Carregar os dados de faces e rótulos
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)

# Ajustando o número de rótulos para o número de faces (se necessário)
LABELS = LABELS[:FACES.shape[0]]

# Normalização dos dados
scaler = StandardScaler()
FACES = scaler.fit_transform(FACES)

# Inicializa o classificador KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Carregar a imagem de fundo
# imgBackground = cv2.imread("background.png")  # imagem do fundo
COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Usando o detector dlib para detectar rostos (alternativa ao Haar Cascade)
    faces = detector(gray)

    for face in faces:
        # Usando as coordenadas do dlib para desenhar o retângulo ao redor do rosto
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())

        # Recorte da face detectada
        crop_img = frame[y:y + h, x:x + w, :]
        
        # Redimensionando a imagem da face e fazendo a predição com KNN
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        label = output[0] if output[0] in LABELS else "Desconhecido"
        
        # Obtendo o timestamp
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        
        # Verificando se o arquivo de presença já existe
        exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")
        
        # Desenhando o retângulo ao redor do rosto
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        
        attendance = [label, timestamp]

    # Ajustando o tamanho do frame de vídeo e aplicando no fundo
    frame_resized = cv2.resize(frame, (640, 480))
    # imgBackground[162:162 + 480, 55:55 + 640] = frame_resized
    cv2.imshow("Frame", frame)  # Exibindo a imagem do fundo, possibilidade de trocar "frame" por "background"

    # Ações de gravação de presença
    k = cv2.waitKey(1)
    if k == ord('o'):  # Quando pressionado 'o', registra a presença
        time.sleep(5)
        if exist:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
        else:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)

    if k == ord('q'):  # Quando pressionado 'q', sai
        break

video.release()
cv2.destroyAllWindows()
