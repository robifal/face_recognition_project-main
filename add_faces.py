import cv2 
import pickle
import numpy as np
import os
import dlib  # Importando o Dlib

# Inicializando a captura de vídeo
video = cv2.VideoCapture(0)

# Inicializando o detector de rostos do Dlib
detector = dlib.get_frontal_face_detector()

faces_data = []
i = 0

name = input("Digite seu nome: ")

# Contador de fotos tiradas
num_photos_taken = 0

while True:
    ret, frame = video.read()
    
    # Verifica se o vídeo foi capturado corretamente
    if not ret:
        print("Falha ao capturar o vídeo")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecção de rostos com o Dlib
    faces = detector(gray)
    
    print(f"Faces detectadas: {len(faces)}")  # Depuração: verifique quantas faces estão sendo detectadas
    
    for face in faces:
        # Dlib retorna um retângulo para a face detectada
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50))  # Ajustei para 50x50 pixels
        
        # Captura de uma nova foto a cada 10 iterações
        if len(faces_data) < 20 and i % 10 == 0:
            # Achata a imagem para um vetor 1D de 2500 elementos (50x50 pixels)
            faces_data.append(resized_img.flatten())
            num_photos_taken += 1  # Atualiza o contador de fotos tiradas

        i += 1
        # Exibindo o número de fotos tiradas e a quantidade de faces detectadas
        cv2.putText(frame, f"Fotos tiradas: {num_photos_taken}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.putText(frame, str(len(faces_data)), (50, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
    
    # Exibindo a imagem com a detecção de rosto
    cv2.imshow("Frame", frame)
    
    k = cv2.waitKey(1)
    
    # Condição para finalizar o processo (pressionar 'q' ou capturar 20 fotos)
    if k == ord('q') or len(faces_data) == 20:
        break

video.release()
cv2.destroyAllWindows()

# Convertendo para um array numpy
faces_data = np.asarray(faces_data)

# Atualiza o arquivo de nomes
if 'names.pkl' not in os.listdir('data/'):
    names = [name] * 20
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    names += [name] * 20
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

# Atualiza o arquivo de faces
if 'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    # Concatenando as novas faces de forma segura
    faces = np.append(faces, faces_data, axis=0)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)

print(f"Total de fotos tiradas: {num_photos_taken}")


