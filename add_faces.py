import cv2
import pickle
import numpy as np
import os
import dlib

# Configuração do diretório de dados
DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Função para salvar dados com pickle
def save_data(filename, data):
    with open(os.path.join(DATA_DIR, filename), 'wb') as f:
        pickle.dump(data, f)

# Função para carregar dados com pickle
def load_data(filename, default_value):
    path = os.path.join(DATA_DIR, filename)
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            try:
                return pickle.load(f)
            except EOFError:
                pass  # Retorna o valor padrão se o arquivo estiver vazio
    return default_value

# Função para capturar rostos
def capture_faces(name, num_samples=20):
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Erro: Não foi possível acessar a câmera.")
        return
  
    detector = dlib.get_frontal_face_detector()
    faces_data = []
    num_photos_taken = 0
    i = 0
    while True:
        ret, frame = video.read()
        if not ret:
            print("Falha ao capturar o vídeo")
            break
      
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
      
        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50))
          
            if len(faces_data) < num_samples and i % 10 == 0:
                faces_data.append(resized_img.flatten())
                num_photos_taken += 1
            i += 1
            # Atualiza o frame com informações
            cv2.putText(frame, f"Fotos tiradas: {num_photos_taken}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == ord('q') or len(faces_data) == num_samples:
            break
    video.release()
    cv2.destroyAllWindows()
    return np.asarray(faces_data), num_photos_taken

# Função para atualizar os arquivos de dados
def update_files(name, faces_data):
    names = load_data('names.pkl', [])
    names.extend([name] * faces_data.shape[0])
    save_data('names.pkl', names)
  
    faces = load_data('faces_data.pkl', np.empty((0, faces_data.shape[1])))
    faces = np.append(faces, faces_data, axis=0)
    save_data('faces_data.pkl', faces)

# Execução principal
name = input("Digite seu nome: ")
faces_data, num_photos_taken = capture_faces(name)
if faces_data is not None:
    update_files(name, faces_data)
    print(f"Total de fotos tiradas: {num_photos_taken}")
else:
    print("Nenhuma foto foi tirada.")



