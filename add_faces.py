import cv2
import pickle
import numpy as np
import os
import face_recognition

# Configuração dos diretórios
KNOWN_FACES_DIR = 'data/known_faces'

if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

# Função para salvar dados com pickle
def save_data(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_data(filename, default_value):
    if os.path.isfile(filename):
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except EOFError:
            print(f"Arquivo {filename} está vazio ou corrompido. Criando novo arquivo.")
            return default_value
    return default_value

# Função para capturar e salvar 20 fotos de uma pessoa
def capture_faces_for_person(name):
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Erro: Não foi possível acessar a câmera.")
        return

    photo_count = 0
    while photo_count < 20:
        ret, frame = video.read()
        if not ret:
            print("Falha ao capturar o vídeo")
            break

        # Converte a imagem para RGB, que é o formato esperado pelo face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Usa o face_recognition para localizar os rostos
        face_locations = face_recognition.face_locations(rgb_frame)

        # Se houver rostos detectados
        if face_locations:
            for face_location in face_locations:
                top, right, bottom, left = face_location

                # Recorta a região do rosto detectado
                crop_img = frame[top:bottom, left:right]
                if crop_img.size == 0:
                    continue

                # Redimensiona para garantir um bom tamanho e salvar
                img_resized = cv2.resize(crop_img, (250, 250))  # Tamanho maior para melhorar a captura
                filename = os.path.join(KNOWN_FACES_DIR, f"{name}_{photo_count}.jpg")
                cv2.imwrite(filename, img_resized)
                photo_count += 1

                # Exibe o rosto detectado
                color = (0, 255, 0)
                cv2.putText(frame, f"Capturando {name}", (left, top - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Exibe o frame
        cv2.imshow("Captura de Fotos", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

    print(f"Captura de fotos para {name} concluída. {photo_count} fotos salvas.")

# Função para capturar rostos e classificá-los
def capture_faces():
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Erro: Não foi possível acessar a câmera.")
        return

    known_faces = load_data('data/known_faces.pkl', {})

    while True:
        ret, frame = video.read()
        if not ret:
            print("Falha ao capturar o vídeo")
            break

        # Converte a imagem para RGB, que é o formato esperado pelo face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detecta os rostos usando face_recognition
        face_locations = face_recognition.face_locations(rgb_frame)

        if face_locations:
            for face_location in face_locations:
                top, right, bottom, left = face_location

                # Recorta o rosto detectado
                crop_img = frame[top:bottom, left:right]
                if crop_img.size == 0:
                    continue

                resized_img = cv2.resize(crop_img, (250, 250))
                flattened_img = resized_img.flatten()

                # Comparar com rostos conhecidos
                min_distance = float('inf')
                name = "Desconhecido"
                for known_name, known_face in known_faces.items():
                    distance = np.linalg.norm(known_face - flattened_img)
                    if distance < min_distance:
                        min_distance = distance
                        name = known_name

                # Caixa de detecção e nome
                color = (0, 255, 0) if min_distance <= 90 else (0, 0, 255)
                cv2.putText(frame, f"Nome: {name}", (left, top - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Salvar os rostos conhecidos
        save_data('data/known_faces.pkl', known_faces)

        # Exibe o frame
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

# Execução principal
if __name__ == "__main__":
    name = input("Digite o nome da pessoa para capturar as fotos: ")
    capture_faces_for_person(name)  # Captura as fotos da pessoa
    capture_faces()  # Inicia a captura e identificação de rostos no vídeo
    print("Captura de rostos concluída.")

