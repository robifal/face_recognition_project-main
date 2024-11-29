import os
import cv2
import face_recognition
import numpy as np
import time

# Configuração dos diretórios
KNOWN_FACES_DIR = 'data/known_faces'

if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)


# Função para carregar rostos conhecidos
def load_known_faces():
    known_faces = []
    known_names = []
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith('.jpg'):
            name = filename.split('_')[0]  # Nome antes do "_X.jpg"
            img_path = os.path.join(KNOWN_FACES_DIR, filename)
            # Carregar imagem e calcular as "encodings"
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:  # Verifica se encontrou um rosto
                known_faces.append(encodings[0])
                known_names.append(name)
    return known_faces, known_names

def create_panel(frame, name):
    person_image_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
    if os.path.exists(person_image_path):
        person_image = cv2.imread(person_image_path)
        person_image = cv2.resize(person_image, (200, 200))  # Ajuste o tamanho da foto
        panel_height = max(frame.shape[0], 200)
        panel_width = frame.shape[1] + 200
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel[:frame.shape[0], :frame.shape[1]] = frame
        panel[:200, frame.shape[1]:] = person_image
        return panel
    return frame

# Função para capturar e identificar rostos
def capture_and_identify_faces():
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Erro: Não foi possível acessar a câmera.")
        return

    # Carrega os rostos conhecidos
    known_faces, known_names = load_known_faces()

    while True:
        ret, frame = video.read()
        if not ret:
            print("Falha ao capturar o vídeo")
            break

        # Convertendo para RGB (necessário para face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Localiza rostos no frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Comparação com rostos conhecidos
            matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.6)
            name = "Desconhecido"

            if True in matches:
                # Encontrar o índice do rosto correspondente
                match_index = matches.index(True)
                name = known_names[match_index]

            # Desenhar o nome e o bounding box no frame
            top, right, bottom, left = face_location
            color = (0, 255, 0) if name != "Desconhecido" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            

            if True in matches:
                match_index = matches.index(True)
                name = known_names[match_index]
                current_time = time.strftime("%Y-%m-%d %H:%M:%S")  # Formato: Ano-Mês-Dia Hora:Minuto:Segundo
                print(f"[{current_time}] Pessoa reconhecida: {name}")
            else:
                current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{current_time}] Rosto desconhecido")

            

            frame = create_panel(frame, name)

        # Exibe o frame
        cv2.imshow("Reconhecimento Facial", frame)

        # Parar o loop ao pressionar 'q'
        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


# Execução principal
if __name__ == "__main__":
    capture_and_identify_faces()
