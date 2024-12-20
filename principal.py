import os
import cv2
import face_recognition
import numpy as np
import datetime
import json
import threading

# Configuração dos diretórios
KNOWN_FACES_DIR = 'data/known_faces'
RECOGNITION_LOG_FILE = 'recognition_log.json'

if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

# Função para carregar rostos conhecidos
def load_known_faces():
    known_faces = []
    known_names = []
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith('.jpg'):
            name = filename.split('_')[0]  # Nome antes do "_X.jpg"
            name = os.path.splitext(name)[0]  # Remover a extensão .jpg do nome
            img_path = os.path.join(KNOWN_FACES_DIR, filename)
            # Carregar imagem e calcular as "encodings"
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:  # Verifica se encontrou um rosto
                known_faces.append(encodings[0])
                known_names.append(name)
    return known_faces, known_names

def create_panel(frame, name, accuracy=None):
    person_image_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
    if os.path.exists(person_image_path):
        person_image = cv2.imread(person_image_path)
        person_image = cv2.resize(person_image, (200, 200))  # Ajuste o tamanho da foto
        panel_height = max(frame.shape[0], 200)
        panel_width = frame.shape[1] + 200
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel[:frame.shape[0], :frame.shape[1]] = frame
        panel[:200, frame.shape[1]:] = person_image
        
        # Adicionar o indicador de precisão, se fornecido
        if accuracy is not None:
            cv2.putText(panel, f"Precisão: {accuracy:.2f}%", (frame.shape[1] + 10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return panel
    return frame

# Função para carregar o histórico de reconhecimentos
def load_recognition_log():
    if os.path.exists(RECOGNITION_LOG_FILE):
        with open(RECOGNITION_LOG_FILE, 'r') as file:
            return json.load(file)
    return {}

# Função para salvar o histórico de reconhecimentos
def save_recognition_log(recognition_log):
    with open(RECOGNITION_LOG_FILE, 'w') as file:
        json.dump(recognition_log, file, indent=4)

# Função para processar reconhecimento facial
def process_frame(frame, known_faces, known_names, recognition_log):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Localiza rostos no frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Comparação com rostos conhecidos
        matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.4)  # Mais rígido
        name = "Desconhecido"
        accuracy = None

        if True in matches:
            # Encontrar o índice do rosto correspondente
            match_index = matches.index(True)
            name = known_names[match_index]

            # Calcular a precisão (distância) entre os rostos
            face_distances = face_recognition.face_distance([known_faces[match_index]], face_encoding)
            accuracy = (1 - face_distances[0]) * 100  # A precisão é inversamente proporcional à distância
            
            # Se a precisão for muito baixa, marcar como "Desconhecido"
            if accuracy < 60:  # Limite de precisão abaixo do qual não é considerado válido
                name = "Desconhecido"
                accuracy = None

        # Desenhar o nome e o bounding box no frame
        top, right, bottom, left = face_location
        color = (0, 255, 0) if name != "Desconhecido" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        frame = create_panel(frame, name, accuracy)

        # Registro de reconhecimento
        if name != "Desconhecido":
            now = datetime.datetime.now()
            current_time = now.strftime("%Y-%m-%d %H:%M:%S")

            # Verifica se já foi registrada anteriormente
            if name in recognition_log:
                # Obter o tempo do último reconhecimento registrado
                last_time_str = recognition_log[name][-1]
                last_time = datetime.datetime.strptime(last_time_str, "%Y-%m-%d %H:%M:%S")

                # Verificar se já se passaram 60 segundos
                time_diff = now - last_time
                if time_diff.total_seconds() >= 60:
                    recognition_log[name].append(current_time)
                    print(f"{name} reconhecido novamente às {current_time} com {accuracy:.2f}% de precisão")
            else:
                # Se for a primeira vez que é reconhecida, registra
                recognition_log[name] = [current_time]
                print(f"{name} reconhecido pela primeira vez às {current_time} com {accuracy:.2f}% de precisão")
    return frame

# Função para capturar e identificar rostos
def capture_and_identify_faces():
    cap = cv2.VideoCapture("rtsp://ulisses:@dev2024@10.1.66.218:554/cam/realmonitor?channel=1&subtype=0")

    if not cap.isOpened():
        print("Erro: Não foi possível acessar o stream da câmera.")
        return

    # Carrega os rostos conhecidos
    known_faces, known_names = load_known_faces()

    # Carrega o histórico de reconhecimentos
    recognition_log = load_recognition_log()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar o vídeo")
            break

        # Reduz a resolução para processamento, mantendo o stream em alta qualidade
        small_frame = cv2.resize(frame, (620, 420))

        # Processar o reconhecimento facial de forma assíncrona
        processed_frame = threading.Thread(target=process_frame, args=(small_frame, known_faces, known_names, recognition_log))
        processed_frame.start()

        # Exibe o frame original
        cv2.imshow("Reconhecimento Facial", frame)
        save_recognition_log(recognition_log)  # Salva o histórico de reconhecimentos no arquivo JSON

        # Parar o loop ao pressionar 'q'
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Execução principal
if __name__ == "__main__":
    capture_and_identify_faces()
