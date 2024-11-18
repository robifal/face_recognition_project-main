import cv2
import pickle
import numpy as np
import os
import mediapipe as mp

# Configuração dos diretórios
KNOWN_FACES_DIR = 'data/known_faces'
UNKNOWN_FACES_DIR = 'data/unknown_faces'

if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)
if not os.path.exists(UNKNOWN_FACES_DIR):
    os.makedirs(UNKNOWN_FACES_DIR)

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

# Função para capturar rostos usando mediapipe
def capture_faces():
    mp_face_detection = mp.solutions.face_detection
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Erro: Não foi possível acessar a câmera.")
        return

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        unknown_counter = 1  # Contador para rostos desconhecidos
        known_faces = load_data('known_faces.pkl', {})
        known_face_counts = load_data('known_face_counts.pkl', {})

        # Carrega rostos conhecidos da pasta KNOWN_FACES_DIR
        for filename in os.listdir(KNOWN_FACES_DIR):
            if filename.endswith('.jpg'):
                name = filename.split('_')[0]  # Nome do arquivo antes do "_1.jpg"
                img_path = os.path.join(KNOWN_FACES_DIR, filename)
                img = cv2.imread(img_path)
                resized_img = cv2.resize(img, (100, 100))  # Redimensiona para 100x100
                known_faces[name] = resized_img.flatten()

        while True:
            ret, frame = video.read()
            if not ret:
                print("Falha ao capturar o vídeo")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                    # Coordenadas dentro dos limites
                    x, y = max(x, 0), max(y, 0)
                    w_box, h_box = min(w_box, w - x), min(h_box, h - y)

                    # Recorte do rosto
                    crop_img = frame[y:y + h_box, x:x + w_box]
                    if crop_img.size == 0:
                        continue

                    resized_img = cv2.resize(crop_img, (100, 100))
                    flattened_img = resized_img.flatten()

                    # Comparar com rostos conhecidos
                    min_distance = float('inf')
                    name = "Desconhecido"
                    for known_name, known_face in known_faces.items():
                        distance = np.linalg.norm(known_face - flattened_img)
                        if distance < min_distance:
                            min_distance = distance
                            name = known_name

                    # Considerar rosto desconhecido se distância for grande
                    if min_distance > 90:
                        name = f"Desconhecido_{unknown_counter}"

                        # Armazena até 20 fotos para um novo rosto desconhecido
                        if name not in known_face_counts:
                            known_face_counts[name] = 0

                        if known_face_counts[name] < 20:
                            # Salva na pasta de rostos desconhecidos
                            cv2.imwrite(f"{UNKNOWN_FACES_DIR}/{name}_{known_face_counts[name]}.jpg", crop_img)
                            known_face_counts[name] += 1
                        else:
                            unknown_counter += 1  # Incrementa para próximo rosto desconhecido
                    else:
                        # Nome do rosto reconhecido
                        if name not in known_face_counts:
                            known_face_counts[name] = 1  # Marca rosto conhecido

                    # Caixa de detecção e nome
                    color = (0, 255, 0) if min_distance <= 90 else (0, 0, 255)
                    cv2.putText(frame, f"Nome: {name}", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
                    cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)

            # Salvar dados após cada iteração
            save_data('known_faces.pkl', known_faces)
            save_data('known_face_counts.pkl', known_face_counts)

            # Exibe o frame
            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

# Execução principal
if __name__ == "__main__":
    capture_faces()
    print("Captura de rostos concluída.")
