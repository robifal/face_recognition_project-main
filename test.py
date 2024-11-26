import cv2
import os
import numpy as np
import face_recognition
import mediapipe as mp
from PIL import Image

# Configuração dos diretórios
KNOWN_FACES_DIR = 'data/known_faces'

if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

def convert_images_to_jpg(directory):
    """Converte imagens .jpeg para .jpg"""
    for filename in os.listdir(directory):
        if filename.endswith('.jpeg'):  # Apenas arquivos .jpeg
            file_path = os.path.join(directory, filename)
            new_filename = filename.replace('.jpeg', '.jpg')  # Renomeia para .jpg
            new_file_path = os.path.join(directory, new_filename)

            # Abre e converte a imagem
            with Image.open(file_path) as img:
                rgb_img = img.convert('RGB')  # Garante que seja RGB
                rgb_img.save(new_file_path, 'JPEG')  # Salva como .jpg

            # Remove o arquivo .jpeg original
            os.remove(file_path)
            print(f"Convertido: {filename} -> {new_filename}")

# Função para carregar os rostos conhecidos e suas características faciais
def load_known_faces():
    known_faces = {}
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith('.jpg'):
            name = filename.split('_')[0]  # Nome antes do "_X.jpg"
            img_path = os.path.join(KNOWN_FACES_DIR, filename)
            img = cv2.imread(img_path)

            # Usando face_recognition para encontrar as características faciais
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(img_rgb)

            if face_encodings:
                known_faces[name] = face_encodings[0]  # Usando a primeira face encontrada
            else:
                print(f"Não foi possível encontrar um rosto em {filename}.")
    return known_faces

def is_new_face(captured_face_encoding, last_face_encoding, threshold=0.4):
    """
    Verifica se a face capturada é significativamente diferente da última.
    Args:
        captured_face_encoding: Codificação da face capturada.
        last_face_encoding: Codificação da última face identificada.
        threshold: Limite de distância para considerar uma nova face.
    Returns:
        bool: True se a face capturada for nova, False caso contrário.
    """
    if last_face_encoding is None:
        return True
    distance = np.linalg.norm(captured_face_encoding - last_face_encoding)
    return distance > threshold


# Função para capturar rostos e identificá-los
def capture_and_identify_faces():
    # Inicializa a detecção facial e a câmera
    mp_face_detection = mp.solutions.face_detection
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Erro: Não foi possível acessar a câmera.")
        return

    # Carrega os rostos conhecidos
    known_faces = load_known_faces()

    def get_face_name(captured_face_encoding):
        """Compara a face capturada com as conhecidas e retorna o nome."""
        min_distance = float('inf')
        name = "Desconhecido"
        for known_name, known_face_encoding in known_faces.items():
            distance = np.linalg.norm(known_face_encoding - captured_face_encoding)
            if distance < min_distance:
                min_distance = distance
                name = known_name
        return name if min_distance <= 0.6 else "Desconhecido"

    def draw_text_box(frame, x, y, w_box, h_box, name, color):
        """Desenha a caixa de texto e o nome da pessoa."""
        text = f"Nome: {name}"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
        text_x, text_y = x, y + h_box + text_height + 10
        cv2.rectangle(frame, (text_x - 5, text_y - text_height - 5), (text_x + text_width + 5, text_y + 5), color, cv2.FILLED)
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

    def create_panel(frame, name):
        """Cria o painel com o feed da câmera e a foto identificada."""
        person_image_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
        if os.path.exists(person_image_path):
            person_image = cv2.imread(person_image_path)
            person_image = cv2.resize(person_image, (200, 200))
            panel_height = max(frame.shape[0], 200)
            panel_width = frame.shape[1] + 200
            panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
            panel[:frame.shape[0], :frame.shape[1]] = frame
            panel[:200, frame.shape[1]:] = person_image
            return panel
        return frame

    current_panel = None
    last_identified_name = None
    last_face_encoding = None

    frame_interval = 5  # Processa um quadro a cada 5 frames
    frame_count = 0

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        while True:
            ret, frame = video.read()
            if not ret:
                print("Falha ao capturar o vídeo")
                break

            frame_count += 1
            if frame_count % frame_interval != 0:
                continue  # Pula os quadros que não são necessários para detecção

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                    x, y = max(x, 0), max(y, 0)
                    w_box, h_box = min(w_box, w - x), min(h_box, h - y)

                    crop_img = frame[y:y + h_box, x:x + w_box]
                    if crop_img.size == 0:
                        continue

                    rgb_crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                    face_encodings = face_recognition.face_encodings(rgb_crop_img)
                    if not face_encodings:
                        continue

                    captured_face_encoding = face_encodings[0]

                    if is_new_face(captured_face_encoding, last_face_encoding):
                        last_face_encoding = captured_face_encoding
                        name = get_face_name(captured_face_encoding)

                        if name != last_identified_name:
                            last_identified_name = name
                            if name != "Desconhecido":
                                current_panel = create_panel(frame, name)
                            else:
                                current_panel = frame

                    color = (0, 255, 0) if name != "Desconhecido" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
                    draw_text_box(frame, x, y, w_box, h_box, name, color)

            if not results.detections:
                last_identified_name = None
                current_panel = frame

            if current_panel is not None:
                cv2.imshow("Painel Principal", current_panel)

            if cv2.waitKey(1) == ord('q'):
                break

    video.release()
    cv2.destroyAllWindows()

# Execução principal
if __name__ == "__main__":
    convert_images_to_jpg(KNOWN_FACES_DIR)
    capture_and_identify_faces()
