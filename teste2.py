import cv2 
import os
import numpy as np
import face_recognition
import mediapipe as mp

# Configuração dos diretórios
KNOWN_FACES_DIR = 'data/known_faces'

if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)


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

    # Variável para armazenar o último nome identificado
    last_identified_name = None

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        while True:
            ret, frame = video.read()
            if not ret:
                print("Falha ao capturar o vídeo")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    # Obtém as coordenadas da face
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                    x, y = max(x, 0), max(y, 0)
                    w_box, h_box = min(w_box, w - x), min(h_box, h - y)

                    # Recorte do rosto
                    crop_img = frame[y:y + h_box, x:x + w_box]
                    if crop_img.size == 0:
                        continue

                    # Extrai as características faciais
                    rgb_crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                    face_encodings = face_recognition.face_encodings(rgb_crop_img)
                    if not face_encodings:
                        continue

                    captured_face_encoding = face_encodings[0]
                    name = get_face_name(captured_face_encoding)
                    color = (0, 255, 0) if name != "Desconhecido" else (0, 0, 255)

                    # Atualiza o painel apenas se o nome mudar
                    if name != last_identified_name:
                        last_identified_name = name
                        if name != "Desconhecido":
                            panel = create_panel(frame, name)
                        else:
                            panel = frame

                        # Atualiza a janela com o painel ou feed da câmera
                        cv2.imshow("Painel Principal", panel)

                    # Desenha a caixa no rosto
                    cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
                    draw_text_box(frame, x, y, w_box, h_box, name, color)

            # Atualiza o feed da câmera se nenhuma detecção for feita
            if not results.detections:
                last_identified_name = None
                cv2.imshow("Painel Principal", frame)

            # Sai do loop ao pressionar 'q'
            if cv2.waitKey(1) == ord('q'):
                break

    video.release()
    cv2.destroyAllWindows()


# Execução principal
if __name__ == "__main__":
    capture_and_identify_faces()