import cv2
import pickle
import numpy as np
import os
import mediapipe as mp


# Configuração dos diretórios
KNOWN_FACES_DIR = 'data/known_faces'


if not os.path.exists(KNOWN_FACES_DIR):
   os.makedirs(KNOWN_FACES_DIR)


# Função para carregar os rostos conhecidos
def load_known_faces():
   known_faces = {}
   for filename in os.listdir(KNOWN_FACES_DIR):
       if filename.endswith('.jpg'):
           name = filename.split('_')[0]  # Nome antes do "_X.jpg"
           img_path = os.path.join(KNOWN_FACES_DIR, filename)
           img = cv2.imread(img_path)
           resized_img = cv2.resize(img, (100, 100))  # Redimensiona para normalização
           known_faces[name] = resized_img.flatten()  # Salva o vetor do rosto
   return known_faces


# Função para capturar rostos e identificá-los
def capture_and_identify_faces():
   mp_face_detection = mp.solutions.face_detection
   video = cv2.VideoCapture(0)
   if not video.isOpened():
       print("Erro: Não foi possível acessar a câmera.")
       return


   # Carrega os rostos conhecidos
   known_faces = load_known_faces()


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


                   # Definir se o rosto foi identificado ou não
                   if min_distance > 50:  # Limite para identificação
                       name = "Desconhecido"


                   # Caixa de detecção e nome
                   color = (0, 255, 0) if name != "Desconhecido" else (0, 0, 255)
                   cv2.putText(frame, f"Nome: {name}", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
                   cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)


           # Exibe o frame
           cv2.imshow("Frame", frame)


           if cv2.waitKey(1) == ord('q'):
               break


       video.release()
       cv2.destroyAllWindows()


# Execução principal
if __name__ == "__main__":
   capture_and_identify_faces()
