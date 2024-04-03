import cv2
import numpy as np
import pytesseract
import re

# Configuración del módulo de reconocimiento de caracteres
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Función para detectar placas
def detect_plate(frame):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar umbral adaptativo para resaltar los bordes
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Encontrar contornos en la imagen umbral
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Obtener el perímetro del contorno
        perimeter = cv2.arcLength(contour, True)
        
        # Aproximar un polígono al contorno
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        
        # Si el contorno tiene 4 vértices (forma rectangular)
        if len(approx) == 4:
            # Dibujar el contorno delimitador
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
            
            # Recortar y extraer la región de la placa
            x, y, w, h = cv2.boundingRect(approx)
            plate_roi = frame[y:y+h, x:x+w]
            
            # Aplicar OCR a la región de la placa
            plate_text = pytesseract.image_to_string(plate_roi, lang='eng', config='--psm 6')
            
            # Filtrar solo letras, números y "-"
            plate_text_filtered = re.sub(r'[^a-zA-Z0-9\-]', '', plate_text)
            
            # Si la placa tiene 7 caracteres y 2 guiones, mostrarla en el marco
            if len(plate_text_filtered) == 9 and plate_text_filtered.count('-') == 2:
                cv2.putText(frame, plate_text_filtered, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Mostrar el texto de la placa en la consola
                print("Placa detectada:", plate_text_filtered)
    
    return frame

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

while True:
    # Leer un nuevo frame del flujo de video
    ret, frame = cap.read()
    
    # Si la captura de video es exitosa
    if ret:
        # Detectar las placas en el frame
        frame = detect_plate(frame)
        
        # Mostrar el frame resultante
        cv2.imshow('Placa Detector', frame)
        
        # Esperar por la tecla 'q' para salir del bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Liberar la captura de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
    