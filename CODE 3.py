#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pytesseract
import cv2
import os
from google.colab.patches import cv2_imshow  # Pour afficher les images dans Colab
from google.colab import drive
drive.mount('/content/drive')
folder_path = '/content/drive/MyDrive/voitures'
file_names = os.listdir(folder_path)
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
custom_config = r'--oem 3 --psm 8'
for file_name in file_names:
    image_path = os.path.join(folder_path, file_name)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Impossible de charger l'image à l'emplacement {image_path}")
        continue
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blurred, 30, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    license_plate_contour = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
        if len(approx) == 4:
            license_plate_contour = approx
            break
    if license_plate_contour is not None:
        x, y, w, h = cv2.boundingRect(license_plate_contour)
        plate_roi = image[y:y+h, x:x+w]
        plate_text = pytesseract.image_to_string(plate_roi, config=custom_config)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, plate_text.strip(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2_imshow(image)

