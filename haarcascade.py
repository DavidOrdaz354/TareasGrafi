import numpy as np
import cv2 as cv

rostro = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
cap = cv.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        break

    gris = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rostros = rostro.detectMultiScale(gris, 1.3, 5)

    for (x, y, w, h) in rostros:

        # Marco de cara
        cv.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

        # -------- OJOS --------
        ojo_y = y + int(h*0.35)
        ojo_x1 = x + int(w*0.3)
        ojo_x2 = x + int(w*0.7)

        cv.circle(img,(ojo_x1,ojo_y),20,(255,255,255),-1)
        cv.circle(img,(ojo_x2,ojo_y),20,(255,255,255),-1)

        cv.circle(img,(ojo_x1,ojo_y),8,(0,0,0),-1)
        cv.circle(img,(ojo_x2,ojo_y),8,(0,0,0),-1)

        # -------- NARIZ --------
        nariz = np.array([
            [x+w//2, y+int(h*0.45)],
            [x+int(w*0.45), y+int(h*0.60)],
            [x+int(w*0.55), y+int(h*0.60)]
        ])
        cv.drawContours(img,[nariz],0,(0,0,255),-1)

        # -------- BIGOTE --------
        bigote_y = y + int(h*0.62)

        cv.ellipse(img,(x+int(w*0.40),bigote_y),(20,10),0,0,180,(0,0,0),-1)
        cv.ellipse(img,(x+int(w*0.60),bigote_y),(20,10),0,0,180,(0,0,0),-1)

        # -------- BOCA --------
        boca_x = x + w//2
        boca_y = y + int(h*0.75)

        cv.ellipse(img,(boca_x,boca_y),(40,20),0,0,180,(0,0,255),3)

        # -------- OREJAS --------
        oreja_y = y + int(h*0.5)

        cv.ellipse(img,(x-15,oreja_y),(15,30),0,0,360,(200,180,160),-1)
        cv.ellipse(img,(x+w+15,oreja_y),(15,30),0,0,360,(200,180,160),-1)

    cv.imshow("Cara caricatura", img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()