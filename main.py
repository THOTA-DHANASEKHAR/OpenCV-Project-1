import cv2
import numpy
from time import sleep

pos = 550
detec = []
count = 0


def obj_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


vc = cv2.VideoCapture('apple.mp4')
subtracao = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret, frame1 = vc.read()
    tempo = float(1 / 60)
    sleep(tempo)
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(grey, (3, 3), 5)

    img_sub = subtracao.apply(blur)

    dilat = cv2.dilate(img_sub, numpy.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    contorno, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1, (0, pos), (1600, pos), (255, 127, 0), 3)
    for (i, c) in enumerate(contorno):
        (x, y, w, h) = cv2.boundingRect(c)
        validar_contorno = (w >= 80) and (h >= 80) and (w <= 300)
        if not validar_contorno:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center = obj_center(x, y, w, h)
        detec.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

        for (x, y) in detec:
            if y < (pos + 10) and y > (pos - 10):
                count += 1
                cv2.line(frame1, (0, pos), (1600, pos), (0, 127, 255), 3)
                detec.remove((x, y))
                print("car is detected : " + str(count))

    cv2.putText(frame1, "apple COUNT : " + str(count), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Video Original", frame1)

    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()
vc.release()
