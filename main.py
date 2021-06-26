import cv2
import numpy as np
import random
import time


# ----------------------------------- Functions


def show_pic(img, t=10000):
    cv2.imshow('pic', img)
    cv2.waitKey(t)


def sticker_on_image(bg, overlay, pos):
    x1, y1, w1, h1 = pos
    roi = bg[y1:y1 + h1, x1:x1 + w1]

    overlay = cv2.resize(overlay, (w1, h1))
    sticker_ = overlay[:, :, :3]
    mask = overlay[:, :, 3]

    # create 3D mask with same dimensions as roi
    mask_3d = np.zeros_like(roi)

    # copy 2D mask to all dimensions of your 3D mask
    # because we need to multiply mask with a color image
    for i in range(3):
        mask_3d[:, :, i] = mask.copy()

    filter_ = mask_3d // 255  # 3D filter with 0 and 1 values

    sticker_ = np.multiply(sticker_, filter_)
    roi = np.multiply(roi, (mask_3d - 255))  # same result as mask_3d//255 (for uint8 dtype)

    return np.add(roi, sticker_)


# ---------------------------------------------

print("""
please choose from blow list: (Enter a number)
1-bounding box on eyes and lips
2-sticker on face
3-sticker on eyes and lips
4-blurring face
5-special effect on face""")

option = int(input('what is your choice? '))

face_detector = cv2.CascadeClassifier('xml\haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier('xml\haarcascade_eye_tree_eyeglasses.xml')
smile_detector = cv2.CascadeClassifier('xml\haarcascade_smile.xml')

# ------------------------------------- Inputs
# video = cv2.VideoCapture(0)
video = cv2.VideoCapture('obama.mp4')
# -------------------------------------

cnt = 1
selected = 0

while True:
    flag, frame = video.read()
    if flag:
        gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_detector.detectMultiScale(gray_image, 1.3)

        for i, face in enumerate(faces):
            x, y, w, h = face
            image_face = frame[y:y + h, x:x + w]
            eyes = eye_detector.detectMultiScale(image_face, 1.8)
            smiles = smile_detector.detectMultiScale(image_face, 1.8)

            if option == 1:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 0)
                for xe, ye, we, he in eyes:
                    cv2.rectangle(frame, (x + xe, y + ye), (x + xe + we, y + ye + he), (125, 255, 0), 0)
                for xs, ys, ws, hs in smile:
                    cv2.rectangle(frame, (x + xs, y + ys), (x + xs + ws, y + ys + hs), (0, 255, 0), 0)

            elif option == 2:
                sticker = cv2.imread('smile.png', cv2.IMREAD_UNCHANGED)
                # sticker = cv2.cvtColor(sticker, cv2.COLOR_RGB2GRAY)
                # sticker = cv2.resize(sticker, (w, h))
                # sticker = cv2.addWeighted(frame[y:y+h, x:x+w], 1, sticker, 1, 0)
                # frame[y:y + h, x:x + w] = sticker
                frame[y:y + h, x:x + w] = sticker_on_image(frame, sticker, face)

            elif option == 3:
                eye = cv2.imread('eye.png', cv2.IMREAD_UNCHANGED)
                lip = cv2.imread('Lips.png', cv2.IMREAD_UNCHANGED)
                for eye_ in eyes:
                    xe, ye, we, he = eye_
                    pos_eye = (x + xe, y + ye, we, he)
                    # eye = cv2.resize(eye, (we, he))
                    # eye = cv2.addWeighted(frame[y+ye:y+ye + he, x+xe:x+xe + we], 0.8, eye, 1,2, 0)
                    # frame[y+ye:y+ye + he, x+xe:x+xe + we] = eye
                    frame[y + ye:y + ye + he, x + xe:x + xe + we] = sticker_on_image(frame, eye, pos_eye)
                for smile in smiles:
                    xs, ys, ws, hs = smile
                    pos_lip = (x + xs, y + ys, ws, hs)
                    # lip = cv2.resize(lip, (ws, hs))
                    # lip = cv2.addWeighted(frame[y+ys:y+ys + hs, x+xs:x+xs + ws], 0.8, lip, 1.2, 0)
                    # frame[y+ys:y+ys + hs, x+xs:x+xs + ws] = lip
                    frame[y + ys:y + ys + hs, x + xs:x + xs + ws] = sticker_on_image(frame, lip, pos_lip)

            elif option == 4:
                # image_face = cv2.GaussianBlur(image_face, (31, 31), 30)
                image_face = cv2.resize(image_face, (w // 10, h // 10))
                image_face = cv2.resize(image_face, (w, h))
                frame[y:y + h, x:x + w] = image_face

            elif option == 5:

                width, height, ch = frame.shape
                middle = (height // 2, width // 2)
                q1 = frame[0:middle[1], 0:middle[0]]
                q2 = frame[middle[1]:, 0:middle[0]]
                q3 = frame[0:middle[1], middle[0]:]
                q4 = frame[middle[1]:, middle[0]:]
                Q = [q1, q2, q3, q4]

                if cnt % 50 == 0:
                    cnt = 1
                    selected = random.choice(range(4))
                else:
                    cnt += 1
                q_grayed = cv2.cvtColor(Q[selected], cv2.COLOR_RGB2GRAY)
                q_grayed = cv2.cvtColor(q_grayed, cv2.COLOR_GRAY2RGB)

                Q[selected] = q_grayed

                frame = cv2.vconcat([cv2.hconcat([Q[0], Q[2]]), cv2.hconcat([Q[1], Q[3]])])

        show_pic(frame, 1)
    else:
        break
