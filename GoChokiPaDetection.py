import os
import datetime
import math
import cv2
import numpy as np
import mediapipe as mp
import time
from pycaret.classification import *
import pickle

### https://www.youtube.com/watch?v=9iEPzbG-xLE

class handDetector():
    # def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        zList = []
        bbox = ()
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                # cx, cy = int(lm.x * w), int(lm.y * h)
                cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                xList.append(cx)
                yList.append(cy)
                zList.append(cz)
                # print(id, cx, cy)
                lmList.append([id, cx, cy, cz])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            zmin, zmax = min(zList), max(zList)
            bbox = xmin, ymin, zmin, xmax, ymax, zmax
            if draw:
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[3], bbox[4]), 
                                (0, 255, 0), thickness = 2)

        return lmList, bbox


# 手首の座標を原点とし、20点との距離を求める。
def get_distance(wklst):
    lst_o = list([wklst[0][1], wklst[0][2], wklst[0][3]])      # 手首のx,y,z
    lst_dist = []
    for i in range(1,21):
        lst_t = list([wklst[i][1], wklst[i][2], wklst[i][3]])  # 手首以外の20箇所のx,y,z
        length = math.sqrt(sum((px - qx) ** 2.0 for px, qx in zip(lst_o, lst_t)))
        lst_dist.append(length)    # 手首の座標からの20個の距離
        # print(lst_t, length)
    # print(lst_dist)
    lst_dist.append(0)
    # lst_dist.append(lst_dist)
    return lst_dist

def main():
    pTime = 0
    cTime = 0

    myPath = os.path.join(os.getcwd(), 'pose_data_2')

    result_dict = {0: 'Go-', 1: 'Choki', 2: 'Paa-'}
    ridge_cls = load_model('Final_Ridge_Model')
    # ridge_cls = load_model('Final_QDA_Model')
    # print(ridge_cls)

    clmn_lst = [str(i) for i in range(1,21)]
    clmn_lst.append('cls')
    df = pd.DataFrame(columns=clmn_lst)

    cap = cv2.VideoCapture(0)
    wCam, hCam = 640, 480
    cap.set(3, wCam)    # 3: CV_CAP_PROP_FRAME_WIDTH
    cap.set(4, hCam)    # 4: CV_CAP_PROP_FRAME_HEIGHT

    detector = handDetector(maxHands=1, detectionCon=0.8, trackCon=0.8)

    lmList_save = []
    wkCounter = 0

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            dist_lst = get_distance(lmList)
            df.loc[0] = dist_lst
            df_pred = predict_model(ridge_cls, data=df)

            cv2.putText(img, "You:" + result_dict[df_pred.loc[0, 'Label']], 
                        (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

            # Training dataset収集用　100個溜まったら書き出す。
            # ポーズごとにこのmain()を実行する。
            # lmList_save.append(lmList)
            # if wkCounter > 100:
            #     wkPath = os.path.join(myPath, 'pose_{0}.dat'.format(
            #                         datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
            #     with open(wkPath, 'wb') as f:
            #         pickle.dump(lmList_save, f)
            #     lmList_save = []
            #     wkCounter = 0
            # wkCounter += 1

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 100), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
    