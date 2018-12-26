import cv2
import numpy as np


drawing = False  # 是否开始画图
start = (-1, -1)

def mouse_event(event, x, y, flags, param):
    global start, drawing, mode

    # 左键按下：开始画图
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start = (x, y)
    # 鼠标移动，画图
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(img, (x, y), 20, (255, 255, 255), -1)
    # 左键释放：结束画图
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img, (x, y), 20, (255, 255, 255), -1)


img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_event)

while(True):
    cv2.imshow('image', img)
    if cv2.waitKey(1) == 27:
        break

cv2.imwrite('C:\\Users\\sdfj\\Desktop\\testyy\\1.png',img)
tempimg = cv2.resize(img,(28,28),3)
cv2.imwrite('C:\\Users\\sdfj\\Desktop\\testyy\\2.png',tempimg)
