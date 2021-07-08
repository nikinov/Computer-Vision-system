import cv2

img = cv2.imread("assets/img_2.png")


cv2.imwrite("assets/img_2_dis.png", cv2.rotate(img, cv2.ROTATE_180))
