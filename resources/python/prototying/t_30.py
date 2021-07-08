import cv2 as cv
import numpy as np


"""
1. import images
2. Convert to gray
3. Init ORB detector
4. Find key points and describe them
5. Match key points - brute force matcher
6. RANSAC (reject bad key points)
7. Register two images (use homology
"""

def preprocess(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    orb = cv.ORB_create(50)
    kp, des = orb.detectAndCompute(img, None)
    return kp, des

img = cv.imread("assets/img_2.png")
img_dis = cv.imread("assets/img_2_dis.png")

matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

kp1, des1 = preprocess(img)
kp2, des2 = preprocess(img_dis)

matches = matcher.match(des1, des2, None)

matches = sorted(matches, key=lambda x:x.distance)

points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = kp1[match.queryIdx].pt
    points2[i, :] = kp2[match.trainIdx].pt

h, mask = cv.findHomography(points1, points2, cv.RANSAC)

# use homography

hight, width, chanels = img.shape

imReg = cv.warpPerspective(img, h, (width, hight))

img = cv.drawMatches(img, kp1, img_dis, kp2, matches[:10], None)

cv.imshow("matches", img)
cv.imshow("Registered image", imReg)
cv.waitKey(0)
