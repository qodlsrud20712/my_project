import math
import numpy as np
import cv2

#이미지상에서 pixel point들에 대응하는 3d(x,y,z)point 모음

worldPoints= np.array([[0.028103701770305634, -0.10956190526485443, 0.45900002121925354],
                      [0.028667885810136795, -0.11864567548036575, 0.4620000123977661],
                      [0.038981154561042786, -0.11837255209684372, 0.4580000340938568],
                      [0.03956202417612076, -0.11297532171010971, 0.45600003004074097],
                      [0.05497755855321884, -0.11297067999839783, 0.45000001788139343],
                      [0.056710921227931976, -0.10603044927120209, 0.4520000219345093],
                      [0.03967481479048729, -0.10402738302946091, 0.453000009059906],
                      [0.03987109288573265, -0.09948388487100601, 0.45100003480911255],
                      [0.028859740123152733, -0.10011492669582367, 0.45900002121925354]], dtype=np.float32)

print(worldPoints)

#이미지상에서 pixel point들의 모음
imagePoints= np.array([[352.4568527918782, 92.38578680203045],
                      [352.96446700507613, 81.21827411167513],
                      [367.1776649746193, 80.20304568527919],
                      [368.1928934010152, 86.80203045685279],
                      [390.02030456852793, 84.77157360406092],
                      [392.0507614213198, 94.9238578680203],
                      [368.7005076142132, 97.96954314720813],
                      [369.20812182741116, 103.55329949238579],
                      [353.4720812182741, 105.0761421319797]], dtype=np.float32)

print(imagePoints)

# img = cv2.imread(r"C:\Users\a\PycharmProjects\lr_1\base\many_2.jpg")

#카메라의 내부 파라미터
# [fx, 0, cx
#  0, fy, cy,
#  0,0,1]

# 여기서 fx,fy는 초점거리(focal length), cx,cy는 주점(principal point)
# realsense d435i를 사용함.
# real_matrix = np.array([[616.358, 0., 314.718],
#                        [0., 616.358, 239.563],
#                        [0.,0.,1.]], dtype=np.float32)
#
# dist = np.zeros((4,1))
#
# print(real_matrix)
# print(type(real_matrix))
# print(dist)
#
# def draw(img, corners, imgpts):
#     corner = tuple(corners[0].ravel())
#     print("corn", corner)
#     print(imgpts[0].ravel())
#     x, y = corner
#     x2, y2 = imgpts[0].ravel()
#     test_px = int((x+x2)/2)
#     test_py = int((y+y2)/2)
#
#     print(test_px,test_py)

#     # x = x/100000
#     # y = y/10
#     # test = [int(x),int(y)]
#     # print(test)
#     # print("imgpts[0]", imgpts[1].ravel())
#     # img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 3)
#     cv2.circle(img, corner, 4, (0,0,255), cv2.FILLED)
#     # img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 3)
#     # img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 3)
#     return img
#
# axis = np.float32([[500,0,0], [0,500,0], [0,0,500]])
# print("solvePNP")

# solvePnP를 쓰기 위해선 2D,3D 각각 대응되는 6개의 point가 필요.
# rvec = rotation vector, tvec = translation vector
# ret,rvec1,tvec1 = cv2.solvePnP(worldPoints,imagePoints,real_matrix,dist)

# projectPoints는 고유한 값들이 주어지면
# 이미지 평면에서 3D, 2D 대응되는 어떤 연산을 해주는듯
# imgpts, jac = cv2.projectPoints(axis, rvec1, tvec1, real_matrix, dist)
# modelpts, jac2 = cv2.projectPoints(worldPoints, rvec1, tvec1,real_matrix, dist)

# print(ret)
# print("rvec : ", rvec1)
# print("tvec : ", tvec1)

# Rodrigues는 회전벡터를 넣어주면 회전매트릭스를 만들어줌.
# rotM = cv2.Rodrigues(rvec1)[0]
# print(rotM)

# 여기서 np.hstack으로 올바른 행렬을 만들어줌.
# proj_matrix = np.hstack((rotM, tvec1))
# print(proj_matrix)
# print(type(proj_matrix))

# 여기서 투영행렬을 분해하여 각 축에 대한 오일러 각도를 리턴해줌.
# eulerang = cv2.decomposeProjectionMatrix(proj_matrix)[6]
# print(eulerang)
# # print(rotM)

# pitch = rx, yaw = ry, roll = rz 실제로 쓰기 위해 밑에서의 연산이 필요함.
# pitch, yaw, roll = [math.radians(_) for _ in eulerang]
# pitch = math.degrees(math.asin(math.sin(pitch)))
# roll = -math.degrees(math.asin(math.sin(roll)))
# yaw = math.degrees(math.asin(math.sin(yaw)))
#
# print("pitch", pitch)
# print("roll", roll)
# print("yaw", yaw)