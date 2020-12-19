import cv2
import numpy as np
import os
from glob import glob
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

# 사진파일이 담겨져있는 경로를 지정해줌 절대경로가 좋을듯
# p = "C:/Users/a/Downloads/train/train/"
#
# #폴더 안에 모든 파일 목록을 읽음
# files = glob(os.path.join(p, '*.jpg'))
#
# # print(files)
# #
# len_img = len(files)
#
# print(len_img)
#
# # 데이터 집합을 만드는데 사용할 셔플 색인 생성, 무작위로 생성된 숫자들
# 이걸 이용해서 숫자들에 대응하는 사진들을 복사하기위해 이렇게 써줌.
# shuffle = np.random.permutation(len_img)
#
# # print(shuffle)
# #검증 이미지를 저장할 검증용 디렉터리 생성
# os.mkdir(os.path.join(p, 'valid'))
# os.mkdir(os.path.join(p, 'train'))
# #

# # 레이블명으로 디렉터리 생성
# for t in ['train', 'valid']:
#     for folder in ['dog/', 'cat/']:
#         os.mkdir(os.path.join(p, t, folder))
#
# # print(files[0].split('\\')[-1])
# # valid 폴더에 이미지 2000장 복사
# for i in shuffle[:2000]:
#     folder = files[i].split('\\')[-1].split('.')[0]
#     img = files[i].split('\\')[-1]
#     try:
#         os.rename(files[i], os.path.join(p, 'valid', folder, img))
#     except FileNotFoundError:
#         print("파일을 못찾았으므로 패스")
#         pass

# # train 폴더에 나머지 이미지 복사
#
# print(files[0])
# print(files[0].split('\\')[-1])
# print(files[0].split('\\')[-1].split('.')[0])
# for i in shuffle[2000:]:
#     folder = files[i].split('\\')[-1].split('.')[0]
#     img = files[i].split('\\')[-1]
#     try:
#         os.rename(files[i], os.path.join(p, 'train', folder, img))
#     except FileNotFoundError:
#         print("파일을 못찾았으므로 패스")
#         pass

# 데이터 전처리를 해줘야됨. 1) 이미지 크기 동일화 2) 데이터셋 정규화 3) tensor로 변환시키기
simple_transform = transforms.Compose([transforms.Scale((224,224)), transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456,0.406],[0.229,0.224,0.225])])
# p = "C:/Users/a/Downloads/train/train/dog"

train = ImageFolder("C:/Users/a/Downloads/train/train/", simple_transform)
valid = ImageFolder("C:/Users/a/Downloads/train/valid/", simple_transform)

print(train)
print(valid)

# tensor 객체를 시각화 하기 위한 함수
# tensor -> numpy 변환 후 데이터 형상 재구성 후 역정규화 해줘야함.
def Tensor_imshow(inp):
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.show()
    # cv2.imshow("Dd",inp)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    # print(train[50][0])
    Tensor_imshow(train[50][0])
