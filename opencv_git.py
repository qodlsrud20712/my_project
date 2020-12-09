import cv2

# cv2.imread 함수를 통하여 사용할 이미지를 담는다,
# under = cv2.imread("C://Users/a/PycharmProjects/detectorn2/under.jpg")
# upper = cv2.imread("C://Users/a/PycharmProjects/detectorn2/upper.jpg")
# all = cv2.imread("C://Users/a/PycharmProjects/detectorn2/all.jpg")
# all2 = all.copy()

# cv2.cvtColor 함수를 이용해 grayscale로 변경
# gray = cv2.cvtColor(under, cv2.COLOR_RGB2GRAY)

# cv2.goodFeaturesToTrack는 코너스-해리스 알고리즘을 사용하여 코너 검출
# goodFeaturesToTrack을 쓰면 모서리 부분이 나오기 때문에 아무래도 pose 구할 때 응용 해보면 좋을듯?
# corners = cv2.goodFeaturesToTrack(gray, 100, 0.02, 5, blockSize=3, useHarrisDetector=True, k=0.03)

# eps = 0.01 * cv2.arcLength(contours3[i], True)
# # cv2.approxPolyDP는 더글라스-페커 알고리즘을 사용하여 도형 근사
# # approxPolyDP 방식이 point는 더 적게 나옴 지금 object 모양이 삼각형이라 3개 정도?
# approx = cv2.approxPolyDP(contours3[i], eps, True)

# cv2.GaussianBlur 함수로 이미지를 좀 더 뭉개서 부드럽게 해줌.
# 이미지의 노이즈를 줄여주기 위한 코드
# gray = cv2.GaussianBlur(gray, (3, 3), 0)

# cv2.Canny 알고리즘으로 Edge를 찾는다.
# gray = cv2.Canny(gray, 100, 300, apertureSize=3)

# threshold 함수로 이미지를 이진화 해줍니다.
# ret, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

# findContours를 이용하여 이진화된 이미지에서 윤곽선을 검출해냅니다.
# contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
