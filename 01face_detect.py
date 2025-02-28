import cv2

# 입력 파일 지정
image_file = "./resData/photo1.jpg"
# 캐스케이드 파일의 경로 지정
cascade_file = cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
print('cascade_file', cascade_file)
# 이미지 읽기
image = cv2.imread(image_file)
# 그레이스케일로 변환
image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 얼굴인식을 위한 특징 파일 로드
cascade = cv2.CascadeClassifier(cascade_file)
# 얼굴 인식 실행
face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=1, minSize=(150, 150))
if len(face_list) > 0:
    # 인식한 부분 표시
    print(face_list)
    color = (0, 0, 255)  # 빨간색
    # 얼굴 영역에 테두리 표시
    for face in face_list:
        x, y, w, h = face
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness=8)
    # 파일로 출력
    cv2.imwrite("./saveFiles/photol-facedetect01.png", image)
else:
    print("얼굴을 인식할 수 없습니다")