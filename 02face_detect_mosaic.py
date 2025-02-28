import cv2
import sys
import re
#입력 파일 지정 
if len(sys.argv) <= 1:
	print("no input file")
	quit()
image_file = sys.argv[1]
#출력 파일 이름
output_file = re.sub(r'\.jpg|jpeg|PNG$', '-mosaic1.jpg', image_file)
print("output", output_file)
#모자이크 강도
mosaic_rate = 30
#캐스캐이드 파일 경로 지정
cascade_file = cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
#이미지를 Numpy 배열로 변환
image = cv2.imread(image_file)
#그레이스케일
image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#얼굴 인식 실행하기
cascade = cv2.CascadeClassifier(cascade_file)
face_list = cascade.detectMultiScale(image_gs,
scaleFactor=1.1, minNeighbors=1, minSize=(100,100))
#얼굴이 감지되지 않으면 프로그램 종료
if len(face_list) == 0:
	print("얼굴을 인식할 수 없습니다.")
	quit()
print(face_list)