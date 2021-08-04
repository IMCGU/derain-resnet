import cv2

image = cv2.imread('C:\\Users\\HanHan\\wavelet_derain-test\\derain-test\\img\\129.jpg')
image = cv2.resize(image, (852, 476), interpolation=cv2.INTER_AREA)
cv2.imshow('Result', image)
cv2.imwrite('C:\\Users\\HanHan\\wavelet_derain-test\\derain-test\\img\\129.jpg', image)
cv2.waitKey(0)