import cv2
import os
from detector import Detector

path = r"/home/zhy/yolov5-6.0/yolov5-6.0/123"
if __name__ == '__main__':
    #self.detector = D。。（）
    detector = Detector()
    imglist1 = []
    #
    for file in os.listdir(path):
        print(path+os.sep+file)
        res,image = detector.infer(cv2.imread(path+os.sep+file))
        #res, image = detector.infer(list[i])
        print(res)
        imglist1.append(res)
        cv2.imshow('dct',image)
        cv2.waitKey(0)  # 1 millisecond


