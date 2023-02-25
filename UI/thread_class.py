import sys, os, qimage2ndarray, json
sys.path.append('..')
sys.path.append('../yolov5')

from yolov5.detector import Detector
from AVIutils.ImageProcess import *
from AVIutils._global_commu import glo
from time import sleep
import cv2
from numpy import ndarray

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import QLabel


class YoloDetect(QThread):
    '''
    Yolov5缺陷检测

    Args:
        small_list: 小图路径list
        weights: 权重路径
        imgsz: 归一化大小
        classes: 分类个数
        conf_thres: 置信度阈值
        iou_thres: iou阈值

    Returns:
        result, image: 缺陷坐标[x1, y1, x2, y2], 图片变量
    '''
    slist_signal = pyqtSignal(list)
    def __init__(self, small_list: list,
                weights=r"model/AVI_yolo.pt",
                imgsz=512,
                classes=None,
                conf_thres=0.45,
                iou_thres=0
                ):
        super().__init__()
        self.small_list = small_list
        # 保存信息
        self.save_path = 'Detect/'

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.detector = Detector(conf_thres, iou_thres, classes, imgsz, weights)
        print('detectorrrr')

    def run(self):
        print("yolo start")
        imglist1 = []   # 带标签的小图

        timg = None # 展示图片
        for i, img_path in enumerate(self.small_list):
            self.save_path = self.save_path + 'yolo' + str(img_path).split('/')[-1].split('.')[0]
            print(self.save_path)
            img = ImageProcess.load_image(img_path)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            res, img = self.detector.infer(img)   # res = [[x1, y1, x2, y2], [x1, y1, x2, y2]]

            # 保存图片
            cv2.imwrite(f'{self.save_path}.png', img)
            timg = img

            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = qimage2ndarray.array2qimage(image)
            imglist1.append(img)

            s_detect_list = []
            for j in res:
                s_detect = {'x1': j[0], 'y1': j[1], 'x2': j[2], 'y2': j[3]}
                s_detect_list.append(s_detect)

            x = glo.get_value('statusBar')
            glo.get_value('statusBar').showMessage('没有缺陷')
            if len(s_detect_list) > 0:
                glo.get_value('statusBar').showMessage(json.dumps(s_detect_list))

        self.slist_signal.emit(imglist1)
        print("yolo end")

        show("detect img", timg)  # 用opencv展示图片
        cv2.waitKey()


class BigZoom(QThread):
    '''
    缩放大图

    Args:
        big_img_org: 大图原图
        label_width: 大图标签的宽

    Signals:
        big_zoom: 缩放过的大图
    '''
    big_zoom = pyqtSignal(QPixmap)

    def __init__(self, big_img_org: ndarray, label_width: int):
        super().__init__()
        self.big_img_org = ImageProcess.load_image(big_img_org)
        self.lwidth = label_width

    def run(self):
        bz = ImageProcess.zoom_image(self.big_img_org, self.lwidth)
        bz = cv2.cvtColor(bz, cv2.COLOR_BGR2RGB)    # 假设BigZoom是显示图像之前的最后一个操作
        bz = qimage2ndarray.array2qimage(bz)
        bz = QPixmap(QImage(bz))

        self.big_zoom.emit(bz)


