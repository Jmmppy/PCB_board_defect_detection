from cProfile import label
import sys, os

from numpy import ndarray
# 自定义模块
sys.path.append('..')
sys.path.append('../yolov5')
from AVIutils._global_commu import glo
from UI import thread_class

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QPixmap, QImage

from PyQt5.QtWidgets import (QWidget, QToolTip, QPushButton, QApplication, QMessageBox, QDesktopWidget, \
    QMainWindow, QAction, QMenu, QTextEdit, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, \
    QInputDialog, QLineEdit, QFrame, QComboBox, QFileDialog, QScrollArea)


# “检测”页面
class DetectPage(QWidget):
    ''' '检测'页面 '''
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        hlayout = QHBoxLayout(self)
        hlayout.addWidget(InfoWidget(), 1)
        # hlayout.addSpacing(10)
        front_big = DefectDisplayWidget()
        hlayout.addWidget(front_big, 8)

        self.setLayout(hlayout)
        # self.resize()

        # front 用于InfoWidget类中展示大小图, 以及imgprocess中获取宽
        glo.set_value("front_big", front_big)
    

class InfoWidget(QWidget):
    '''左侧信息栏，料号，批号等以及总控制按钮'''
    def __init__(self):
        super().__init__()
        self.initUI()
        self.setObjectName("InfoWidget")
        self.is_front = True
    
    def initUI(self):
        vlayout = QVBoxLayout(self)
        # vlayout.addSpacing(10)

        # 控制按钮 and 当前片
        vlayout.addWidget(self.controlBtns(), 8)

        vlayout.addStretch(4)

        self.setLayout(vlayout)
        # self.resize(self.sizeHint())
    
    def controlBtns(self) -> QWidget:
        '''总控按钮'''
        control_btns_widget = QWidget(self)
        vlayout = QVBoxLayout()

        import_img = QPushButton("导入大图") 
        import_img.clicked.connect(self.openFile)

        vlayout.addWidget(import_img, 1)

        control_btns_widget.setLayout(vlayout)
        # control_btns_widget.resize(control_btns_widget.sizeHint())
        return control_btns_widget

    def openFile(self):
        '''打开文件 槽函数'''

        selectedFile = QFileDialog.getOpenFileName(self,
                "Select one file to open",
                "../")
        print(selectedFile[0])
        if not selectedFile[0]:
            return
        # return
        '''
        yolo 缺陷检测
        '''
        self.big_img_org = selectedFile[0]
        self.yolo_thread = thread_class.YoloDetect([selectedFile[0]], 'model/AVI_yolo.pt', 512)

        self.yolo_thread.slist_signal.connect(self.deliverImg)
        self.yolo_thread.finished.connect(self.yolo_thread.deleteLater)

        self.yolo_thread.start()

    def deliverImg(self, small_detect: list=[]):
        '''将大图 小图展示到软件上'''
        # 修改front大图
        someend_big = glo.get_value("front_big")
        someend_big.changeBigImg(self.big_img_org, True)

        # someend_big.changeSmallImg(small_detect)


class DefectDisplayWidget(QWidget):

    big_img_label = None    # 大图标签
    small_scroll = None     # 小图滚动框

    '''显示大图原图、缺陷小图、以及相关信息'''
    def __init__(self, front=True):
        super().__init__()
        self.front = front

        self.setObjectName("DefectDisplayWidget")
        self.setStyleSheet("QWidget#DefectDisplayWidget{background-color: cyan;}")
        self.initUI()
    
    def initUI(self):
        grid = QHBoxLayout()

        self.big_img_label = QLabel()
        # self.big_img_label.setStyleSheet("QLabel{background: #ccc}")
        self.big_img_label.setMaximumHeight(1000)
        grid.addWidget(self.big_img_label)
        
        # 可滚动gird layout，显示小图
        # self.small_scroll = QScrollArea(self)   # self 成员变量
        # self.small_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # self.small_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # self.changeSmallImg()

        # grid.addWidget(self.small_scroll)

        self.setLayout(grid)

    # 更换大图
    def changeBigImg(self, big_img_org, is_org: bool=False):
        '''更换大图  is_org是否是原图'''
        if is_org:
            self.big_img_org = big_img_org  # 保存大图原图

        # big_img_org是路径
        # zoom大图
        self.big_zoom_thread = thread_class.BigZoom(big_img_org, 700)
        self.big_zoom_thread.big_zoom.connect(self.big_img_label.setPixmap)
        self.big_zoom_thread.finished.connect(self.big_zoom_thread.deleteLater)

        self.big_zoom_thread.start()
    
    # 改变小图
    def changeSmallImg(self, small_imgs:list =[]): #, scale_factor: float=1):
        '''改变小图'''

        t_widget = QWidget()
        t_grid = QGridLayout()
       
        for i, img in enumerate(small_imgs):
            t_lab = QLabel(img)
            t_lab.setPixmap(QPixmap(QImage(img)).scaled(270, 270, Qt.AspectRatioMode.KeepAspectRatio))
            # t_lab.setScaledContents(True)
            t_grid.addWidget(t_lab, int(i / 3) * 2, (i % 3) * 2, 2, 2)

        t_widget.setLayout(t_grid)
        self.small_scroll.setWidget(t_widget)



if __name__=="__main__":
    glo._init()
    
    app = QApplication(sys.argv)
    ex = DetectPage()
    ex.showMaximized()
    ex.show()
    
    sys.exit(app.exec_())
