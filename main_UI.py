import sys

from PyQt5.QtWidgets import (QWidget, QToolTip, QPushButton, QApplication, QMessageBox, QDesktopWidget, \
    QMainWindow, QHBoxLayout, QVBoxLayout)

from UI.detect_page import DetectPage
from AVIutils._global_commu import glo

# 整体大块用QWidget，里面包含各种组件
# 函数名-->驼峰         变量-->abc_def
SOFT_VERSION = 'Version: α'


class MainUI(QMainWindow):

    def __init__(self):
        super().__init__()
        # QMainWindow有自己的布局方式
        # QMainWindow中以QWidget作为主框架，利于VBox/HBox布局
        self.mainWidget = QWidget(self)
        self.setCentralWidget(self.mainWidget)

        self.statusBar = self.statusBar()
        glo.set_value('statusBar', self.statusBar)
        self.statusBar.showMessage(SOFT_VERSION)
        self.initUI()

    def initUI(self):
        main_vlayout = QVBoxLayout()

        # “测试”页面
        main_vlayout.addWidget(DetectPage())  # , 20, Qt.AlignmentFlag.AlignTop)

        self.mainWidget.setLayout(main_vlayout)
        # self.resize(700, 500)
        # self.center()
        self.setWindowTitle("AVI")

    # 将窗口放在屏幕中心
    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move(int((screen.width() - size.width()) / 2),
                  int((screen.height() - size.height()) / 2))


if __name__ == "__main__":
    glo._init()

    app = QApplication(sys.argv)
    ex = MainUI()
    ex.showMaximized()
    ex.show()
    sys.exit(app.exec_())
