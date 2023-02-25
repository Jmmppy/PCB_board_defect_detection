#!/usr/bin/env python
# coding: utf-8

import cv2
import math
import time
import random
import numpy as np
from PIL import Image
from typing import Tuple


def show(win_name, img: np.ndarray) -> None:
    """
    :param win_name: 窗口名字
    :param img: 图片
    :return: 显示图片
    """
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, img.shape[1], img.shape[0])
    cv2.imshow(win_name, img)


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print(f'{func.__name__} time: {end - start}')
        return res
    return wrapper


class BigPic:
    
    def __init__(self):
        self.origin_im = np.array([])
        self.only_board_im = np.array([])
        self.image_path = ''  # 当前version alpha需要这个东西，后续可能不需要


class SmallPic:
    
    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col
        self.__image = np.array([])
        
    def set_image(self, img: np.array([])):
        self.__image = img

    def get_image(self) -> np.array([]):
        return self.__image


class ImageProcess:
    """图像处理类"""
    @staticmethod
    def __remove_noise(img: np.ndarray, k_gaussian=(11, 11), sigma_x=11, k_close=cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))) -> np.ndarray:
        """
        :param img: 图片
        :param k_gaussian: 高斯模糊kernel大小
        :param sigma_x: 高斯模糊sigma_x参数
        :param k_close: 形态学闭操作kernel大小
        :return: 经过降噪后的图片
        """
        img = cv2.GaussianBlur(img, k_gaussian, sigma_x)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k_close)
        return img

    @staticmethod
    def __frame_image(img: np.ndarray, thresh=200, max_val=255, thresh_type=cv2.THRESH_BINARY_INV, find_mode=cv2.RETR_EXTERNAL, find_method=cv2.CHAIN_APPROX_NONE) -> list:
        """
        :param img: 图片
        :param thresh: 起始阈值
        :param max_val: 最大值
        :param thresh_type: 阈值类型
        :param find_mode: 输出轮廓的信息的组织形式
        :param find_method: 指定轮廓的近似方法
        :return: 图片中物体的n个轮廓
        """
        ret, thresh = cv2.threshold(img, thresh, max_val, thresh_type)
        contours, hierarchy = cv2.findContours(thresh, find_mode, find_method)
        return contours

    @staticmethod
    def __ps_image(img: np.ndarray, contour) -> Tuple[np.ndarray, float, float]:
        """
        :param img: 图片
        :param contour: 一个轮廓信息
        :return: 只有物体的图片，中心点和旋转角度
        """
        rect = cv2.minAreaRect(contour)
        center = rect[0]
        angle = rect[2]
        box = cv2.boxPoints(rect)
        box = np.int64(box)

        if 0 < angle <= 45:
            angle = angle
        elif 45 < abs(angle) < 90:
            angle = angle - 90

        x1 = min(box[:, 0])
        x2 = max(box[:, 0])
        y1 = min(box[:, 1])
        y2 = max(box[:, 1])
        img = img[y1:y2, x1:x2]
        return img, center, angle

    @staticmethod
    def __change_resolution(img: np.ndarray, multiple: float) -> np.ndarray:
        height = img.shape[0]
        width = img.shape[1]
        new_height = int(height * multiple)
        new_width = int(width * multiple)
        img = cv2.resize(img, (new_width, new_height))
        return img

    @staticmethod
    def zoom_image(img: np.ndarray, longer: int) -> np.ndarray:
        """
        将大图进行缩放后显示于界面上。（因为大图往往分辨率非常大，需要进行处理后才能在界面上进行显示）
        :param img: 大图
        :param longer: 长边
        :return: 缩放之后的大图
        """
        zoom_longer = longer
        src = img
        print('src.shape', src.shape)
        img_type = len(src.shape)
        if img_type == 2:
            h, w = src.shape
        else:
            h, w, c = src.shape
        if h > w:
            longer = h
            shorter = w
        else:
            longer = w
            shorter = h
        print(longer)
        zoom_shorter = (shorter/longer)*zoom_longer
        print(zoom_shorter)
        if h > w:
            res = cv2.resize(src, (int(zoom_shorter), int(zoom_longer)))
        else:
            res = cv2.resize(src, (int(zoom_longer), int(zoom_shorter)))
        return res
    
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """
        :param image_path: 大图路径
        :return: 大图的numpy格式
        """
        Image.MAX_IMAGE_PIXELS = None
        img = np.array(Image.open(image_path))
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        print(img.shape)
        return img

    @staticmethod
    def get_only_board_image(origin_im: np.ndarray) -> list:
        """
        :param origin_im: 原始大图
        :return: 摆正后只有板的图片
        """
        multiple = 0.1
        lower_im = ImageProcess.__change_resolution(origin_im, multiple)
        if len(lower_im.shape) == 3:
            gray = cv2.cvtColor(lower_im, cv2.COLOR_BGR2GRAY)
        else:
            gray = lower_im

        clean = ImageProcess.__remove_noise(gray)
        contours = ImageProcess.__frame_image(clean)

        images = []
        print('len_contours', len(contours))
        for c in contours:
            c = np.int64(c * (1 / multiple))
            image, center, angle = ImageProcess.__ps_image(origin_im, c)
            images.append((image, center, angle))

        only_board_images = []
        for i, (image, center, angle) in enumerate(images):

            if angle < 90.0:
                height = image.shape[0]
                width = image.shape[1]
                new_height = int(width * math.fabs(math.sin(np.radians(angle))) + height * math.fabs(math.cos(np.radians(angle))))
                new_width = int(width * math.fabs(math.cos(np.radians(angle))) + height * math.fabs(math.sin(np.radians(angle))))
                m = cv2.getRotationMatrix2D(center, angle, 1)

                m[0, 2] += (new_width - width) / 2
                m[1, 2] += (new_height - height) / 2

                correct_im = cv2.warpAffine(image, m, (new_width, new_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

                lower_im = ImageProcess.__change_resolution(correct_im, multiple)
                if len(lower_im.shape) == 3:
                    gray = cv2.cvtColor(lower_im, cv2.COLOR_BGR2GRAY)
                else:
                    gray = lower_im

                clean = ImageProcess.__remove_noise(gray)
                contours = ImageProcess.__frame_image(clean)

                print('len_contours2', len(contours))
                contour = np.int64(contours[0] * (1 / multiple))
                only_board_image, center, angle = ImageProcess.__ps_image(correct_im, contour)
            else:
                only_board_image = image
            only_board_images.append(only_board_image)
        return only_board_images

    @staticmethod
    def crop_image(only_board_im: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, list]:
        """
        :param only_board_im: 摆正后只有板的图片
        :param width: 小图的宽度
        :param height: 小图的高度
        :return: 必要时经填充后的大图，小图的列表
        """
        origin_width = only_board_im.shape[1]
        origin_height = only_board_im.shape[0]

        if origin_height % height != 0:
            diff_height = height - origin_height % height
            new_height = origin_height + diff_height
            only_board_im = cv2.copyMakeBorder(only_board_im, 0, diff_height, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        else:
            new_height = origin_height
        if origin_width % width != 0:
            diff_width = width - origin_width % width
            new_width = origin_width + diff_width
            only_board_im = cv2.copyMakeBorder(only_board_im, 0, 0, 0, diff_width, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        else:
            new_width = origin_width

        col_num = new_width // width
        row_num = new_height // height
        small_images = []
        for r in range(row_num):
            rows = []
            for c in range(col_num):
                small_image = only_board_im[height * r:height * (r + 1), width * c: width * (c + 1)]
                rows.append(small_image)
            small_images.append(rows)
        return only_board_im, small_images

    @staticmethod
    @timer
    def mark(img: np.ndarray, row: int, col: int, width: int, height: int) -> np.ndarray:
        """
        :param img: 大图
        :param row: 小图的行位置
        :param col: 小图的列位置
        :param width: 小图的宽度
        :param height: 小图的高度
        :return: 标记出小图位置的大图
        """
        img = np.array(img)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        x = col * width
        y = row * height
        pt = (int(x + width * 0.5), int(y + height * 0.5))
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        length = 50
        img = cv2.line(img, (pt[0] - length, pt[1] - length), (pt[0] + length, pt[1] + length), color, 10, 16)
        img = cv2.line(img, (pt[0] + length, pt[1] - length), (pt[0] - length, pt[1] + length), color, 10, 16)
        return img


if __name__ == '__main__':
    im = ImageProcess.load_image('./data/needrotate.bmp')
    ims = ImageProcess.get_only_board_image(im)
    for i, im in enumerate(ims):
        # show(str(i), im)
        filled_im, small_ims = ImageProcess.crop_image(im, 500, 500)
        # for r, row in enumerate(small_ims):
        #     for c, pic in enumerate(row):
        #         cv2.imwrite(f'./needrotate/{r},{c}.png', pic)
        marked_im = ImageProcess.mark(filled_im, 1, 17, 500, 500)
        # show(f'mark_{i}', marked_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
