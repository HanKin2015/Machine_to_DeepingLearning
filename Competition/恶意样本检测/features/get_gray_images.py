# -*- coding: utf-8 -*-
"""
文 件 名: get_gray_images.py
文件描述: 获取二进制文件的灰度图像
备    注: 安装capstone库(pip install capstone)
作    者: HeJian
创建日期: 2022.06.14
修改日期：2022.06.14
Copyright (c) 2022 HeJian. All rights reserved.
"""

import subprocess
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split
import time
from log import logger
import pandas as pd
from PIL import Image
import binascii
import pefile
from capstone import *
import re
from collections import *
from concurrent.futures import ThreadPoolExecutor

SAMPLE_PATH      = './AIFirst_data/'
TRAIN_WHITE_PATH = SAMPLE_PATH+'train/white/' # 训练集白样本路径
TRAIN_BLACK_PATH = SAMPLE_PATH+'train/black/' # 训练集黑样本路径
TEST_PATH        = SAMPLE_PATH+'test/'        # 测试集样本路径
DATA_PATH        = './data/'                  # 数据路径
TRAIN_WHITE_STRING_FEATURES_PATH = DATA_PATH+'train_white_string_features.csv' # 训练集白样本字符串特征数据集路径
TRAIN_BLACK_STRING_FEATURES_PATH = DATA_PATH+'train_black_string_features.csv' # 训练集黑样本字符串特征数据集路径
TEST_STRING_FEATURES_PATH        = DATA_PATH+'test_string_features.csv'        # 测试集样本数字符串特征据集路径
TRAIN_WHITE_CUSTOM_STRINGS_PATH  = DATA_PATH+'train_white_strings.csv'         # 训练集白样本自定义字符串数据集路径
TRAIN_BLACK_CUSTOM_STRINGS_PATH  = DATA_PATH+'train_black_strings.csv'         # 训练集黑样本自定义字符串数据集路径
TEST_CUSTOM_STRINGS_PATH         = DATA_PATH+'test_strings.csv'                # 测试集样本自定义字符串数据集路径

TRAIN_WHITE_GRAY_IMAGES_PATH = './gray_images/train/white/'
TRAIN_BLACK_GRAY_IMAGES_PATH = './gray_images/train/black/'
TEST_GRAY_IMAGES_PATH        = './gray_images/test/'

# 线程数量
THREAD_NUM = 64

def get_image_width(file_path):
    """获取图像宽度
    根据论文《Malware Images: Visualization and Automatic Classification》
    文件大小KB
    """

    file_size = os.path.getsize(file_path) / 1024
    logger.debug('file size: {} KB.'.format(file_size))
    
    if file_size < 10: return 32
    elif file_size < 30: return 64
    elif file_size < 60: return 128
    elif file_size < 100: return 256
    elif file_size < 200: return 384
    elif file_size < 500: return 512
    elif file_size < 1000: return 768
    return 1024

def binary_file_to_grayscale_image(root, file, save_path):
    """恶意样本灰度图像绘制
    """
    
    file_path = '{}{}'.format(root, file)
    image_width = get_image_width(file_path)
    logger.debug('file[{}] grayscale image width: {}.'.format(file, image_width))

    with open(file_path, 'rb') as fd:
        content = fd.read()
    hexst = binascii.hexlify(content)  #将二进制文件转换为十六进制字符串
    file_bytes = np.array([int(hexst[i:i+2],16) for i in range(0, len(hexst), 2)])  #按字节分割
    image_hight = int(len(file_bytes)/image_width)
    logger.debug('grayscale image size: {}x{}.'.format(image_width, image_hight))
    
    matrix = np.reshape(file_bytes[:image_hight*image_width], (-1, image_width))  #根据设定的宽度生成矩阵
    matrix = np.uint8(matrix)
    
    im = Image.fromarray(matrix)    #转换为图像
    im.save('{}{}.png'.format(save_path, file))

def gray_image_progressing(data_path, save_path):
    """灰度图像处理
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    count = 0
    with ThreadPoolExecutor(max_workers=THREAD_NUM) as pool:
        for root, dirs, files in os.walk(data_path):
            for file in files:
                pool.submit(binary_file_to_grayscale_image, root, file, save_path)
                count += 1
    
    logger.info('directory[{}] transform {} binary files to gray images done.'.format(data_path, count))

def main():
    gray_image_progressing(TRAIN_WHITE_PATH, TRAIN_WHITE_GRAY_IMAGES_PATH)
    gray_image_progressing(TRAIN_BLACK_PATH, TRAIN_BLACK_GRAY_IMAGES_PATH)
    gray_image_progressing(TEST_PATH, TEST_GRAY_IMAGES_PATH)

def debug():
    """调试
    """
    
    root = 'D:\\Github\\Machine_to_DeepingLearning\\Competition\\恶意样本检测\\features\\backup\\'
    file = 'FTOOL.exe'
    #binary_file_to_grayscale_image(root, file)
    
if __name__ == '__main__':
    start_time = time.time()

    main()

    end_time = time.time()
    logger.info('process spend {} s.'.format(round(end_time - start_time, 3)))











