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

SAMPLE_PATH                      = './AIFirst_data/'                           # 样本数据集存储路径
TRAIN_WHITE_PATH                 = SAMPLE_PATH+'train/white/'                  # 训练集白样本路径
TRAIN_BLACK_PATH                 = SAMPLE_PATH+'train/black/'                  # 训练集黑样本路径
TEST_PATH                        = SAMPLE_PATH+'test/'                         # 测试集样本路径
DATA_PATH                        = './data/'                                   # 数据路径
TRAIN_WHITE_GRAY_IMAGES_PATH     = './gray_images/train/white/'                # 训练集白样本灰度图像存储路径
TRAIN_BLACK_GRAY_IMAGES_PATH     = './gray_images/train/black/'                # 训练集黑样本灰度图像存储路径
TEST_GRAY_IMAGES_PATH            = './gray_images/test/'                       # 测试集样本灰度图像存储路径
TRAIN_WHITE_IMAGE_MATRIX_PATH    = DATA_PATH+'train_white_image_matrix.csv'    # 训练集白样本图像矩阵数据集存储路径
TRAIN_BLACK_IMAGE_MATRIX_PATH    = DATA_PATH+'train_black_image_matrix.csv'    # 训练集黑样本图像矩阵数据集存储路径
TEST_IMAGE_MATRIX_PATH           = DATA_PATH+'test_image_matrix.csv'           # 测试集样本图像矩阵数据集存储路径

TRAIN_BLACK_0_3000_IMAGE_MATRIX_PATH = DATA_PATH+'train_black_0_3000_image_matrix.csv'    # 训练集黑样本图像矩阵数据集存储路径
TRAIN_BLACK_3000_IMAGE_MATRIX_PATH   = DATA_PATH+'train_black_3000_image_matrix.csv'    # 训练集黑样本图像矩阵数据集存储路径
TEST_0_3000_IMAGE_MATRIX_PATH        = DATA_PATH+'test_0_3000_image_matrix.csv'          # 测试集样本操作指令码3-gram特征存储路径
TEST_3000_6000_IMAGE_MATRIX_PATH     = DATA_PATH+'test_3000_6000_image_matrix.csv'
TEST_6000_IMAGE_MATRIX_PATH          = DATA_PATH+'test_6000_image_matrix.csv'          # 测试集样本操作指令码3-gram特征存储路径

# 线程数量
THREAD_NUM = 64

def get_image_width(file_path):
    """获取图像宽度
    根据论文《Malware Images: Visualization and Automatic Classification》
    文件大小KB
    """

    file_size = os.path.getsize(file_path) / 1024
    logger.debug('file size: {} KB.'.format(file_size))
    
    file_size_array = [10, 30, 60, 100, 200, 500, 1000]
    image_width_array = [32, 64, 128, 256, 384, 512, 768]
    for i in range(7):
        if file_size < file_size_array[i]:
            return image_width_array[i]
    return 1024

def get_image_matrix_from_binary_file(root, file, is_onerow=True):
    """从二进制文件获取图像矩阵
    """
    
    file_path = '{}{}'.format(root, file)

    with open(file_path, 'rb') as fd:
        content = fd.read()
    
    # 将二进制文件转换为十六进制字符串
    hexst = binascii.hexlify(content)
    # 按字节分割
    image_matrix = np.array([int(hexst[i:i+2],16) for i in range(0, len(hexst), 2)])
    # 根据设定的宽度生成矩阵
    if is_onerow is False:
        image_width = get_image_width(file_path)
        logger.debug('file[{}] grayscale image width: {}.'.format(file, image_width))
        image_hight = int(len(image_matrix)/image_width)
        logger.debug('grayscale image size: {}x{}.'.format(image_width, image_hight))
        image_matrix = np.reshape(image_matrix[:image_hight*image_width], (-1, image_width))
    image_matrix = np.uint8(image_matrix)
    
    return file, image_matrix

def save_to_csv(mapimg, save_path):
    """保存数据到csv文件
    """
    
    dataframelist = []
    for file_name, image_matrix in mapimg.items():
        logger.debug('file_name: {}. image_matrix len: {}.'.format(file_name, len(image_matrix)))
        standard = {}
        standard['FileName'] = file_name
        for key, value in enumerate(image_matrix):
            column_name = "pix{0}".format(str(key))
            logger.debug('key: {}.'.format(key))
            standard[column_name] = value
        dataframelist.append(standard)

    df = pd.DataFrame(dataframelist)
    logger.info('{} shape: ({}, {}).'.format(save_path, df.shape[0], df.shape[1]))
    df.to_csv(save_path, index=False)
    
def image_matrix_progressing(data_path, save_path, file_index_start=0, file_index_end=-1):
    """图像矩阵处理
    """

    mapimg = defaultdict(list)

    tasks = []
    file_index = 0
    is_progressing = False
    with ThreadPoolExecutor(max_workers=THREAD_NUM) as pool:
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file_index == file_index_start:
                   is_progressing = True
                
                if is_progressing:
                    task = pool.submit(get_image_matrix_from_binary_file, root, file)
                    tasks.append(task)
                
                file_index += 1
                if file_index == file_index_end:
                    is_progressing = False
    
    for task in tasks:
        file_name, image_matrix = task.result()
        logger.debug('{} : {}.'.format(file_name, image_matrix))
        mapimg[file_name] = image_matrix
        logger.debug('image_matrix: {}.'.format(image_matrix))
    logger.info('{} get image matrix done.'.format(data_path))
    save_to_csv(mapimg, save_path)

def main():
    #image_matrix_progressing(TRAIN_WHITE_PATH, TRAIN_WHITE_IMAGE_MATRIX_PATH)
    #image_matrix_progressing(TRAIN_BLACK_PATH, TRAIN_BLACK_IMAGE_MATRIX_PATH)
    #image_matrix_progressing(TEST_PATH, TEST_IMAGE_MATRIX_PATH)
    
    # 数据分段解析处理
    image_matrix_progressing(TRAIN_BLACK_PATH, TRAIN_BLACK_0_3000_IMAGE_MATRIX_PATH, 0, 3000)
    image_matrix_progressing(TRAIN_BLACK_PATH, TRAIN_BLACK_3000_IMAGE_MATRIX_PATH, 3000)
    image_matrix_progressing(TEST_PATH, TEST_0_3000_IMAGE_MATRIX_PATH, 0, 3000)
    image_matrix_progressing(TEST_PATH, TEST_3000_6000_IMAGE_MATRIX_PATH, 3000, 6000)
    image_matrix_progressing(TEST_PATH, TEST_6000_IMAGE_MATRIX_PATH, 6000)

def debug():
    """调试
    """
    
    root = 'D:\\Github\\Machine_to_DeepingLearning\\Competition\\恶意样本检测\\features\\AIFirst_data\\test\\'
    file = 'FTOOL.exe'
    file, image_matrix = get_image_matrix_from_binary_file(root, file)
    mapimg = defaultdict(list)
    mapimg[file] = image_matrix
    logger.info('image_matrix: {}.'.format(image_matrix))
    save_to_csv(mapimg, TRAIN_WHITE_IMAGE_MATRIX_PATH)
    
if __name__ == '__main__':
    start_time = time.time()

    main()
    #debug()

    end_time = time.time()
    logger.info('process spend {} s.'.format(round(end_time - start_time, 3)))











