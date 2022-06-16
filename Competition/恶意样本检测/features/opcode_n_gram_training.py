# -*- coding: utf-8 -*-
"""
文 件 名: opcode_n_gram_training.py
文件描述: 训练模型
作    者: HeJian
创建日期: 2022.05.29
修改日期：2022.05.30
Copyright (c) 2022 HeJian. All rights reserved.
"""

from common import *



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









