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

from common import *

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

def gray_image_progressing(data_path, save_path, file_index_start=0, file_index_end=-1):
    """灰度图像处理
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_index = 0
    is_progressing = False
    with ThreadPoolExecutor(max_workers=THREAD_NUM) as pool:
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file_index == file_index_start:
                   is_progressing = True
                
                if is_progressing:
                    pool.submit(binary_file_to_grayscale_image, root, file, save_path)
                
                file_index += 1
                if file_index == file_index_end:
                    is_progressing = False
    
    logger.info('directory[{}] transform {} binary files to gray images done[{}, {}].'.format(data_path, file_index, file_index_start, file_index_end))

def main():
    #gray_image_progressing(TRAIN_WHITE_PATH, TRAIN_WHITE_GRAY_IMAGES_PATH)
    #gray_image_progressing(TRAIN_BLACK_PATH, TRAIN_BLACK_GRAY_IMAGES_PATH)
    #gray_image_progressing(TEST_PATH, TEST_GRAY_IMAGES_PATH)

    # 数据分段解析处理
    gray_image_progressing(TEST_PATH, TEST_GRAY_IMAGES_PATH, 0, 3000)
    gray_image_progressing(TEST_PATH, TEST_GRAY_IMAGES_PATH, 3000, 6000)
    gray_image_progressing(TEST_PATH, TEST_GRAY_IMAGES_PATH, 6000)

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











