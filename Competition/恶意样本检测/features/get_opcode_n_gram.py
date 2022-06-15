# -*- coding: utf-8 -*-
"""
文 件 名: get_opcode_n_gram.py
文件描述: 获取二进制文件的n-gram特征(使用pefile库反汇编获取操作指令码)
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
TRAIN_WHITE_STRING_FEATURES_PATH = DATA_PATH+'train_white_string_features.csv' # 训练集白样本字符串特征数据集路径
TRAIN_BLACK_STRING_FEATURES_PATH = DATA_PATH+'train_black_string_features.csv' # 训练集黑样本字符串特征数据集路径
TEST_STRING_FEATURES_PATH        = DATA_PATH+'test_string_features.csv'        # 测试集样本数字符串特征据集路径
TRAIN_WHITE_CUSTOM_STRINGS_PATH  = DATA_PATH+'train_white_strings.csv'         # 训练集白样本自定义字符串数据集路径
TRAIN_BLACK_CUSTOM_STRINGS_PATH  = DATA_PATH+'train_black_strings.csv'         # 训练集黑样本自定义字符串数据集路径
TEST_CUSTOM_STRINGS_PATH         = DATA_PATH+'test_strings.csv'                # 测试集样本自定义字符串数据集路径
TRAIN_WHITE_GRAY_IMAGES_PATH     = './gray_images/train/white/'                # 训练集白样本灰度图像存储路径
TRAIN_BLACK_GRAY_IMAGES_PATH     = './gray_images/train/black/'                # 训练集黑样本灰度图像存储路径
TEST_GRAY_IMAGES_PATH            = './gray_images/test/'                       # 测试集样本灰度图像存储路径
TRAIN_WHITE_OPCODE_3_GRAM_PATH   = DATA_PATH+'train_white_opcode_3_gram.csv'   # 训练集白样本操作指令码3-gram特征存储路径
TRAIN_BLACK_OPCODE_3_GRAM_PATH   = DATA_PATH+'train_black_opcode_3_gram.csv'   # 训练集黑样本操作指令码3-gram特征存储路径
TEST_OPCODE_3_GRAM_PATH          = DATA_PATH+'test_opcode_3_gram.csv'          # 测试集样本操作指令码3-gram特征存储路径

# 线程数量
THREAD_NUM = 64

def get_opcode_sequence(root, file):
    """获取操作指令码序列
    """
    
    file_path = '{}{}'.format(root, file)
    try:
        pe = pefile.PE(file_path)
    except Exception as e:
        logger.error('file[{}] pefile parse failed, error[{}].'.format(file, e))
        return None
    
    #从程序头中获取程序入口点的地址
    entrypoint = pe.OPTIONAL_HEADER.AddressOfEntryPoint
    #计算入口代码被加载到内存中的内存地址
    entrypoint_address = entrypoint+pe.OPTIONAL_HEADER.ImageBase
    #从PE文件对象获取二进制代码
    binary_code = pe.get_memory_mapped_image()[entrypoint:-1]
    logger.debug('len(binary_code) = {}'.format(len(binary_code)))
    #初始化反汇编程序以反汇编32位x86二进制代码
    disassembler = Cs(CS_ARCH_X86, CS_MODE_32)
    #反汇编代码
    opcode_seq = []
    for instruction in disassembler.disasm(binary_code, entrypoint_address):
        logger.debug('{}\t{}.'.format(instruction.mnemonic, instruction.op_str))
        if instruction.mnemonic not in ['int1', 'int3', 'align']:
            opcode_seq.append(instruction.mnemonic)
    logger.debug('opcode_seq count: {}.'.format(opcode_seq))
    return opcode_seq

def get_opcode_n_gram(ops, n=3):
    """根据opcode序列，统计对应的n-gram
    """
    
    opngramlist = [tuple(ops[i:i+n]) for i in range(len(ops)-n)]
    opngram = Counter(opngramlist)
    return opngram
    
def get_opcode_3_gram(root, file):
    """获取二进制文件的3-gram特征
    """
    
    ops = get_opcode_sequence(root, file)
    logger.debug(ops)
    if ops is None:
        return None,None
    op3gram = get_opcode_n_gram(ops)
    logger.debug(op3gram)
    return file,op3gram
    
def save_to_csv(map3gram, save_path):
    """保存数据到csv文件
    """
    
    cc = Counter([])
    for d in map3gram.values():
        cc += d
    selectedfeatures = {}
    tc = 0
    for k,v in cc.items():
        if v >= 500:
            selectedfeatures[k] = v
            logger.info('k = {}, v = {}.'.format(k, v))
            tc += 1
    dataframelist = []
    for file_name,op3gram in map3gram.items():
        standard = {}
        standard["file_name"] = file_name
        for feature in selectedfeatures:
            if feature in op3gram:
                standard[feature] = op3gram[feature]
            else:
                standard[feature] = 0
        dataframelist.append(standard)
    df = pd.DataFrame(dataframelist)
    df.to_csv(save_path, index=False)

def opcode_n_gram_progressing(data_path, save_path):
    """n-gram处理
    """
    
    map3gram = defaultdict(Counter)

    tasks = []
    with ThreadPoolExecutor(max_workers=THREAD_NUM) as pool:
        for root, dirs, files in os.walk(data_path):
            for file in files:
                task = pool.submit(get_opcode_3_gram, root, file)
                tasks.append(task)
    
    failed_count = 0
    for task in tasks:
        file_name, op3gram = task.result()
        if op3gram is not None:
            map3gram[file_name] = op3gram
        else:
            failed_count += 1
    logger.info('{} has {} samples which got opcode 3-gram failed.'.format(data_path, failed_count))
    save_to_csv(map3gram, save_path)
    
def main():
    opcode_n_gram_progressing(TRAIN_WHITE_PATH, TRAIN_WHITE_OPCODE_3_GRAM_PATH)
    opcode_n_gram_progressing(TRAIN_BLACK_PATH, TRAIN_BLACK_OPCODE_3_GRAM_PATH)
    opcode_n_gram_progressing(TEST_PATH, TEST_OPCODE_3_GRAM_PATH)

def debug():
    """调试
    """
    
    root = 'D:\\Github\\Machine_to_DeepingLearning\\Competition\\恶意样本检测\\features\\backup\\'
    file = 'FTOOL.exe'
    #get_binary_code(root, file)
    file_path = './0ACDbR5M3ZhBJajygTuf.asm'
    get_n_gram(root, file)
    
if __name__ == '__main__':
    start_time = time.time()

    main()

    end_time = time.time()
    logger.info('process spend {} s.'.format(round(end_time - start_time, 3)))











