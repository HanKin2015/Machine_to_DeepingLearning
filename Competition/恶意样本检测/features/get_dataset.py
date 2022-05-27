# -*- coding: utf-8 -*-
"""
文 件 名: get_dataset.py
文件描述: 使用字典暴力破解登陆ftp
作    者: HeJian
创建日期: 2022.05.27
修改日期：2022.05.27

Copyright (c) 2022 HeJian. All rights reserved.
"""

import os
import pefile
import pandas as pd

class PEFile:
    def __init__(self, filename):
        self.pe = pefile.PE(filename, fast_load=True)
        self.filename   = filename
        #pe.OPTIONAL_HEADER.DATA_DIRECTORY 数据目录表，保存了各种表的RVA及大小，数据的起始RVA和数据块的长度
        self.ExportRVA  = self.pe.OPTIONAL_HEADER.DATA_DIRECTORY[0].VirtualAddress  #导出表
        self.ExportSize = self.pe.OPTIONAL_HEADER.DATA_DIRECTORY[0].Size            #
        self.ResSize    = self.pe.OPTIONAL_HEADER.DATA_DIRECTORY[2].Size            #资源表
                                                                                    #
        self.DebugSize  = self.pe.OPTIONAL_HEADER.DATA_DIRECTORY[6].Size            #调试表
        self.DebugRVA   = self.pe.OPTIONAL_HEADER.DATA_DIRECTORY[6].VirtualAddress  #
        self.IATRVA     = self.pe.OPTIONAL_HEADER.DATA_DIRECTORY[12].VirtualAddress #导入地址表（Import Address Table）地址
        self.ImageVersion = self.pe.OPTIONAL_HEADER.MajorImageVersion               #可运行于操作系统的主版本号
        self.OSVersion = self.pe.OPTIONAL_HEADER.MajorOperatingSystemVersion        #要求操作系统最低版本号的主版本号
        self.LinkerVersion = self.pe.OPTIONAL_HEADER.MajorLinkerVersion             #链接程序的主版本号
        self.StackReserveSize =self.pe.OPTIONAL_HEADER.SizeOfStackReserve           #初始化时的栈大小
        self.Dll =self.pe.OPTIONAL_HEADER.DllCharacteristics                        #DllMain()函数何时被调用，默认为 0
        self.NumberOfSections = self.pe.FILE_HEADER.NumberOfSections                #区块表的个数

    def Construct(self):
        sample = {}
        for attr, k in self.__dict__.items():
            if(attr != "pe"):
                sample[attr] = k
        return sample

def pe2vec(direct):
    """生成数据文件

    遍历文件夹，获取文件夹中所有pe文件信息
    
    Parameters
    ------------
    source_path : str
        源文件路径
    data_size : int
        源文件大小
        
    Returns
    -------
    bool
        成功True, 失败False
    """

    dataset = {}
    
    for subdir, dirs, files in os.walk(direct):
        for f in files:
            file_path = os.path.join(subdir, f)
            try:
                pe = PEFile(file_path)
                dataset[str(f)] = pe.Construct()
            except Exception as e:
                print('file:{}, {}'.format(f, e))
    return dataset

def vec2csv(dataset, filename):
    df = pd.DataFrame(dataset)
    infected = df.transpose()
    infected.to_csv(filename, sep=',', encoding='utf-8')

if __name__ == '__main__':
    black_data = './data/black/'
    black_dataset_path = './data/black_data.csv'
    write_data = './data/write/'
    write_dataset_path = './data/write_data.csv'
    
    black_dataset = pe2vec(black_data)
    vec2csv(black_dataset, black_dataset_path)
    
    write_dataset = pe2vec(write_data)
    vec2csv(write_dataset, write_dataset_path)
