# -*- coding: utf-8 -*-
"""
文 件 名: get_data.py
文件描述: 通过pefile库获取pe文件信息
作    者: HeJian
创建日期: 2022.05.27
修改日期：2022.06.06
Copyright (c) 2022 HeJian. All rights reserved.
"""

import os
os.environ['NUMEXPR_MAX_THREADS'] = '64'
import pefile
import pandas as pd
import time
from log import logger
from concurrent.futures import ThreadPoolExecutor
import stat
#import subprocess
import hashlib

TRAIN_WHITE_PATH = './AIFirst_data/train/white/' # 训练集白样本路径
TRAIN_BLACK_PATH = './AIFirst_data/train/black/' # 训练集黑样本路径
TEST_PATH        = './AIFirst_data/test/'        # 测试集样本路径
DATA_PATH        = './data/'                     # 数据路径
TRAIN_WHITE_DATASET_PATH = DATA_PATH+'train_white_dataset.csv' # 训练集白样本数据集路径
TRAIN_BLACK_DATASET_PATH = DATA_PATH+'train_black_dataset.csv' # 训练集黑样本数据集路径
TEST_DATASET_PATH        = DATA_PATH+'test_dataset.csv'        # 测试集样本数据集路径

# 线程数量
THREAD_NUM = 64

# 创建数据文件夹
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

exception_error = {
    'Invalid NT Headers signature.': 1,
    'Invalid NT Headers signature. Probably a NE file': 2,
    'Invalid e_lfanew value, probably not a PE file': 3,
    'Invalid NT Headers signature. Probably a LE file': 4,
    'Unable to read the DOS Header, possibly a truncated file.': 5,
    'DOS Header magic not found.': 6,
    'list index out of range': 7,
}

# 数据目录表
directory_names = (
    'IMAGE_DIRECTORY_ENTRY_EXPORT',
    'IMAGE_DIRECTORY_ENTRY_IMPORT',
    'IMAGE_DIRECTORY_ENTRY_RESOURCE',
    'IMAGE_DIRECTORY_ENTRY_EXCEPTION',
    'IMAGE_DIRECTORY_ENTRY_SECURITY',
    'IMAGE_DIRECTORY_ENTRY_BASERELOC',
    'IMAGE_DIRECTORY_ENTRY_DEBUG',
    'IMAGE_DIRECTORY_ENTRY_COPYRIGHT',
    'IMAGE_DIRECTORY_ENTRY_GLOBALPTR',
    'IMAGE_DIRECTORY_ENTRY_TLS',
    'IMAGE_DIRECTORY_ENTRY_LOAD_CONFIG',
    'IMAGE_DIRECTORY_ENTRY_BOUND_IMPORT',
    'IMAGE_DIRECTORY_ENTRY_IAT',
    'IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT',
    'IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR',
    'IMAGE_DIRECTORY_ENTRY_RESERVED',
)

def get_index(directory_name):
    """获取数据目录表索引
    """
    
    complete_name = 'IMAGE_DIRECTORY_ENTRY_' + directory_name
    return pefile.DIRECTORY_ENTRY[complete_name]

class PEFile:
    def __init__(self, root, file):
        file_path = os.path.join(root, file)
        self.FileName       = file                                                 # PE文件名
        self.FileMD5        = self.get_file_md5(file_path)
        self.FileSize       = os.path.getsize(file_path)
        self.FileModifyTime = int(os.path.getmtime(file_path))
        self.FileCreateTime = int(os.path.getctime(file_path))
        self.FileAccessTime = int(os.path.getatime(file_path))
        
        mode = os.stat(file_path).st_mode
        #self.IsReadonly    = ('i' in subprocess.check_output(['lsattr', file_path])[:16])
        self.IsRead    = bool(stat.S_IREAD & mode)
        self.IsWrite   = bool(stat.S_IWRITE & mode)
        self.IsExecute = bool(stat.S_IEXEC & mode)
        #self.Permissions   = oct(stat.S_IMODE(mode)) 
       
        # 打开PE文件失败
        self.ExceptionError = 0
        try:
            pe = pefile.PE(file_path)
        except Exception as e:
            err = str(e).replace("'", "")
            self.ExceptionError = exception_error[err]
            return

        self.ImageBase        = pe.OPTIONAL_HEADER.ImageBase                         # 镜像基址
        self.ImageSize        = pe.OPTIONAL_HEADER.SizeOfImage                       # 镜像大小
        self.EpAddress        = pe.OPTIONAL_HEADER.AddressOfEntryPoint               # 程序入口
        self.LinkerVersion    = pe.OPTIONAL_HEADER.MajorLinkerVersion                # 编译器
        self.TimeDateStamp    = pe.FILE_HEADER.TimeDateStamp                         # 编译时间
        # pe.OPTIONAL_HEADER.DATA_DIRECTORY 数据目录表，保存了各种表数据的起始RVA及数据块的长度
        if pe.OPTIONAL_HEADER.DATA_DIRECTORY:
            try:
                self.ExportRVA    = pe.OPTIONAL_HEADER.DATA_DIRECTORY[0].VirtualAddress  # 导出表的RVA
                self.ExportSize   = pe.OPTIONAL_HEADER.DATA_DIRECTORY[0].Size            # 导出表的大小
                self.NumberOfExFunctions = 0                                             # 导出函数的个数
                if self.ExportSize:
                    # 有部分文件字段存在值但实际上无法解析到
                    try:
                        self.NumberOfExFunctions = pe.DIRECTORY_ENTRY_EXPORT.struct.NumberOfFunctions
                    except Exception as e:
                        self.NumberOfExFunctions = -1
                        logger.debug('file[{}] get EXPORT info failed, error[{}].'.format(file, e))
                self.ImportRVA    = pe.OPTIONAL_HEADER.DATA_DIRECTORY[1].VirtualAddress  # 导入表的RVA
                self.ImportSize   = pe.OPTIONAL_HEADER.DATA_DIRECTORY[1].Size            # 导入表的大小
                self.NumberOfImFunctions = 0                                              # 导入函数的个数
                if pe.OPTIONAL_HEADER.DATA_DIRECTORY[get_index('IMPORT')].Size:
                    try:
                        for entry in pe.DIRECTORY_ENTRY_IMPORT:
                            self.NumberOfImFunctions += len(entry.imports)
                    except Exception as e:
                        self.NumberOfImFunctions = -1
                        logger.debug('file[{}] get IMPORT info failed, error[{}].'.format(file, e))
                self.ResourceSize   = pe.OPTIONAL_HEADER.DATA_DIRECTORY[2].Size            # 资源表的大小
                self.HasResources   = bool(pe.OPTIONAL_HEADER.DATA_DIRECTORY[get_index('RESOURCE')].Size)
                self.HasRelocations = bool(pe.OPTIONAL_HEADER.DATA_DIRECTORY[get_index('BASERELOC')].Size)
                self.DebugRVA       = pe.OPTIONAL_HEADER.DATA_DIRECTORY[6].VirtualAddress  # 调试表的RVA
                self.DebugSize      = pe.OPTIONAL_HEADER.DATA_DIRECTORY[6].Size            # 调试表的大小
                self.HasDebug       = bool(self.DebugSize)
                self.HasTls         = bool(pe.OPTIONAL_HEADER.DATA_DIRECTORY[get_index('TLS')].Size)
                self.IATRVA         = pe.OPTIONAL_HEADER.DATA_DIRECTORY[12].VirtualAddress # 导入地址表（Import Address Table）地址
            except Exception as e:
                err = str(e).replace("'", "")
                self.ExceptionError = exception_error[err]
                return
        self.ImageVersion     = pe.OPTIONAL_HEADER.MajorImageVersion                 # 可运行于操作系统的主版本号
        self.OSVersion        = pe.OPTIONAL_HEADER.MajorOperatingSystemVersion       # 要求操作系统最低版本号的主版本号
        self.NumberOfSymbols  = pe.FILE_HEADER.NumberOfSymbols
        self.Signature        = pe.NT_HEADERS.Signature
        self.StackReserveSize = pe.OPTIONAL_HEADER.SizeOfStackReserve                # 保留的栈大小
        self.Dll              = pe.OPTIONAL_HEADER.DllCharacteristics                # DllMain()函数何时被调用，默认为 0
        self.NumberOfSections = pe.FILE_HEADER.NumberOfSections                      # 区块表的个数

    def get_file_md5(self, file_path):
        """获取文件md5值

        通过hashlib库计算文件的md5值
        
        Parameters
        ------------
        file_path : str
            文件路径

        Returns
        -------
        str
            返回文件的md5值
        """

        m = hashlib.md5()   #创建md5对象
        with open(file_path,'rb') as fobj:
            while True:
                data = fobj.read(4096)
                if not data:
                    break
                m.update(data)  #更新md5对象
        return m.hexdigest()    #返回md5值
        
    def construct(self):
        sample = {}
        for key, value in self.__dict__.items():
            sample[key] = value
        return sample

def get_pefile_info(root, file):
    """获取PE文件信息

    通过类PEFile获取信息
    
    Parameters
    ------------
    root : str
        当前路径
    file : str
        PE文件名
        
    Returns
    -------
    pefile_info : dict
        PE文件信息
    """

    try:
        pe = PEFile(root, file)
        return pe.construct()
    except Exception as e:
        logger.info('file[{}] get info failed, error[{}].'.format(file, e))

def get_dataset(data_path):
    """获取数据集

    遍历文件夹，获取文件夹中所有样本文件的信息集
    
    Parameters
    ------------
    data_path : str
        数据路径
        
    Returns
    -------
    dataset : list
        数据集列表
    """

    dataset = []
    tasks = []
    
    with ThreadPoolExecutor(max_workers=THREAD_NUM) as pool:
        for root, dirs, files in os.walk(data_path):
            for file in files:
                task = pool.submit(get_pefile_info, root, file)
                tasks.append(task)
    
    failed_samples_count = 0
    for task in tasks:
        sample = task.result()
        if sample is not None:
            dataset.append(sample)
        else:
            failed_samples_count += 1
    logger.info('{} has {} samples which got pe info failed .'.format(data_path, failed_samples_count))
    return dataset

def dataset_to_csv(dataset, csv_path):
    """数据集保存到本地csv文件中

    数据集保存到本地csv文件中
    
    Parameters
    ------------
    dataset : str
        数据集
    csv_path : str
        数据集的csv存储路径
        
    Returns
    -------
    """
    
    df = pd.DataFrame(dataset)
    df.to_csv(csv_path, sep=',', encoding='utf-8', index=False)

def data_processing(data_path, csv_path):
    """数据处理
    
    Parameters
    ------------
    data_path : str
        数据路径
    csv_path : str
        数据集的csv存储路径
        
    Returns
    -------
    """
    dataset = get_dataset(data_path)
    dataset_to_csv(dataset, csv_path)

def main():
    data_processing(TRAIN_BLACK_PATH, TRAIN_BLACK_DATASET_PATH)
    data_processing(TRAIN_WHITE_PATH, TRAIN_WHITE_DATASET_PATH)
    data_processing(TEST_PATH, TEST_DATASET_PATH)

if __name__ == '__main__':
    start_time = time.time()

    main()
    
    end_time = time.time()
    logger.info('process spend {} s.'.format(round(end_time - start_time, 3)))
