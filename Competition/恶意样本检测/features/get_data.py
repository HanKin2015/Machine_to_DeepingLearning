# -*- coding: utf-8 -*-
"""
文 件 名: get_data.py
文件描述: 通过pefile库获取pe文件信息
作    者: HeJian
创建日期: 2022.05.27
修改日期：2022.06.11
Copyright (c) 2022 HeJian. All rights reserved.
"""

import os
os.environ['NUMEXPR_MAX_THREADS'] = '64'
import pefile
import pandas as pd
import time
from log import logger
from concurrent.futures import ThreadPoolExecutor
import numpy as np

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

exception_error = {
    'Invalid NT Headers signature.': 1,
    'Invalid NT Headers signature. Probably a NE file': 2,
    'Invalid e_lfanew value, probably not a PE file': 3,
    'Invalid NT Headers signature. Probably a LE file': 4,
    'Unable to read the DOS Header, possibly a truncated file.': 5,
    'DOS Header magic not found.': 6,
    'list index out of range': 7,
}

def get_index(directory_name):
    complete_name = 'IMAGE_DIRECTORY_ENTRY_' + directory_name
    return pefile.DIRECTORY_ENTRY[complete_name]

class PEFile:
    def __init__(self, root, file):
        file_path = os.path.join(root, file)
        self.FileName       = file
        self.ExceptionError = 0
        try:
            pe = pefile.PE(file_path)
        except Exception as e:
            err = str(e).replace("'", "")
            self.ExceptionError = exception_error[err]
            logger.error('file {} pefile parse failed, {}.'.format(file, err))
            return

        # DOS_HEADER
        dos_header_members = ['e_magic', 'e_cblp', 'e_cp', 'e_crlc', 'e_cparhdr', 'e_minalloc', 'e_maxalloc', 'e_ss', 'e_sp',
            'e_csum', 'e_ip', 'e_cs', 'e_lfarlc', 'e_ovno', 'e_res', 'e_oemid', 'e_oeminfo', 'e_res2', 'e_lfanew']
        for dos_header_member in dos_header_members:
            exec('self.{} = pe.DOS_HEADER.{}'.format(dos_header_member, dos_header_member))
        
        # NT_HEADERS
        nt_header_members = ['Signature']
        for nt_header_member in nt_header_members:
            exec('self.{} = pe.NT_HEADERS.{}'.format(nt_header_member, nt_header_member))
            
        # FILE_HEADER
        file_header_members = ['Machine', 'NumberOfSections', 'TimeDateStamp', 'PointerToSymbolTable', 'NumberOfSymbols', 'SizeOfOptionalHeader', 'Characteristics']
        for file_header_member in file_header_members:
            exec('self.{} = pe.FILE_HEADER.{}'.format(file_header_member, file_header_member))
        
        # OPTIONAL_HEADER(BaseOfData字段可能不存在)
        optional_header_members = ['Magic', 'MajorLinkerVersion', 'MinorLinkerVersion', 'SizeOfCode', 'SizeOfInitializedData',
            'SizeOfUninitializedData', 'AddressOfEntryPoint', 'BaseOfCode', 'ImageBase',
            'SectionAlignment', 'FileAlignment', 'MajorOperatingSystemVersion', 'MinorOperatingSystemVersion',
            'MajorImageVersion', 'MinorImageVersion', 'MajorSubsystemVersion',  'MinorSubsystemVersion', 'Reserved1', 'SizeOfImage',
            'SizeOfHeaders', 'CheckSum', 'Subsystem', 'DllCharacteristics', 'SizeOfStackReserve', 'SizeOfStackCommit', 
            'SizeOfHeapReserve', 'SizeOfHeapCommit', 'LoaderFlags', 'NumberOfRvaAndSizes']
        for optional_header_member in optional_header_members:
            exec('self.{} = pe.OPTIONAL_HEADER.{}'.format(optional_header_member, optional_header_member))
        self.BaseOfData = 0
        if hasattr(pe.OPTIONAL_HEADER, 'BaseOfData'):
            self.BaseOfData = pe.OPTIONAL_HEADER.BaseOfData
        
        # PE Sections
        Entropy          = []
        SizeOfRawData    = []
        Misc_VirtualSize = []
        for section in pe.sections:
            Entropy.append(section.get_entropy())
            SizeOfRawData.append(section.SizeOfRawData)
            Misc_VirtualSize.append(section.Misc_VirtualSize)
        self.SectionsMeanEntropy          = np.mean(Entropy) if Entropy else 0
        self.SectionsMinEntropy           = min(Entropy) if Entropy else 0
        self.SectionsMaxEntropy           = max(Entropy) if Entropy else 0
        self.SectionsSizeOfRawData        = np.mean(SizeOfRawData) if SizeOfRawData else 0
        self.SectionsMinRawsize           = min(SizeOfRawData) if SizeOfRawData else 0
        self.SectionMaxRawsize            = max(SizeOfRawData) if SizeOfRawData else 0
        self.SectionsMeanMisc_VirtualSize = np.mean(Misc_VirtualSize) if Misc_VirtualSize else 0
        self.SectionsMinMisc_VirtualSize  = min(Misc_VirtualSize) if Misc_VirtualSize else 0
        self.SectionMaxMisc_VirtualSize   = max(Misc_VirtualSize) if Misc_VirtualSize else 0
        
        # Directories
        # pe.OPTIONAL_HEADER.DATA_DIRECTORY 数据目录表，保存了各种表数据的起始RVA及数据块的长度
        if len(pe.OPTIONAL_HEADER.DATA_DIRECTORY) != 16:
            self.ExceptionError = exception_error['list index out of range']
            logger.debug('this pe file invalid, {}.'.format('list index out of range'))
            return
        for i in range(16):
            RVA_key  = 'DirectoryRVA{}'.format(i)
            Size_key = 'DirectorySize{}'.format(i)
            exec('self.{} = pe.OPTIONAL_HEADER.DATA_DIRECTORY[{}].VirtualAddress'.format(RVA_key, i))
            exec('self.{} = pe.OPTIONAL_HEADER.DATA_DIRECTORY[{}].Size'.format(Size_key, i))

        # Exported symbols
        exported_symbols_members = ['Characteristics', 'TimeDateStamp', 'MajorVersion', 'MinorVersion', 'Name', 'Base',
            'NumberOfFunctions', 'NumberOfNames', 'AddressOfFunctions', 'AddressOfNames', 'AddressOfNameOrdinals']
        for exported_symbols_member in exported_symbols_members:
            key = 'Export{}'.format(exported_symbols_member)
            if not hasattr(pe, 'DIRECTORY_ENTRY_EXPORT'):
                exec('self.{} = 0'.format(key))
            else:
                exec('self.{} = pe.DIRECTORY_ENTRY_EXPORT.struct.{}'.format(key, exported_symbols_member))

        # Imported symbols
        self.ImportNbDLL     = 0
        self.ImportNb        = 0
        self.ImportNbOrdinal = 0
        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            dll = []
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                dll.append(entry.dll)
                for function in entry.imports:
                    if function.ordinal:
                        self.ImportNbOrdinal += function.ordinal
            self.ImportNbDLL = len(set(dll))
            self.ImportNb = len(pe.DIRECTORY_ENTRY_IMPORT)

        # Resource directory
        self.ResourcesNb = 0
        self.ResourcesNumberOfNamedEntries = 0
        self.ResourcesMajorVersion = 0
        if hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE'):
            self.ResourcesNb = pe.DIRECTORY_ENTRY_RESOURCE.struct.NumberOfIdEntries
            self.ResourcesNumberOfNamedEntries = pe.DIRECTORY_ENTRY_RESOURCE.struct.NumberOfNamedEntries
            self.ResourcesMajorVersion = pe.DIRECTORY_ENTRY_RESOURCE.struct.MajorVersion
            
        # LOAD_CONFIG
        self.LoadConfigSize = 0
        if hasattr(pe, 'DIRECTORY_ENTRY_LOAD_CONFIG'):
            self.LoadConfigSize = pe.DIRECTORY_ENTRY_LOAD_CONFIG.struct.Size

        # Debug information
        self.DebugNb = 0
        if hasattr(pe, 'DIRECTORY_ENTRY_DEBUG'):
            self.DebugNb = len(pe.DIRECTORY_ENTRY_DEBUG)

        # Version Information
        version_info_members = ['Length', 'ValueLength', 'Type']
        for version_info_member in version_info_members:
            key = 'VS_VERSIONINFO{}'.format(version_info_member)
            if not hasattr(pe, 'VS_VERSIONINFO'):
                exec('self.{} = 0'.format(key))
            else:
                exec('self.{} = pe.VS_VERSIONINFO[0].{}'.format(key, version_info_member))

        fixversion_info_members = ['Signature', 'StrucVersion', 'FileVersionMS', 'FileVersionLS', 'ProductVersionMS', 'ProductVersionLS',
            'FileFlagsMask', 'FileFlags', 'FileOS', 'FileType', 'FileSubtype', 'FileDateMS', 'FileDateLS']
        for fixversion_info_member in fixversion_info_members:
            key = 'VS_FIXEDFILEINFO{}'.format(fixversion_info_member)
            if not hasattr(pe, 'VS_FIXEDFILEINFO'):
                exec('self.{} = 0'.format(key))
            else:
                exec('self.{} = pe.VS_FIXEDFILEINFO[0].{}'.format(key, fixversion_info_member))
        self.FileInfoNb = 0
        if hasattr(pe, 'FileInfo'):
            self.FileInfoNb = len(pe.FileInfo)
        """
        file_info_members = ['Length', 'ValueLength', 'Type']
        for file_info_member in file_info_members:
            string_key = 'StringFileInfo{}'.format(file_info_member)
            var_key = 'VarFileInfo{}'.format(file_info_member)
            if not hasattr(pe, 'FileInfo') or len(pe.FileInfo) == 0 or len(pe.FileInfo[0]) == 0:
                exec('self.{} = 0'.format(string_key))
                exec('self.{} = 0'.format(var_key))
            elif len(pe.FileInfo[0]) == 2:
                    exec('self.{} = pe.FileInfo[0][0].{}'.format(string_key, file_info_member))
                    exec('self.{} = pe.FileInfo[0][1].{}'.format(var_key, file_info_member))
            elif len(pe.FileInfo[0]) == 1:
                if pe.FileInfo[0][0].name == 'StringFileInfo':
                    exec('self.{} = pe.FileInfo[0][0].{}'.format(string_key, file_info_member))
                    exec('self.{} = 0'.format(var_key))
                if pe.FileInfo[0][0].name == 'VarFileInfo':
                    exec('self.{} = 0'.format(string_key))
                    exec('self.{} = pe.FileInfo[0][0].{}'.format(var_key, file_info_member))
        """
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
        logger.error('file[{}] get info failed, error[{}].'.format(file, e))

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
    del dataset

def main():
    logger.info('::: PE-Analysis :::')
    data_processing(TRAIN_BLACK_PATH, TRAIN_BLACK_DATASET_PATH)
    data_processing(TRAIN_WHITE_PATH, TRAIN_WHITE_DATASET_PATH)
    data_processing(TEST_PATH, TEST_DATASET_PATH)

def debug():
    root = 'C:\\Users\\Administrator\\AppData\\Local\\口袋助理\\files\\'
    file = 'procexp（中文）64.exe'
    #pe = PEFile(root, file)
    pe = PEFile(TRAIN_BLACK_PATH, '000057b32c463c4bcda99933faf15dd4')
    print(pe.construct())

if __name__ == '__main__':
    start_time = time.time()

    #debug()
    main()
    
    end_time = time.time()
    logger.info('process spend {} s.'.format(round(end_time - start_time, 3)))
