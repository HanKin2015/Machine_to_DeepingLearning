# -*- coding: utf-8 -*-
"""
文 件 名: get_data.py
文件描述: 通过pefile库获取pe文件信息
备    注：测试数据集19个脏数据
作    者: HeJian
创建日期: 2022.05.27
修改日期：2022.06.06
Copyright (c) 2022 HeJian. All rights reserved.
"""

from common import *

# pefile解析失败错误信息，7是部分索引不存在
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
        self.FileName       = file                                                   # PE文件名
        self.ExceptionError = 0
        try:
            pe = pefile.PE(file_path)
        except Exception as e:
            err = str(e).replace("'", "")
            self.ExceptionError = exception_error[err]
            logger.error('file {} pefile parse failed, {}.'.format(file, err))
            return

        # FILE_HEADER
        file_header_members = ['Machine', 'NumberOfSections', 'SizeOfOptionalHeader', 'Characteristics', 'TimeDateStamp']
        for file_header_member in file_header_members:
            exec('self.{} = pe.FILE_HEADER.{}'.format(file_header_member, file_header_member))
        
        # OPTIONAL_HEADER(BaseOfData字段可能不存在)
        optional_header_members = ['MajorLinkerVersion', 'MinorLinkerVersion', 'SizeOfCode', 'SizeOfInitializedData',
            'SizeOfUninitializedData', 'AddressOfEntryPoint', 'BaseOfCode', 'BaseOfData', 'ImageBase',
            'SectionAlignment', 'FileAlignment', 'MajorOperatingSystemVersion', 'MinorOperatingSystemVersion',
            'MajorImageVersion', 'MinorImageVersion', 'MajorSubsystemVersion',  'MinorSubsystemVersion', 'SizeOfImage',
            'SizeOfHeaders', 'CheckSum', 'Subsystem', 'DllCharacteristics', 'SizeOfStackReserve', 'SizeOfStackCommit', 
            'SizeOfHeapReserve', 'SizeOfHeapCommit', 'LoaderFlags', 'NumberOfRvaAndSizes']
        for optional_header_member in optional_header_members:
            exec('self.{} = 0'.format(optional_header_member))
            try:
                exec('self.{} = pe.OPTIONAL_HEADER.{}'.format(optional_header_member, optional_header_member))
            except Exception as e:
                logger.debug('file[{}] Structure object has no {}.'.format(file, optional_header_member))
                exec('self.{} = -1'.format(optional_header_member))
        
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
        if pe.OPTIONAL_HEADER.DATA_DIRECTORY:
            try:
                self.ExportRVA    = pe.OPTIONAL_HEADER.DATA_DIRECTORY[get_index('EXPORT')].VirtualAddress  # 导出表的RVA
                self.ExportSize   = pe.OPTIONAL_HEADER.DATA_DIRECTORY[get_index('EXPORT')].Size            # 导出表的大小
                self.ExportNb = 0
                if pe.OPTIONAL_HEADER.DATA_DIRECTORY[get_index('EXPORT')].Size:
                    try:
                        self.ExportNb = len(pe.DIRECTORY_ENTRY_EXPORT)
                    except Exception as e:
                        logger.debug('file[{}] get EXPORT info failed, error[{}].'.format(file, e))
                        self.ExportNb = -1
                #self.ImportRVA       = pe.OPTIONAL_HEADER.DATA_DIRECTORY[get_index('IMPORT')].VirtualAddress  # 导入表的RVA
                #self.ImportSize      = pe.OPTIONAL_HEADER.DATA_DIRECTORY[get_index('IMPORT')].Size            # 导入表的大小
                self.ImportNbDLL     = 0
                self.ImportNb        = 0
                self.ImportNbOrdinal = 0
                if pe.OPTIONAL_HEADER.DATA_DIRECTORY[get_index('IMPORT')].Size:
                    try:
                        dll = []
                        for entry in pe.DIRECTORY_ENTRY_IMPORT:
                            dll.append(entry.dll)
                            for function in entry.imports:
                                if function.ordinal:
                                    self.ImportNbOrdinal += function.ordinal
                        self.ImportNbDLL = len(set(dll))
                        self.ImportNb = len(pe.DIRECTORY_ENTRY_IMPORT)
                    except Exception as e:
                        logger.debug('file[{}] get DIRECTORY_ENTRY_IMPORT info failed, error[{}].'.format(file, e))
                        self.ImportNbDLL     = -1
                        self.ImportNb        = -1
                        self.ImportNbOrdinal = -1

                self.ResourceSize = pe.OPTIONAL_HEADER.DATA_DIRECTORY[get_index('RESOURCE')].Size            # 资源表的大小
                self.ResourcesNb = 0
                if pe.OPTIONAL_HEADER.DATA_DIRECTORY[get_index('RESOURCE')].Size:
                    try:
                        self.ResourcesNb = pe.DIRECTORY_ENTRY_RESOURCE.struct.NumberOfIdEntries
                    except Exception as e:
                        logger.debug('file[{}] get DIRECTORY_ENTRY_RESOURCE info failed, error[{}].'.format(file, e))
                        self.ResourcesNb = -1
                self.BaseRelocSize = pe.OPTIONAL_HEADER.DATA_DIRECTORY[get_index('BASERELOC')].Size
                self.DebugRVA      = pe.OPTIONAL_HEADER.DATA_DIRECTORY[get_index('DEBUG')].VirtualAddress  # 调试表的RVA
                self.DebugSize     = pe.OPTIONAL_HEADER.DATA_DIRECTORY[get_index('DEBUG')].Size            # 调试表的大小
                self.TlsSize       = pe.OPTIONAL_HEADER.DATA_DIRECTORY[get_index('TLS')].Size
                self.IATRVA        = pe.OPTIONAL_HEADER.DATA_DIRECTORY[get_index('IAT')].VirtualAddress # 导入地址表（Import Address Table）地址
                self.LoadConfigurationSize = 0
                if pe.OPTIONAL_HEADER.DATA_DIRECTORY[get_index('LOAD_CONFIG')].Size:
                    try:
                        self.LoadConfigurationSize = pe.DIRECTORY_ENTRY_LOAD_CONFIG.struct.Size
                    except Exception as e:
                        logger.debug('file[{}] has no DIRECTORY_ENTRY_LOAD_CONFIG.'.format(file))
                        self.LoadConfigurationSize = -1
                
                # Version Information
                try:
                    self.HasVS_VERSIONINFO = bool(len(pe.VS_VERSIONINFO))
                except Exception as e:
                    logger.debug('file[{}] has no VS_VERSIONINFO.'.format(file))
                    self.HasVS_VERSIONINFO = False
                try:
                    self.HasVS_FIXEDFILEINFO = bool(len(pe.VS_FIXEDFILEINFO))
                except Exception as e:
                    logger.debug('file[{}] has no VS_FIXEDFILEINFO.'.format(file))
                    self.HasVS_FIXEDFILEINFO = False
                
            except Exception as e:
                err = str(e).replace("'", "")
                self.ExceptionError = exception_error[err]
                logger.debug('other error, {}.'.format(err))
                return

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
        #logger.error('file[{}] get info failed, error[{}].'.format(file, e))
        err = str(e).replace("'", "")
        self.ExceptionError = exception_error[err]
        logger.error('other error, {}.'.format(err))

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
    df.to_csv(DATA_PATH+csv_path, sep=',', encoding='utf-8', index=False)

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

def debug():
    root = '../AIFirst_data/train/white/'
    file = 'ff9a71ae3c6b12089414b05074457b64'
    pe = PEFile(root, file)
    logger.info(pe.construct())
    
def main():
    data_processing(TRAIN_BLACK_PATH, TRAIN_BLACK_DATASET_FILENAME)
    data_processing(TRAIN_WHITE_PATH, TRAIN_WHITE_DATASET_FILENAME)
    data_processing(TEST_PATH, TEST_DATASET_FILENAME)

if __name__ == '__main__':
    start_time = time.time()

    #debug()
    main()
    
    end_time = time.time()
    logger.info('process spend {} s.'.format(round(end_time - start_time, 3)))
