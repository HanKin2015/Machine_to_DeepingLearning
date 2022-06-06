from log import logger
import os
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import ctypes
import win32file
import win32con

DATA_PATH        = './data/' # 数据路径
TRAIN_BLACK_DIRTY_DATA_PATH = './backup/'
TRAIN_WHITE_DIRTY_DATASET_PATH = DATA_PATH+'train_white_dirty_dataset.csv' # 训练集白样本脏数据路径
TRAIN_BLACK_DIRTY_DATASET_PATH = DATA_PATH+'train_black_dirty_dataset.csv' # 训练集黑样本脏数据集路径
TEST_DIRTY_DATASET_PATH        = DATA_PATH+'dirty_test_dataset.csv'        # 测试集脏数据集路径
THREAD_NUM = 64 # 线程数量

class File:
    def __init__(self, root, file):
        file_path = os.path.join(root, file)
        logger.info('file path: {}'.format(file_path))
        self.FileName = file
        self.FileMD5  = self.get_file_md5(file_path)
        self.FileSize = os.path.getsize(file_path)
        self.FileModifyTime = int(os.path.getmtime(file_path))
        self.FileCreateTime = int(os.path.getctime(file_path))
        self.FileAccessTime = int(os.path.getatime(file_path))
        self.FileUsageSize  = self.get_file_usage_size(file_path)
        file_flag = win32file.GetFileAttributesW(file_path)
        self.is_readonly = bool(file_flag & win32con.FILE_ATTRIBUTE_READONLY)
        self.is_hiden    = bool(file_flag & win32con.FILE_ATTRIBUTE_HIDDEN)
        self.is_system   = bool(file_flag & win32con.FILE_ATTRIBUTE_SYSTEM)
        self.is_execute  = os.access(file_path, os.X_OK)
        #self.get_file_attributes(file_path)
        
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

    def get_file_usage_size(self, file_path):
        """获取文件的所占空间大小
        
        函数可用性有待考验
        
        """

        sectors_per_cluster = ctypes.c_ulonglong(0)
        path = ctypes.c_char_p(file_path.encode('gb2312'))
        usage_size = ctypes.windll.kernel32.GetCompressedFileSizeA(path, ctypes.pointer(sectors_per_cluster))

        logger.info('{}: {} {} {} {}.'.format(file_path, usage_size, 
        os.access(file_path, os.R_OK), os.access(file_path, os.W_OK), os.access(file_path, os.X_OK)))
        return usage_size

    def get_file_attributes(self, file_path):
        file_flag = win32file.GetFileAttributesW(file_path)
        is_readonly = bool(file_flag & win32con.FILE_ATTRIBUTE_READONLY)
        is_hiden = bool(file_flag & win32con.FILE_ATTRIBUTE_HIDDEN)
        is_system = bool(file_flag & win32con.FILE_ATTRIBUTE_SYSTEM)

        t = os.path.getmtime(file_path)
        timeStruce = time.localtime(t)
        times = time.strftime('%Y-%m-%d %H:%M:%S', timeStruce)
        
        logger.info('{}: {} {} {} {} {}.'.format(file_path, is_readonly, is_hiden, is_system, 
        timeStruce, times))

    def construct(self):
        sample = {}
        for key, value in self.__dict__.items():
            sample[key] = value
        return sample
        
def get_data_files(file_dir, file_type_suffix='.txt'):
    """获取文件夹中的指定类型的文件名及路径
    
    深度为1
    
    Parameters
    ------------
    file_dir : str
        文件夹路径
    file_type_suffix : str
        文件类型后缀

    Returns
    -------
    list
        返回文件的md5值
    """
    
    data_files = []   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == file_type_suffix:  
                data_files.append(os.path.join(root, file))  
    return data_files

def get_file_info(root, file):
    """获取文件信息
    
    通过类File获取文件的创建时间(CreationTime)，最后修改时间(LastAccessTime)，最后访问时间(LastAccessTime)，
    文件所占空间大小(AllocationSize)，文件大小(EndOfFile)，是否只读(ReadOnly)，是否隐藏(Hide)等等
    
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
        pe = File(root, file)
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
                task = pool.submit(get_file_info, root, file)
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

def main():
    dataset = get_dataset(TRAIN_BLACK_DIRTY_DATA_PATH)
    logger.info('directory {} has {} data files.'.format(TRAIN_BLACK_DIRTY_DATA_PATH, len(dataset)))
    
    logger.info(dataset)

if __name__ == '__main__':
    start_time = time.time()

    main()
    
    end_time = time.time()
    logger.info('process spend {} s.'.format(round(end_time - start_time, 3)))
