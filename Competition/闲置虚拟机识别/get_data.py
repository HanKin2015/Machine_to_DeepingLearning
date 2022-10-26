# -*- coding: utf-8 -*-
"""
文 件 名: get_data.py
文件描述: 获取数据
作    者: HanKin
创建日期: 2022.07.20
修改日期：2022.07.20

Copyright (c) 2022 HanKin. All rights reserved.
"""

from common import *

class MachineState:
    def __init__(self, root, file):
        file_path = os.path.join(root, file)
        self.ids = file[:-4]
        df = pd.read_csv(file_path)
        #print(df.shape)
        
        column_names = df.columns
        #print(column_names)
        
        for column_name in column_names:
            exec('self.{}_mean   = round(df.{}.mean())'.format(column_name, column_name))
            exec('self.{}_median = round(df.{}.median())'.format(column_name, column_name))
            exec('self.{}_max    = round(df.{}.max())'.format(column_name, column_name))
            exec('self.{}_min    = round(df.{}.min())'.format(column_name, column_name))

    def construct(self):
        state = {}
        for key, value in self.__dict__.items():
            state[key] = value
        return state

def get_machine_state(root, file):
    """
    """
    
    try:
        state = MachineState(root, file)
        return state.construct()
    except Exception as e:
        print('file[{}] get machine state failed, error[{}].'.format(file, e))

def get_data(raw_data_path):
    """获取数据集
    """

    data  = []
    tasks = []
    
    with ThreadPoolExecutor(max_workers=THREAD_NUM) as pool:
        for root, dirs, files in os.walk(raw_data_path):
            for file in files:
                task = pool.submit(get_machine_state, root, file)
                tasks.append(task)
    
    failed_count = 0
    for task in tasks:
        sample = task.result()
        if sample is not None:
            data.append(sample)
        else:
            failed_count += 1
    print('{} has {} samples which got data failed .'.format(raw_data_path, failed_count))
    return data

def data_to_csv(data, csv_path):
    """
    数据保存到本地csv文件中
    """
    
    df = pd.DataFrame(data)
    df.to_csv(csv_path, sep=',', encoding='utf-8', index=False)

def raw_data_processing(raw_data_path, csv_path):
    """
    数据处理
    """
    
    data = get_data(raw_data_path)
    data_to_csv(data, csv_path)

def main():
    raw_data_processing(TRAIN_RAW_DATA_PATH, TRAIN_DATA_PATH)
    raw_data_processing(TEST_RAW_DATA_PATH, TEST_DATA_PATH)

def debug():
    root = r'D:\Users\User\Desktop\AI\idle_machine_detect\idle_machine_detect\train\data\\'
    file = '7.csv'
    machine_state = get_machine_state(root, file)
    print(machine_state)
    print(int(2.7920231712))
    print(round(2.7920231712))

if __name__ == '__main__':
    start_time = time.time()

    #debug()
    main()

    end_time = time.time()
    print('process spend {} s.'.format(round(end_time - start_time, 3)))