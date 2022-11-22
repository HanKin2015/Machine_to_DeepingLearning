# -*- coding: utf-8 -*-
"""
文 件 名: main.py
文件描述: 主程序
作    者: HeJian
创建日期: 2022.10.26
修改日期：2022.10.26
Copyright (c) 2022 HeJian. All rights reserved.
"""

import time
from log import logger
import get_data
import feature_engineering
import training_model
import predict_result

def main():
    feature_engineering.main()
    training_model.main()
    
if __name__ == '__main__':
    #os.system('chcp 936 & cls')
    logger.info('******** starting ********')
    start_time = time.time()

    main()

    end_time = time.time()
    logger.info('process spend {} s.\n'.format(round(end_time - start_time, 3)))