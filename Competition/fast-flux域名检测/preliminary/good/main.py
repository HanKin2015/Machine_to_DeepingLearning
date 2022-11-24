# -*- coding: utf-8 -*-
"""
文 件 名: main.py
文件描述: 主程序
作    者: 重在参与快乐加倍队
创建日期: 2022.10.26
修改日期：2022.11.21
Copyright (c) 2022 ParticipationDoubled. All rights reserved.
"""

import time
from log import logger
import feature_engineering
import training_model
import predict_result

def main():
    feature_engineering.main()
    training_model.main()
    predict_result.main()
    
if __name__ == '__main__':
    """程序入口
    """
    
    #os.system('chcp 936 & cls')
    logger.info('******** starting ********')
    start_time = time.time()

    main()

    end_time = time.time()
    logger.info('process spend {} s.\n'.format(round(end_time - start_time, 3)))