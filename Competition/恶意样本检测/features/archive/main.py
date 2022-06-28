# -*- coding: utf-8 -*-
"""
文 件 名: main.py
文件描述: 主程序
作    者: HeJian
创建日期: 2022.06.16
修改日期：2022.06.16
Copyright (c) 2022 HeJian. All rights reserved.
"""

import time
from log import logger
import get_data
import get_opcode_n_gram
import opcode_n_gram_features
import features
import combine
import predict

def main():
    """程序入口，需要先运行脚本
    """
    
    get_data.main()
    get_opcode_n_gram.main()
    opcode_n_gram_features.main()
    features.main()
    combine.main()
    predict.main()
    
if __name__ == '__main__':
    start_time = time.time()

    main()

    end_time = time.time()
    logger.info('process spend {} s.'.format(round(end_time - start_time, 3)))