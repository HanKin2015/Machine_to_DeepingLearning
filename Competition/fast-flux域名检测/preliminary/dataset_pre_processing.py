# -*- coding: utf-8 -*-
"""
文 件 名: dataset_pre_processing.py
文件描述: 数据预处理，保存文件是否DataFrame内存减少消耗
作    者: 重在参与快乐加倍队
创建日期: 2022.11.23
修改日期：2022.11.23

Copyright (c) 2022 ParticipationDoubled. All rights reserved.
"""

from common import *

def main():
    """主函数
    """

    
    gc.collect()
    
    # 保存数据集
    agg_df.to_csv(PRE_DATASET_PATH, sep=',', encoding='utf-8', index=False)

if __name__ == '__main__':
    """程序入口
    """
    
    #os.system('chcp 936 & cls')
    logger.info('******** starting ********')
    start_time = time.time()

    main()

    end_time = time.time()
    logger.info('process spend {} s.\n'.format(round(end_time - start_time, 3)))