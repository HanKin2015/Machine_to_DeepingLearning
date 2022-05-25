# -*- coding: utf-8 -*-
"""
文 件 名: multithreading.py
文件描述: 线程池和进程池
作    者: HanKin
创建日期: 2022.05.25
修改日期：2022.05.25

Copyright (c) 2022 HanKin. All rights reserved.
"""

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

# 线程池容量
THREAD_NUM = 10

data = []

def test(x, y):
    print('x = {}, y = {}.'.format(x, y))
    data.append(x)
    return x

def main():
    begin_time = time.time()

    tasks = []

    with ThreadPoolExecutor(THREAD_NUM) as pool:
    #with ProcessPoolExecutor(THREAD_NUM) as pool:
        for arg in range(10000):
            task = pool.submit(test, arg, arg)
            tasks.append(task)
            #print(task.result())

    diff_cnt = 0
    for i, elem in enumerate(data):
        #print('i = {}, arg = {}.'.format(i, task.result()))
        if i != elem:
            diff_cnt += 1
            #print('i = {}, elem = {}.'.format(i, elem))

    print('diff_cnt = {}.'.format(diff_cnt))

    end_time = time.time()
    print('共花费 {} s时间'.format(round(end_time - begin_time, 2)))

if __name__ == '__main__':
    main()
