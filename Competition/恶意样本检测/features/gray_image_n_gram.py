# -*- coding: utf-8 -*-
"""
文 件 名: gray_image_n_gram.py
文件描述: 获取二进制文件的灰度图像和n-gram特征(草稿测试)
备    注: 安装capstone库(pip install capstone)
作    者: HeJian
创建日期: 2022.06.14
修改日期：2022.06.14
Copyright (c) 2022 HeJian. All rights reserved.
"""

from common import *

def get_image_width(file_path):
    """获取图像宽度
    根据论文《Malware Images: Visualization and Automatic Classification》
    文件大小KB
    """

    file_size = os.path.getsize(file_path) / 1024
    logger.info('file size: {} KB.'.format(file_size))
    
    if file_size < 10: return 32
    elif file_size < 30: return 64
    elif file_size < 60: return 128
    elif file_size < 100: return 256
    elif file_size < 200: return 384
    elif file_size < 500: return 512
    elif file_size < 1000: return 768
    return 1024

def binary_file_to_grayscale_image(root, file):
    """恶意样本灰度图像绘制
    """
    
    file_path = '{}{}'.format(root, file)
    image_width = get_image_width(file_path)
    logger.info('file[{}] grayscale image width: {}.'.format(file, image_width))

    with open(file_path, 'rb') as fd:
        content = fd.read()
    hexst = binascii.hexlify(content)  #将二进制文件转换为十六进制字符串
    file_bytes = np.array([int(hexst[i:i+2],16) for i in range(0, len(hexst), 2)])  #按字节分割
    image_hight = int(len(file_bytes)/image_width)
    logger.info('grayscale image size: {}x{}.'.format(image_width, image_hight))
    
    matrix = np.reshape(file_bytes[:image_hight*image_width], (-1, image_width))  #根据设定的宽度生成矩阵
    matrix = np.uint8(matrix)
    
    im = Image.fromarray(matrix)    #转换为图像
    im.save('{}.png'.format(file))

def grayscale_image_progressing(filename):
    pass

def get_binary_code(root, file):
    file_path = '{}{}'.format(root, file)
    pe = pefile.PE(file_path)
    
    #从程序头中获取程序入口点的地址
    entrypoint = pe.OPTIONAL_HEADER.AddressOfEntryPoint
    #计算入口代码被加载到内存中的内存地址
    entrypoint_address = entrypoint+pe.OPTIONAL_HEADER.ImageBase
    #从PE文件对象获取二进制代码
    binary_code = pe.get_memory_mapped_image()[entrypoint:-1]
    #print(binary_code)
    logger.info('len(binary_code) = {}'.format(len(binary_code)))
    #初始化反汇编程序以反汇编32位x86二进制代码
    disassembler = Cs(CS_ARCH_X86, CS_MODE_32)
    #反汇编代码
    opcode_seq = []
    for instruction in disassembler.disasm(binary_code, entrypoint_address):
        logger.debug('{}\t{}.'.format(instruction.mnemonic, instruction.op_str))
        if instruction.mnemonic not in ['int1', 'int3', 'align']:
            opcode_seq.append(instruction.mnemonic)
    logger.info('opcode_seq count: {}.'.format(opcode_seq))
    return opcode_seq

def get_n_gram(root, file):
    ops = get_binary_code(root, file)
    print(ops)
    opngram = getOpcodeNgram(ops)
    print(opngram)
    
# 根据Opcode序列，统计对应的n-gram
def getOpcodeNgram(ops ,n = 3):
    opngramlist = [tuple(ops[i:i+n]) for i in range(len(ops)-n)]
    opngram = Counter(opngramlist)
    return opngram

def main():
    root = 'D:\\Github\\Machine_to_DeepingLearning\\Competition\\恶意样本检测\\features\\backup\\'
    file = 'FTOOL.exe'
    #binary_file_to_grayscale_image(root, file)
    #get_binary_code(root, file)
    file_path = './0ACDbR5M3ZhBJajygTuf.asm'
    get_n_gram(root, file)

if __name__ == '__main__':
    start_time = time.time()

    main()

    end_time = time.time()
    logger.info('process spend {} s.'.format(round(end_time - start_time, 3)))











