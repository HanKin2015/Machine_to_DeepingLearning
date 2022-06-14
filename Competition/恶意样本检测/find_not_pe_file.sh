#!/bin/bash

TRAIN_WHITE_PATH='./AIFirst_data/train/white/'                 # ѵ����������·��
TRAIN_BLACK_PATH='./AIFirst_data/train/black/'                 # ѵ����������·��
TEST_PATH='./AIFirst_data/test/'                               # ���Լ�����·��
DATA_PATH='./data/'                                            # ����·��
TRAIN_WHITE_STRINGS_PATH=${DATA_PATH}'train_white_strings.csv' # ѵ�������������ݼ�·��
TRAIN_BLACK_STRINGS_PATH=${DATA_PATH}'train_black_strings.csv' # ѵ�������������ݼ�·��
TEST_STRINGS_PATH=${DATA_PATH}'test_strings.csv'               # ���Լ��������ݼ�·��

TRAIN_DIRTY_DATASET_PATH='./dataset/train_dirty_dataset.csv' # ѵ���������ݼ��ļ���
TEST_DIRTY_DATASET_PATH='./dataset/test_dirty_dataset.csv'  # ���Լ������ݼ��ļ���
DIRTY_FILES_PATH='./dirty_files/'

IN='/media/sangfor/vdb/study/udev/'
OUT='/media/sangfor/vdb/study/udev/not_pe'
CSV='/media/sangfor/vdb/study/udev/result.csv'

# ʹ��file�����ж��ļ��Ƿ���pe�ļ�
function get_not_pefile()
{
    dir_path=$1
    move_path=$2
    echo ${dir_path} ${move_path}

    for file_name in `ls ${dir_path}`; do
        file_path=${dir_path}${file_name}
        if file ${file_path} | grep -q PE; then
            echo "${file_name} is a PE file."
        else
            echo "${file_name} is not a PE file."
            cp ${file_path} ${move_path}
        fi
    done
}

# ͨ����ȡ�ļ���һ���ļ�����ȡpe�ļ�
function get_not_pefile_read_csv()
{
    dir_path=$1
    move_path=$2
    csv_path=$3
    echo ${dir_path} ${move_path} ${csv_path}
    
    while read line
    do
        file_name=`echo ${line} | awk -F , '{print$1}'`
        echo ${file_name}
        cp ${dir_path}${file_name} ${move_path}
    done < ${csv_path}
}

#get_not_pefile ${IN} ${OUT}
#get_strings ${TRAIN_WHITE_PATH} ${TRAIN_WHITE_STRINGS_PATH}
#get_strings ${TRAIN_BLACK_PATH} ${TRAIN_BLACK_STRINGS_PATH}
#get_strings ${TEST_PATH} ${TEST_STRINGS_PATH}

#get_not_pefile_read_csv ${IN} ${OUT} ${CSV}
#get_not_pefile_read_csv ${TRAIN_DIRTY_DATASET_PATH} ${DIRTY_FILES_PATH}
get_not_pefile_read_csv ${TEST_PATH} ${DIRTY_FILES_PATH} ${TEST_DIRTY_DATASET_PATH}