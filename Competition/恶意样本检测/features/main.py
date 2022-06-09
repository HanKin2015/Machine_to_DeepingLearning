import time
from log import logger
import get_data
import features
import training
import predict

def main():
    features.main()
    training.main()
    
if __name__ == '__main__':
    start_time = time.time()

    main()

    end_time = time.time()
    logger.info('process spend {} s.'.format(round(end_time - start_time, 3)))