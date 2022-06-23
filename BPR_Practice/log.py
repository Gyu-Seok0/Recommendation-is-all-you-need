# https://hamait.tistory.com/880
import logging

def get_logger():
    logger = logging.getLogger("BPR")
    logger.setLevel(logging.INFO)

    stream_hander = logging.StreamHandler()
    logger.addHandler(stream_hander)

    file_handler = logging.FileHandler('BPR.log')
    logger.addHandler(file_handler)

    return logger


if __name__ == '__main__':
    get_logger()