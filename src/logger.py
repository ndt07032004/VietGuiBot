#ghi nhât ký
import logging

def setup_logger(name = "Tour-Guide-AI" ):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch=logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(ch)
    return logger


logger = setup_logger()
logger.info("Rag bắt đầu chạy")
logger.debug("gỡ lỗi")
logger.error("lỗi")
logger.critical("lỗi nghiêm trọng")