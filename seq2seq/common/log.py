import logging

def info(msg):
    logging.basicConfig(format='%(asctime)s:%(message)s', level=logging.DEBUG)
    logging.info(msg)

def error(msg):
    logging.basicConfig(format='%(asctime)s:%(message)s')
    logging.error(msg)

