import logging

def setup_logger(logger_name, log_file, level=logging.DEBUG):
    """ Function to setup a logger """

    formatter = logging.Formatter('%(levelname)s: %(asctime)s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')
    logger = logging.getLogger(logger_name)

    # File handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(formatter)

    # Stream handler
    # stream_handler = logging.StreamHandler()
    # stream_handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(file_handler)
    # logger.addHandler(stream_handler)

    return logger

# Create loggers
loggers = {
    'dialogue': setup_logger('__dialogue__', 'logs/dialogue_logger.log'),
    'runner': setup_logger('__runner__', 'logs/runner_logger.log'),
    'debug': setup_logger('__debug__', 'logs/debug_logger.log')
}
