import logging


def log_display(epoch, global_step, time_elapse, **kwargs):
    display = "epoch=" + str(epoch) + "\tglobal_step=" + str(global_step)
    for key, value in kwargs.items():
        if type(value) == str:
            display = "\t" + key + "=" + value
        else:
            display += "\t" + str(key) + "=%.4f " % value
    display += "\ttime=%.2fit/s" % (1.0 / time_elapse)
    return display


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    logger = logging.getLogger(name)
    logging.basicConfig(
        format="[%(asctime)s] - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=level,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logger
