import logging

def newloggingfunction(category, rundate):
    global print
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler("federated_learning/log/" + logging.FileHandler(category + str(rundate) + "_log.txt", "w"))
    print = logger.info
    return (logger.info)

