import logging
import rospy

from logging import * # so this module can be used as a drop-in replacement for logging

class ConnectPythonLoggingToRos(logging.Handler):
    MAP = {
        logging.DEBUG: rospy.logdebug,
        logging.INFO: rospy.loginfo,
        logging.WARNING: rospy.logwarn,
        logging.ERROR: rospy.logerr,
        logging.CRITICAL: rospy.logfatal,
    }

    def emit(self, record):
        try:
            self.MAP[record.levelno]("%s: %s" % (record.name, record.getMessage()))
        except KeyError:
            rospy.logerr("unknown log level %s LOG: %s: %s" % (record.levelno, record.name, record.getMessage()))

def getLogger(name):
    logger = logging.getLogger(name)
    # reconnect logging calls which are children of this to the ros log system
    logger.addHandler(ConnectPythonLoggingToRos())
    # logs sent to children of trigger with a level >= this will be redirected to ROS
    logger.setLevel(logging.DEBUG)
    return logger
