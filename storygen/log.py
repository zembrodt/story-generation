# log.py
import datetime
import os

LOG_DIR = 'logs/'
FILE_EXT='log'
LOG_TAGS = ['[INFO]', '[DEBUG]', '[ERROR]']
PADDING = max(map(len, LOG_TAGS))

class Log:
    def __init__(self, log_dir=LOG_DIR, file_ext=FILE_EXT):
        t = datetime.datetime.now()
        self.dir = '{}log_{}{:02d}{:02d}_{:02d}{:02d}{:02d}/'.format(log_dir, t.year, t.month, t.day, t.hour, t.minute, t.second)
        self.file_ext = file_ext
        self.logfiles = []
    def create(self, logname):
        t = datetime.datetime.now()
        logfile = '{}{}.{}'.format(self.dir, logname, self.file_ext)
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        with open(logfile, 'w+') as f:
            f.write('Log file created at {}\n'.format(t))
        self.logfiles.append(logfile)
        return logfile
    def _write(self, logfile, tag, content, newline):
        nl = '\n' if newline else ''
        if logfile in self.logfiles:
            t = datetime.datetime.now()
            with open(logfile, 'a') as f:
                f.write(('{:<'+str(PADDING)+'} {} : {}{}').format(tag, t, content, nl))
        else:
            print('Logfile [{}] doesn\'t exist'.format(logfile))
    def info(self, logfile, content, newline=True):
        self._write(logfile, LOG_TAGS[0], content, newline)
    def debug(self, logfile, content, newline=True):
        self._write(logfile, LOG_TAGS[1], content, newline)
    def error(self, logfile, content, newline=True):
        self._write(logfile, LOG_TAGS[2], content, newline)
