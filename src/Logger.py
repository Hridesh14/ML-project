import logging
import os
from datetime import datetime

log_file  = f'{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log'
logs_Path = os.path.join(os.getcwd(),'logs',log_file)
os.makedirs(logs_Path,exist_ok=True)
log_file_Path = os.path.join(logs_Path,log_file)

logging.basicConfig(
    filename=log_file_Path,
    format='[%(asctime)s]%(lineno)d %(name)s- %(levelname)s- %(message)s',
    level=logging.INFO,
)