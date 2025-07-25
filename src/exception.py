import sys
import logging
from src.logger import logging

def error_message_detail(error, error_detail:sys):
    a,b,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "error occured in script name [{0}] line number [{1}] error message [{2}]".format(
    file_name, exc_tb.tb_lineno, str(error))
    
    return error_message
    
class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail = error_detail)
        
    
    def __str__(self):
        return self.error_message


if __name__ == "__main__":
    try:
        x = 1/0
    except Exception as ex:
        logging.info("division by zero error")
        raise CustomException(ex, sys)
    finally:
        logging.info("division by zero error handled")
        logging.info("end of script")
    
