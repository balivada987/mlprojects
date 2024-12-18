import sys
import logging
from src.logger import logging






def error_msg_detail(error, error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    filename=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(filename, exc_tb.tb_lineno,
                                                                                                           str(error))
    return error_message
class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        error_message=super().__init__(error_message)
        self.error_message=error_msg_detail(error=error_message, error_detail=error_detail)
    
    def __str__(self):
        return self.error_message
    


if __name__=="__main__":
    try:
        a=1/0
    except Exception as error:
        print("Hi")
        logging.info("Division error")
        raise CustomException(error,sys )