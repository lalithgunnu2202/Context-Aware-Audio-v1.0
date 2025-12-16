import sys

def error_msg_detail(error, error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_msg = "Error occured in script name [{0}] line [{1}] error message is [{2}] ".format(
        file_name,
        exc_tb.tb_lineno,
        str(error)
    )
    return error_msg

class CustomException(Exception):
    def __init__(self, error_msg, error_detail:sys):
        self.__init__(error_msg)
        self.error_msg = error_msg_detail(error_msg,error_detail=error_detail)

    def __str__(self):
        return self.error_msg