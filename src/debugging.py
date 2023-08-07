# Python standard libraries
import threading
from enum import IntEnum

# Installed with pip
import cv2


class Debug:
    verbosity: int = 0
    
    @staticmethod
    def set_verbosity(level: int):
        Debug.verbosity = level
    
    @staticmethod  
    def print(text: str, level: int = 0, color: str = None):
        if Debug.verbosity >= level:
            match color:
                case 'red':
                    Debug.print_red(text)
                case 'green':
                    Debug.print_green(text)
                case 'blue':
                    Debug.print_blue(text)
                case _:
                    print(text)

    @staticmethod
    def print_red(text: str):
        print("\033[91m {}\033[00m".format(text))
    
    @staticmethod
    def print_green(text: str):
        print("\033[92m {}\033[00m".format(text))
    
    @staticmethod
    def print_blue(text: str):
        print("\033[94m {}\033[00m".format(text))
        
        
    @staticmethod
    def display_image(image):
        display_thread = threading.Thread(target=Debug.__show_image, args=(image))
        display_thread.start()
        

    def __show_image(image):
        cv2.imshow("Resulting image after preprocessing", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

class ExitCode(IntEnum):
    UserError = 1
    
    