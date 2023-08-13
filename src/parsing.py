# Python standard libraries
import configparser
from typing import List

def parse(predictions: List[str], expected: str | None = None) -> str:
    config = configparser.ConfigParser()
    config.read('config.ini')
    shape_mapping_dict = config['Shapes']
    shape_mapping = {k: v for k, v in shape_mapping_dict.items()}
    
    symbols = [shape_mapping[p] for p in predictions]
    
    code = ''.join(symbols)
    
    if expected is not None:
        print(f'Result: {code}')
        print(f'Expected: {expected}')
    
    return code
