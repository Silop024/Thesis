# Python standard libraries
import configparser
from typing import List, Tuple

# Installed with pip
import cv2

def parse(predictions: List[str], expected: str | None = None) -> Tuple[str, List]:
    config = configparser.ConfigParser()
    config.read('config.ini')
    shape_mapping_dict = config['Shapes']
    shape_mapping = {k: v for k, v in shape_mapping_dict.items()}
    
    symbols = [shape_mapping[p] for p in predictions]
    
    parsed_code = ''.join(symbols)
    
    wrong_indices = []
    if expected is not None:
        percentage_correct, wrong_indices = get_amount_correctly_parsed(expected, parsed_code)
        print(f'{percentage_correct}% correctly parsed')
        
        if percentage_correct != 100:
            for (i, expected_symbol, parsed_symbol) in wrong_indices:
                print(f'Wrong at index {i}, expected {expected_symbol}, got {parsed_symbol}')
    
    
    return parsed_code, wrong_indices


def get_amount_correctly_parsed(expected: str, parsed: str) -> Tuple[float, List[Tuple[int, chr, chr]]]:
    wrong = []
    
    for i, (expected_symbol, parsed_symbol) in enumerate(zip(expected, parsed)):
        if expected_symbol != parsed_symbol:
            wrong.append((i, expected_symbol, parsed_symbol))
            
    if len(wrong) == 0:
        return 100, []
    
    wrong_ratio = len(wrong) / len(parsed)
    
    return 100 - (100 * wrong_ratio), wrong