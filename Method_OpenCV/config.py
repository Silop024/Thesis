import configparser
from shapes import Shape

# Read config file
config = configparser.ConfigParser()
config.read('config.ini')

# Get shape mapping
shape_mapping_dict = config['Shapes']
shape_to_keyword_map = {Shape(k): v for k, v in shape_mapping_dict.items()}

# Get paths
paths = config['paths']