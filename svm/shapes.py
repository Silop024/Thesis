from dataclasses import dataclass
from enum import Enum
import configparser

class Shape(Enum):
    Undefined = "-1" # One-class svm classifier marks it as a shape it has not seen before.
    Unknown = "1"    # Multi-class svm classifier marks it as a shape that it has seen before.
    OneByOne = "1x1"
    OneByTwo = "1x2"
    OneByThree = "1x3"
    OnyByFour = "1x4"
    TwoByTwo = "2x2"
    TwoByThree = "2x3"
    TwoByFour = "2x4"
    
    def to_keyword(self) -> str:
        if self.is_keyword() and self != Shape.Unknown:
            return shape_mapping[self]
        else:
            return ''
    
    def is_keyword(self) -> bool:
        return self != Shape.Undefined
    
    

@dataclass
class ShapePrediction:
    """Class for keeping track of the classification of a shape
    
    Fields:
        index (int) The position of the shape in relation to other shapes, 0 is the first shape.

        position (tuple) The absolute position of the shape in the image (x, y).
    
        is_keyword (bool) A flag to know if the shape is a restricted keyword
    
        keyword (str) The keyword that the shape represents, if any.
    
        id (int) The id of the shape, given that is does not represent a keyword.
        
        image_path (str) The path to the image where the shape was classified. For debugging.
    """
    index: int
    position: tuple
    shape: Shape
    probability: float
    id: int
    image_path: str
    
    @property
    def is_keyword(self) -> bool:
        return self.shape.is_keyword()

    @property
    def keyword(self) -> str:
        return self.shape.to_keyword() if self.is_keyword else ""
        

    def __str__(self) -> str:
        string = f'index: {self.index}\nposition: {self.position}\n'
        if self.shape.is_keyword():
            string += f'keyword: {self.keyword}\nprobability: {self.probability}\n'
        else:
            string += f'id: {self.id}\n'
        return string
    
    
    def __repr__(self) -> str:
        return f'{self.index}: {self.shape.name}'
    
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ShapePrediction):
            return False
        
        return self.index == other.index and self.image_path == other.image_path


config = configparser.ConfigParser()
config.read('config.ini')
shape_mapping_dict = config['Shapes']
shape_mapping = {Shape(k): v for k, v in shape_mapping_dict.items()}
