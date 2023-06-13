from dataclasses import dataclass

@dataclass
class ShapePrediction:
    """Class for keeping track of the classification of a shape
    
    Fields:
        index -- The position of the shape in relation to other shapes, 0 is the first shape.

        position -- The absolute position of the shape in the image (x, y).
    
        is_keyword -- A flag to know if the shape is a restricted keyword
    
        keyword -- The keyword that the shape represents, if any.
    
        id -- The id of the shape, given that is does not represent a keyword.
    """
    index: int
    position: tuple
    is_keyword: bool
    keyword: str
    id: int

    def __str__(self):
        string = f'index: {self.index}\nposition: {self.position}\n'
        if (self.is_keyword):
            string += f'keyword: {self.keyword}\n'
        else:
            string += f'id: {self.id}\n'
        return string