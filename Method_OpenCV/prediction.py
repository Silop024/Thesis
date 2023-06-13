import cv2
from dataclasses import dataclass
import preprocessing

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
    is_keyword: bool
    keyword: str
    id: int
    image_path: str
    
    def show(self):
        image = preprocessing.read_image_and_scale(self.image_path)
        cv2.putText(image, self.__repr__(), self.position, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 255), thickness=1)
        cv2.imshow('Predictions', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

    def __str__(self) -> str:
        string = f'index: {self.index}\nposition: {self.position}\n'
        if self.is_keyword:
            string += f'keyword: {self.keyword}\n'
        else:
            string += f'id: {self.id}\n'
        return string
    
    
    def __repr__(self) -> str:
        string = f'{self.index}: '
        if self.is_keyword:
            string += self.keyword
        else:
            string += str(self.id)
        return string
    
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ShapePrediction):
            return False
        
        return self.index == other.index and self.image_path == other.image_path