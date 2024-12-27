import numpy as np
import depthai as dai
import cv2


class RRect:
    def __init__(self, topleft: tuple[float, float], botright: tuple[float, float], angle: float) -> None:
   
        # 
        #   *-----------------------* 
        #   |           |           |
        #   |           |           | halfheight
        #   |           | centre    |
        #   |-----------X-----------|--
        #   |           |           |
        #   |           |           | halfheight
        #   |           |           |
        #   *-----------------------*
        #     halfwidth | halfwidth
        __slots__ = ['x', 'y', 'halfwidth', 'halfheight', 'angle']
        self.x: float = (topleft[0] + botright[0]) / 2
        self.y: float = (topleft[1] + botright[1]) / 2
        self.halfwidth: float = abs(topleft[0] - botright[0]) / 2
        self.halfheight: float = abs(topleft[1] - botright[1]) / 2
        self.angle: float = -1 * angle


    @property
    def _unrotated_corner_points(self) -> tuple[tuple[int,int]]:
        """Corner points of the rectangle

        Returns
        -------
        tuple[tuple[int,int]]
            0: left-bottom, next in counterclockwise direction
        """
        return ((int(self.x - self.halfwidth), int(self.y - self.halfheight)),
                (int(self.x + self.halfwidth), int(self.y - self.halfheight)),
                (int(self.x + self.halfwidth), int(self.y + self.halfheight)),
                (int(self.x - self.halfwidth), int(self.y + self.halfheight)))


    def get_rotated_points(self) -> np.array:
        """Returns coordinates of corners of a rotated rectangle

        Returns:
            np.array (Point, Point, Point, Point): list of 4 corners
        """
        #  D                       C
        #   *---------------------* 
        #   |                     |
        #   |                     |
        #   |                     |
        #   |                     |
        #   |                     |
        #   |                     |
        #   *---------------------*
        # A                         B
        #
        

        cos = np.cos(self.angle)
        sin = np.sin(self.angle)
        
        # operations below perform rectangle rotation around its centre
        matrix = np.array([[cos,    -sin,   self.x*(1-cos) + self.y*sin ],
                           [sin,     cos,   self.y*(1-cos) - self.x*sin ],
                           [ 0,       0,    1                           ]])

    
        tmp = np.array([[self.x - self.halfwidth], [self.y + self.halfheight], [1]])
        tmp_prime: np.ndarray = np.dot(matrix, tmp)
        A = [int(tmp_prime[0][0]), int(tmp_prime[1][0])]
        
        tmp = np.array([[self.x + self.halfwidth], [self.y + self.halfheight], [1]])
        tmp_prime: np.ndarray = np.dot(matrix, tmp)
        B = [int(tmp_prime[0][0]), int(tmp_prime[1][0])]
        
        tmp = np.array([[self.x + self.halfwidth], [self.y - self.halfheight], [1]])
        tmp_prime: np.ndarray = np.dot(matrix, tmp)
        C = [int(tmp_prime[0][0]), int(tmp_prime[1][0])]
        
        tmp = np.array([[self.x - self.halfwidth], [self.y - self.halfheight], [1]])
        tmp_prime: np.ndarray = np.dot(matrix, tmp)
        D = [int(tmp_prime[0][0]), int(tmp_prime[1][0])]
    
        return np.array([A.copy(), B.copy(), C.copy(), D.copy()])

    
    def get_depthai_RotatedRect(self) -> dai.RotatedRect:
        """Retrieve depthai.RotatedRect"""
        # dai.RotatedRect is in format 
        # [[x_centre, y_centre], [width, height], angle]
        rr: dai.RotatedRect = dai.RotatedRect()
        rr.center.x   = int(self.x)
        rr.center.y   = int(self.y)
        rr.size.width = 2 * int(self.halfwidth)
        rr.size.height = 2 * int(self.halfheight)
        rr.angle = self.angle
        return rr


    def get_cv_RotatedRect(self) -> cv2.RotatedRect:
        """Retrieve cv2.RotatedRect"""
        # [[x_centre, y_centre], [width, height], angle]
        return cv2.RotatedRect(self.listint(), [2 * self.halfwidth, 2 * self.halfheight],  np.rad2deg(self.angle))
    

    def scale(self, scale_factor: float):
        """Scale coordinates and dimentions"""
        self.x *= scale_factor
        self.y *= scale_factor
        self.halfwidth *= scale_factor
        self.halfheight *= scale_factor


    def scalex(self, scale_factor_x: float) -> None:
        """Scale coordinates and dimentions in x direction"""
        self.x *= scale_factor_x
        self.halfwidth *= scale_factor_x


    def scaley(self, scale_factor_y: float) -> None:
        """Scale coordinates and dimentions in y direction"""
        self.y *= scale_factor_y
        self.halfheight *= scale_factor_y


    def __str__(self) -> str:
        return f'{self.__class__.__name__}(centre=({self.x}, {self.y}), width={2*self.halfwidth}, height={2*self.halfheight} angle={self.angle})'