import depthai as dai
import numpy as np
from pathlib import Path
from enum import Flag, auto


__all__ = ['PathLibrary', 'BlobPaths', 'Lens', 'Device']

#-------------------------------------------------------------------------------------------------------------------------------
# Class containing useful paths in this project
#-------------------------------------------------------------------------------------------------------------------------------

class RunMode(Flag):
	CROPPED_STACK = auto()
	PREVIEW = auto()
	VERBOSE = auto()
	

class PathLibrary:
	class DetectionNetwork:
		_EAST_DETECTION: Path = (Path('.') / 'models' / 'east_text_detection_256x256_openvino_2021.2_6shave.blob').absolute()
		_TEXT_DETECTION_0003: Path = (Path('.') / 'models' / 'text-detection-0003_openvino_2022.1_6shave.blob').absolute()
		_TEXT_DETECTION_0004: Path = (Path('.') / 'models' / 'text-detection-0004_openvino_2022.1_6shave.blob').absolute()

	class RecognitionNetwork:
		_TEXT_RECOGNITION_0012: Path = (Path('.') / 'models' / 'text-recognition-0012_openvino_2021.2_6shave.blob').absolute()
	
 
#-------------------------------------------------------------------------------------------------------------------------------
# Class containing paths to detection and recognition networks, choose the models for this project here
#-------------------------------------------------------------------------------------------------------------------------------
class BlobPaths:
	DETECTION_NETWORK: Path = PathLibrary.DetectionNetwork._EAST_DETECTION
	RECOGNITION_NETWORK: Path = PathLibrary.RecognitionNetwork._TEXT_RECOGNITION_0012


#-------------------------------------------------------------------------------------------------------------------------------
# Class containing camera's lens settings
#-------------------------------------------------------------------------------------------------------------------------------
class Lens:
	pass


#-------------------------------------------------------------------------------------------------------------------------------
# Class with device settings
#-------------------------------------------------------------------------------------------------------------------------------
class Device:
	PREVIEW_SIZE: tuple[int, int] = (256,256)
	VIDEO_SIZE: tuple[int, int] = (512, 512) # VIDEO_SIZE >= PREVIEW_SIZE
	FPS: int = 22 # > 0
	BOARD_SOCKET: dai.CameraBoardSocket = dai.CameraBoardSocket.CAM_A
	INTERLEAVED: bool = False
	SENSOR_RESOLUTION: dai.ColorCameraProperties.SensorResolution = dai.ColorCameraProperties.SensorResolution.THE_1080_P
	VID_TO_PREV_RATIO_X: float | None = None
	VID_TO_PREV_RATIO_Y: float| None = None
	
	@classmethod
	def _calculate_vid_prev_ratio_x(cls) -> float:
		cls.VID_TO_PREV_RATIO_X =  cls.VIDEO_SIZE[0] / cls.PREVIEW_SIZE[0]

	@classmethod
	def _calculate_vid_prev_ratio_y(cls) -> float:
		cls.VID_TO_PREV_RATIO_Y = cls.VIDEO_SIZE[1] / cls.PREVIEW_SIZE[1]


Device._calculate_vid_prev_ratio_x()
Device._calculate_vid_prev_ratio_y()
