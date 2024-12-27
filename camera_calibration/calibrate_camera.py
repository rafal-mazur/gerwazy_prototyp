import depthai as dai
import cv2
import numpy as np

message =\
"""
T - this help
Q - exit
F/C - contrast up/down
I/J - iso up/down
W/A - white balance up/down
S/Z - saturation up/down
D/X - sharpness up/down
H/B - brightness up/down
P/L - Focus up/down
O/K - luma denoise up/down
G/V - chroma denoise up/down
E/R - exposure time up/down
"""

def clamp(x, low, high):
        if x > high:
            return high
        elif x < low:
            return low
        else:
            return x


class Parameter:
    def __init__(self, minimum, maximum, value = 0, name: str=""):
        self.max = maximum
        self.min = minimum
        self.val = clamp(value, minimum, maximum)
        self.name = name
    

    def __iadd__(self, x: int):
        a = clamp(self.val + x, self.min, self.max)
        print(f'{self.name} = {a}')
        self.val = a
        return self

    def __isub__(self, x: int):
        a = clamp(self.val - x, self.min, self.max)
        print(f'{self.name} = {a}')
        self.val = a
        return self

    

brightness: Parameter = Parameter(-10, 10, name='brightness')
iso: Parameter = Parameter(100, 1600, 1000, name='iso')
saturation: Parameter = Parameter(-10, 10, name='saturation')
sharpness: Parameter = Parameter(0, 4, name='sharpness')
white_balance: Parameter = Parameter(1000, 12000, 5600, name='white balance') 
lens_position: Parameter = Parameter(0, 255, name='lens position')
luma_denoise: Parameter = Parameter(0, 4, 1, name='luma denoise')
chroma_denoise: Parameter = Parameter(0, 4, 1, name='chroma denoise')
contrast: Parameter = Parameter(-10, 10, name='contrast')
exposure_time_us: Parameter = Parameter(100_000, 1_000_000, 100, name='exposure time us')

p = dai.Pipeline()

cam = p.create(dai.node.ColorCamera)
cam_ctrl_xin = p.create(dai.node.XLinkIn)
cam_xout = p.create(dai.node.XLinkOut)

cam.setFps(30)
cam.setVideoSize(1300, 500)
cam_ctrl_xin.setStreamName('cam_ctrl')
cam_xout.setStreamName('cam_out')

cam_ctrl_xin.out.link(cam.inputControl)
cam.video.link(cam_xout.input)


with dai.Device(p) as device:
    q_ctrl: dai.DataInputQueue = device.getInputQueue('cam_ctrl', 1, blocking=False)
    q_out: dai.DataOutputQueue = device.getOutputQueue('cam_out', 1, blocking=False)
    ctrl = dai.CameraControl()
    print(f'{ctrl.getLensPosition()=}\n{ctrl.getExposureTime()=}\n{ctrl.getSensitivity()=}')

    while True:

        frame = q_out.get()

        cv2.imshow('Press \'Q\' to exit', frame.getCvFrame())

        key = cv2.waitKey(1)
        ctrl = dai.CameraControl()

        # quit
        if key == ord('q'):
            break
        elif key == ord('i'):
            iso += 10
            ctrl.setManualExposure(exposure_time_us.val, iso.val)
        elif key == ord('j'):
            iso -= 10
            ctrl.setManualExposure(exposure_time_us.val, iso.val)

        elif key == ord('e'):
            exposure_time_us += 10
            ctrl.setManualExposure(exposure_time_us.val, iso.val)
        elif key == ord('r'):
            exposure_time_us -= 10
            ctrl.setManualExposure(exposure_time_us.val, iso.val)

        elif key == ord('a'):
            white_balance -= 50
            ctrl.setManualWhiteBalance(white_balance.val)
        elif key == ord('w'):
            white_balance += 50
            ctrl.setManualWhiteBalance(white_balance.val)

        elif key == ord('z'):
            saturation -= 1
            ctrl.setSaturation(saturation.val)
        elif key == ord('s'):
            saturation += 1
            ctrl.setSaturation(saturation.val)

        elif key == ord('d'):
            sharpness += 1
            ctrl.setSharpness(sharpness.val)
        elif key == ord('x'):
            sharpness -= 1
            ctrl.setSharpness(sharpness.val)
        
        elif key == ord('b'):
            brightness -= 1
            ctrl.setBrightness(brightness.val)
        elif key == ord('h'):
            brightness += 1
            ctrl.setBrightness(brightness.val)

        elif key == ord('l'):
            lens_position -= 1
            ctrl.setManualFocus(lens_position.val)
        elif key == ord('p'):
            lens_position += 1
            ctrl.setManualFocus(lens_position.val)
        
        elif key == ord('k'):
            luma_denoise += 1
            ctrl.setLumaDenoise(luma_denoise.val)
        elif key == ord('o'):
            luma_denoise -= 1
            ctrl.setLumaDenoise(luma_denoise.val)

        elif key == ord('g'):
            chroma_denoise += 1
            ctrl.setChromaDenoise(chroma_denoise.val)
        elif key == ord('v'):
            chroma_denoise -= 1
            ctrl.setChromaDenoise(chroma_denoise.val)
        
        elif key == ord('f'):
            contrast += 1
            ctrl.setContrast(contrast.val)
        elif key == ord('c'):
            contrast -= 1
            ctrl.setContrast(contrast.val)
        elif key == ord('t'):
            print(message)
        q_ctrl.send(ctrl)