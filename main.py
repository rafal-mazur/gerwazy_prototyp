# TODO: cropped stack

import argparse
import depthai as dai
import cv2
import numpy as np
import time
import sys
import copy

import decoding.east256x256 as east
import decoding.text_recognition_0012 as tr12
from utils.pipeline import create_pipeline
from utils.Logger import Logger, Color
from utils.communication import SerialPort

logger = Logger()

def parse_args():
    parser = argparse.ArgumentParser(prog='Gerwazy')
    parser.add_argument('--port', type=str, default='/dev/serial0', help='Name of the serial port to use')
    parser.add_argument('--baudrate', type=int, default=9600)
    parser.add_argument('--bytesize', type=int, default=8)
    parser.add_argument('--parity', type=str, default='N')
    parser.add_argument('--stopbits', type=int, default=1)
    parser.add_argument('--timeout', type=float|None, default=None)
    parser.add_argument('-p', '--preview', action='store_true', help='Show preview with bounding boxes')
    parser.add_argument('-cs', '--cropped_stack', action='store_true', help='Show window with stacked text regions')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print additional info to the console')

    return parser.parse_args()


def main(args):
    logger('Creating pipeline...')
    pipeline: dai.Pipeline = create_pipeline()
    logger.done()

    logger('Opening serial port and device...')

    
    with dai.Device(pipeline) as device, SerialPort(args.port,
                                                    args.baudrate,
                                                    args.bytesize,
                                                    args.parity,
                                                    args.stopbits,
                                                    args.timeout) as port:
        logger.done()
        logger('USB speed:', device.getUsbSpeed().name)

        logger(f'\nAvaillable input queues: {device.getInputQueueNames()}')
        logger(f'Availlable output queues: {device.getOutputQueueNames()}\n')
        logger('Creating queues...')

        q_cam_ctrl: dai.DataInputQueue  = device.getInputQueue('cam_ctrl', 1, blocking=False)
        q_manip_img: dai.DataInputQueue = device.getInputQueue('manip_img', 4, blocking=False)
        q_manip_cfg: dai.DataInputQueue = device.getInputQueue('manip_cfg', 4, blocking=False)

        q_detnn_out: dai.DataOutputQueue  = device.getOutputQueue('detnn_out', 1, blocking=False)
        q_detnn_pass: dai.DataOutputQueue = device.getOutputQueue('detnn_pass', 1, blocking=False)
        q_manip_out: dai.DataOutputQueue  = device.getOutputQueue('manip_out', 1, blocking=False)
        q_recnn_out: dai.DataOutputQueue  = device.getOutputQueue('recnn_out', 1, blocking=False)

        logger.done()

        ctrl: dai.CameraControl = dai.CameraControl()
        ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
        ctrl.setAutoFocusTrigger()
        q_cam_ctrl.send(ctrl)
        del ctrl

        logger('\nStarting main loop\n')

        while True:
            time.sleep(0.01)
            
            # get detection output
            detnn_output: dai.NNData = q_detnn_out.get()
            detnn_pass: np.ndarray = q_detnn_pass.get().getCvFrame()
            preview_frame: np.ndarray = copy.deepcopy(detnn_pass)

            # decode detection;
            for idx, (rect, _) in enumerate(east.decode(detnn_output)):
                # Crop rectangle
                cfg: dai.ImageManipConfig = dai.ImageManipConfig()
                cfg.setCropRotatedRect(rect.get_depthai_RotatedRect(), False)
                cfg.setResize(120, 32)

                if idx == 0:
                    w, h, _ = detnn_pass.shape
                    imgFrame = dai.ImgFrame()
                    imgFrame.setData(detnn_pass.transpose(2, 0, 1).flatten())
                    imgFrame.setType(dai.ImgFrame.Type.BGR888p)
                    imgFrame.setWidth(w)
                    imgFrame.setHeight(h)
                    q_manip_img.send(imgFrame)
                else:
                    # if there is more than one detection (idx != 0) reuse image
                    cfg.setReusePreviousImage(True)
                q_manip_cfg.send(cfg)
                
                # add bounding box to image
                if args.preview:
                    preview_frame = cv2.polylines(preview_frame, [rect.get_rotated_points()], True, (255,0,0), 1, cv2.LINE_8)
            
            # show preview
            if args.preview:
                cv2.imshow('preview', preview_frame)


            # harvest all availlable recognitions
            while True:
                recnn_out: dai.NNData|None = q_recnn_out.tryGet()

                if recnn_out is None:
                    break
                
                # send to another device
                port.send(tr12.decode(recnn_out))

            
            if cv2.waitKey(1) == ord('q'):
                break


if __name__ == '__main__':
    args = parse_args()
    logger.set_logging(args.verbose)
    logger('Starting program\n', color=Color.HEADER)
    main(args)

