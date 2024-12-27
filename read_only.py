#! /usr/bin/env python3
"""Use only on RPI"""

import depthai as dai
from time import sleep
import datetime
from pathlib import Path

import decoding.east256x256 as east
import decoding.text_recognition_0012 as tr12
from utils.communication import SerialPort


def create_pipeline() -> dai.Pipeline:
    pipeline: dai.Pipeline = dai.Pipeline()

    #------------------------------------------------------------------
    # declarations
    #-----------------------------------------------------------------
    
    cam_control_xin = pipeline.create(dai.node.XLinkIn)
    cam = pipeline.create(dai.node.ColorCamera)

    detnn = pipeline.create(dai.node.NeuralNetwork)
    detnn_sync = pipeline.create(dai.node.Sync)
    detnn_demux = pipeline.create(dai.node.MessageDemux)
    detnn_pass_xout = pipeline.create(dai.node.XLinkOut)
    detnn_out_xout = pipeline.create(dai.node.XLinkOut)

    manip_img_xin = pipeline.create(dai.node.XLinkIn)
    manip_cfg_xin = pipeline.create(dai.node.XLinkIn)
    manip = pipeline.create(dai.node.ImageManip)
    recnn = pipeline.create(dai.node.NeuralNetwork)
    recnn_out_xout = pipeline.create(dai.node.XLinkOut)

    #------------------------------------------------------------------
    # properties
    #------------------------------------------------------------------

    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setInterleaved(False)
    cam.setPreviewSize(256,256)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setFps(2)
    cam_control_xin.setStreamName('cam_ctrl')
    
    detnn.setBlobPath((Path('.')/'models'/'east_text_detection.blob').resolve().absolute())
    detnn_out_xout.setStreamName('detnn_out')
    detnn_pass_xout.setStreamName('detnn_pass')

    detnn_sync.setSyncThreshold(datetime.timedelta(seconds=0.01))

    manip.setWaitForConfigInput(True)
    manip_cfg_xin.setStreamName('manip_cfg')
    manip_img_xin.setStreamName('manip_img')

    recnn.setBlobPath((Path('.')/'models'/'text-recognition-0012.blob').resolve().absolute())
    recnn_out_xout.setStreamName('recnn_out')

    #------------------------------------------------------------------
    # linking
    #------------------------------------------------------------------

    cam_control_xin.out.link(cam.inputControl)
    cam.preview.link(detnn.input)

    # 1st stage
    detnn.out.link(detnn_sync.inputs['demux_out'])
    detnn.passthrough.link(detnn_sync.inputs['demux_pass'])

    # Syncing
    detnn_sync.out.link(detnn_demux.input)
    detnn_demux.outputs['demux_out'].link(detnn_out_xout.input)
    detnn_demux.outputs['demux_pass'].link(detnn_pass_xout.input)

    # 2nd stage
    manip_cfg_xin.out.link(manip.inputConfig)
    manip_img_xin.out.link(manip.inputImage)
    manip.out.link(recnn.input)
    recnn.out.link(recnn_out_xout.input)

    return pipeline



def main():
    pipeline: dai.Pipeline = create_pipeline()

    
    with dai.Device(pipeline) as device, SerialPort('/dev/serial0') as port:
        q_cam_ctrl: dai.DataInputQueue  = device.getInputQueue('cam_ctrl', 1, blocking=False)
        q_manip_img: dai.DataInputQueue = device.getInputQueue('manip_img', 6, blocking=False)
        q_manip_cfg: dai.DataInputQueue = device.getInputQueue('manip_cfg', 6, blocking=False)

        q_detnn_out: dai.DataOutputQueue  = device.getOutputQueue('detnn_out', 1, blocking=False)
        q_detnn_pass: dai.DataOutputQueue = device.getOutputQueue('detnn_pass', 1, blocking=False)
        q_recnn_out: dai.DataOutputQueue  = device.getOutputQueue('recnn_out', 6, blocking=True)

        # set camera settings
        ctrl: dai.CameraControl = dai.CameraControl()
        ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
        ctrl.setAutoFocusTrigger()
        q_cam_ctrl.send(ctrl)
        del ctrl


        while True:
            sleep(0.01)
            detnn_output: dai.NNData = q_detnn_out.get()
            detnn_pass: dai.ImgFrame = q_detnn_pass.get().getCvFrame()

            # decode detection
            for idx, (rect, _) in enumerate(east.decode(detnn_output)):

                # create bounding rect
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


            while True:
                recnn_out: dai.NNData|None = q_recnn_out.tryGet()

                if recnn_out is None:
                    break
                
                # decode and send
                port.send(tr12.decode(recnn_out))
    
if __name__ == '__main__':
    main()
