import depthai as dai
from pathlib import Path
import datetime

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
    manip_out_xout = pipeline.create(dai.node.XLinkOut)

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
    manip_out_xout.setStreamName('manip_out')

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
    manip.out.link(manip_out_xout.input)
    recnn.out.link(recnn_out_xout.input)

    return pipeline
