"""Module for decoding east output"""
import numpy as np
import depthai as dai
from utils.geometry import RRect


_conf_threshold: float = 0.5
_overlap_thresh: float = 0.3


def decode(east_output: dai.NNData) -> np.ndarray:
    """Decodes east output into array of tuples ``(RRect, confidence)``

    Parameters
    ----------
    east_output : dai.NNData
        Output of east neural network

    Returns
    -------
    np.ndarray
        Array of tuples ``(RRect, confidence)``
    """
    coded_scores, coded_bboxes, coded_angles = (np.array(east_output.getLayerFp16(tensor.name)) for tensor in east_output.getRaw().tensors)
    coded_scores = coded_scores.reshape(64, 64)
    coded_bboxes = coded_bboxes.reshape(4, 64, 64).transpose(1, 2, 0)
    coded_angles = coded_angles.reshape(64, 64)

    n_rows, n_cols = coded_scores.shape

    # get bboxes with sufficien score (> _conf_thresh)
    scores: list = [] # list of decoded scores
    bboxes: list = [] # list of decoded rects
    angles: list = [] # list of decoded angles


    for y in range(n_rows):
        scores_row = coded_scores[y]
        bbox_row = coded_bboxes[y]
        angle_row = coded_angles[y]

        for x in range(n_cols):
            if scores_row[x] < _conf_threshold:
                continue

            offsetX, offsetY = x * 4., y * 4.

            sin = np.sin(angle_row[x])
            cos = np.cos(angle_row[x])
            h = bbox_row[x, 0] + bbox_row[x, 2]
            w = bbox_row[x, 1] + bbox_row[x, 3]
            
            endX = int(offsetX + (cos * bbox_row[x, 1]) + (sin * bbox_row[x, 2]))
            endY = int(offsetY - (sin * bbox_row[x, 1]) + (cos * bbox_row[x, 2]))
            startX = int(endX - w)
            startY = int(endY - h)


            scores.append(scores_row[x])
            bboxes.append((startX, startY, endX, endY))
            angles.append(angle_row[x])
            
    scores, bboxes, angles = np.array(scores), np.array(bboxes), np.array(angles)

    # apply non max supression to aviod overlap
    if len(scores) == 0:
        return np.array([])
    
    pick: list = []

    # lists of coordinates of bounding boxes
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    
    # compute intersection area
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # argsort by scores (most probable is last)
    idxs = np.argsort(scores)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > _overlap_thresh)[0])))

    scores = scores[pick]
    bboxes = bboxes[pick]
    angles = angles[pick]

    return np.array([(RRect((b[0], b[1]), (b[2], b[3]), a), s) for s, b, a in zip(scores, bboxes, angles)])
