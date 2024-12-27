import numpy as np
import cv2
import depthai as dai


_chars_map: list = list('0123456789abcdefghijklmnopqrstuvwxyz#')


def decode(tr12_output: dai.NNData) -> str:
	coded_texts = np.array(tr12_output.getFirstLayerFp16()).reshape(30, 1, 37)

	texts: list = []
	index: int = 0
	# Select max probabilty (greedy decoding) then decode index to character
	coded_texts = coded_texts.astype(np.float16)
	preds_index = np.argmax(coded_texts, 2).transpose(1, 0)
	preds_sizes = np.array([preds_index.shape[1]] * preds_index.shape[0])
	preds_index = preds_index.reshape(-1)

	for l in preds_sizes:
		t = preds_index[index:index + l]

		# NOTE: t might be zero size
		if t.shape[0] == 0:
			continue

		texts.append(''.join([_chars_map[t[i]] for i in range(l) if _chars_map[t[i]] != '#' and not (i > 0 and t[i - 1] == t[i])]))
		index += l

	return texts[0]
