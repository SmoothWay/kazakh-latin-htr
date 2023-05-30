import json

import numpy as np
import math
import cv2
from path import Path

# from typing import Optional

from .model import Model, Batch, Preprocessor
# from .ctc import ctc_best_path, ctc_single_word_beam_search, PrefixTree

# singleton instance of the CRNN model

CRNN = Model()
file_path = Path(__file__).abspath().dirname() / 'model'

# with open(file_path / 'model.json') as f:
#     _CHARS = json.load(f)['chars']

def read(img: np.ndarray) -> str:
    """Recognizes text in image provided by file path."""

    preprocessor = Preprocessor((128, 32), dynamic_width=True, padding=16)
    img = preprocessor.process_img(img)

    batch = Batch([img], None, 1)
    text = CRNN.infer_batch(batch)
    print(text)
    # if decoder == 'best_path':
    #     res = ctc_best_path(text[1], _CHARS)[0]
    # elif decoder == 'word_beam_search':
    #     res = ctc_single_word_beam_search(text[1], _CHARS, 25, prefix_tree)[0]
    # else:
    #     raise Exception('Unknown decoder. Available: "best_path" and "word_beam_search".')

    return text[0]


def transform(img: np.ndarray) -> np.ndarray:
    """Bring image into suitable shape for the model."""
    target_height = 48
    padding = 32

    # compute shape of target image
    fh = target_height / img.shape[0]
    f = min(fh, 2)
    h = target_height
    w = math.ceil(img.shape[1] * f)
    w = w + (4 - w) % 4
    w += padding

    # create target image
    res = 255 * np.ones((h, w), dtype=np.uint8)

    # copy image into target image
    img = cv2.resize(img, dsize=None, fx=f, fy=f)
    th = (res.shape[0] - img.shape[0]) // 2
    tw = (res.shape[1] - img.shape[1]) // 2
    res[th:img.shape[0] + th, tw:img.shape[1] + tw] = img
    res = res

    return res / 255 - 0.5