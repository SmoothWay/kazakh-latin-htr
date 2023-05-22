from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import cv2

from .reader import read
from .word_detector import detect_aabb, detect, sort_multiline, prepare_img, BBox, AABB


@dataclass
class WordReadout:
    text: str
    bbox: BBox
    aabb: AABB


@dataclass
class DetectorConfig:
    height: int = 1000  # input image is resized to this height for detection algo
    kernel_size: int = 25
    sigma: float = 11
    theta: float = 7
    min_area: int = 100
    enlarge: int = 5


@dataclass
class LineClusteringConfig:
    min_words_per_line: int = 1  # minimum number of words per line, if less, line gets discarded
    max_dist: float = 0.7  # threshold for clustering words into lines, value between 0 and 1

# @dataclass
# class ReaderConfig:
#     """Configure how the detected words are read."""
#     decoder: str = 'best_path'  # 'best_path' or 'word_beam_search'
#     prefix_tree: Optional[PrefixTree] = None


def read_page(img: np.ndarray,
              detector_config: DetectorConfig = DetectorConfig(1000),
              line_clustering_config=LineClusteringConfig()) -> List[List[WordReadout]]:
    # prepare image
    img, f = prepare_img(img, detector_config.height)

    # detect words
    detections = detect_aabb(img, detector_config.height, detector_config.enlarge)

    # detections = detect(img,
    #                     detector_config.kernel_size,
    #                     detector_config.sigma,
    #                     detector_config.theta,
    #                     detector_config.min_area)

    # sort words (cluster into lines and ensure reading order top->bottom and left->right)
    lines = sort_multiline(detections, min_words_per_line=line_clustering_config.min_words_per_line)

    # go through all lines and words and read all of them
    read_lines = []
    for line in lines:
        read_lines.append([])
        for word in line:
            text = read(word.img)
            # read_lines[-1].append(WordReadout(text, word.bbox * (1 / f)))
            read_lines[-1].append(WordReadout(text, word.aabb))

    return read_lines