#!/usr/bin/env python3
import cv2
from Processor import Processor

class generate:
    def __init__(self, file_path=None, data_dir=None, num_samples=1e2):
        if file_path is not None:
            cap = cv2.VideoCapture(file_path)
            Processor(cap=cap)
