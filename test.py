import os
import cv2
import numpy as np
import segment_funcs.img_funcs as img_f
import aruco.aruco_funcs as ar_f
import aruco.aruco_board as ar_b
import segment_funcs.segmentacion as seg
import models.yolo_funcs as yolo_f
import models.mask_r_cnn_funcs as mask_f

ar_b.process_img_with_pose("./images/aruco test 6.jpg", "./results/charuco/charuco_calibration.npz", False)

