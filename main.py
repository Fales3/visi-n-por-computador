import segment_funcs.segmentacion as seg
import aruco.aruco_board as ar_b


seg.segment_with_aruco_and_yolo("./images/aruco 5cm test2.jpg", white_background=False, aruco_side_cm=4.9, show_img=True)
# ar_b.process_img_with_pose("./images/charuco test 5.jpg", "./results/charuco/charuco_calibration.npz", True, 0.049)
