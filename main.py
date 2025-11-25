import segment_funcs.segmentacion as seg
import aruco.aruco_board as ar_b
import aruco.aruco_funcs as ar_f

# seg.segment_with_aruco_and_yolo("./images/aruco 5cm test2.jpg", white_background=False, aruco_side_cm=4.9, show_img=True)
# ar_b.process_img_with_pose("./images/charuco test 6.jpg", "./results/charuco/charuco_calibration.npz", True, 0.049)
ar_f.generate_aruco(aruco_id=1)