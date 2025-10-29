import test_folder.aruco_board as ar_b
import test_folder.segmentacion as seg

#Con pose de cámara:
ar_b.process_img_with_pose_and_contour("./images/charuco test 3.jpg", "./results/charuco/charuco_calibration.npz", True, "mask", white_background=False)

# sin pose de cámara:
seg.segment_with_aruco_and_mask("./images/aruco test 6.jpg", "mask", white_background=False, aruco_side_cm=4.9, show_img=True)