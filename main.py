import segment_funcs.segmentacion as seg

event={
    "img_name":"aruco test 4.jpg",
    "aruco": True,
    "blurring_method":"median",
    "threshold_method": "OTSU",
    "show":True,
    "defect": "picadura"
}

seg.segment_with_aruco_and_yolo("./images/aruco test 6.jpg", white_background=False, aruco_side_cm=4.9, show_img=True)