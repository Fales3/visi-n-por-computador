import segment_funcs.segmentacion as seg

event={
    "img_name":"aruco test.jpg",
    "aruco": True,
    "blurring_method":"median",
    "threshold_method": "OTSU",
    "show":True
}

seg.classic_segment_img("aruco test.jpg",True, "median", "adaptative", True)