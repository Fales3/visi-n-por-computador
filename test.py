import os
import segment_funcs.img_funcs as img_f
import segment_funcs.segmentacion as seg
import models.yolo_funcs as yolo_f
import models.mask_r_cnn_funcs as mask_f

model=yolo_f.load_yolo_model()
yolo_f.load_img_to_model("aruco test 3.jpg",model)

# model=mask_f.load_mask_r_cnn_model()
# mask_f.extract_mask(model,"aruco test 3.jpg")

# in_dir="./dataset/test/Potato White 1"
# out_dir="./dataset/masks/val"

# seg.save_masks(in_dir, out_dir)

# path_name="./dataset/"
# yolo_f.get_yolo_format(path_name)