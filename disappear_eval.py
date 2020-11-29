import os
import numpy as np

area_dir = '../common_data/dap/disappear/area'
# area_dir = '../common_data/scatter_patch/disappear/area'
area_list = os.listdir(area_dir)

fail = 0
area_fail = 0
area_sum = 0
area_val_list = []
for area_file in area_list:
    record_path = os.path.join(area_dir, area_file)
    with open(record_path,"r") as f:
        area = f.read()
        area = float(area)
        area_sum += area
        area_val_list.append(area)
        
print(area_sum/len(area_list))
print(max(area_val_list))
# print(fail)



 
# success_all_list = []
# for area_file in area_list:
#     record_path = os.path.join(area_dir, area_file)
#     with open(record_path,"r") as f:
#         min_max_area_old = f.read()
#         min_max_area_old = float(min_max_area_old)
#         if min_max_area_old<0.5:
#             success_all_list.append(area_file.split('.')[0])

# for cls_fail in success_list:
#     success_all_list.append(cls_fail.split('.')[0])

# success_all_np = np.array(success_all_list)
# success_all_np = np.unique(success_all_np)
# print(success_all_np.shape, len(area_list))
