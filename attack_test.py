from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import argparse

"""hyper parameters"""
use_cuda = True


import cv2

cfgfile = './cfg/yolov4.cfg'
weightfile = '../common_data/yolov4.weights'

m = Darknet(cfgfile)

m.print_network()
m.load_weights(weightfile)
print('Loading weights from %s... Done!' % (weightfile))

if use_cuda:
    m.cuda()

num_classes = m.num_classes
if num_classes == 20:
    namesfile = 'data/voc.names'
elif num_classes == 80:
    namesfile = 'data/coco.names'
else:
    namesfile = 'data/x.names'
class_names = load_class_names(namesfile)


imgfile = '/disk2/mycode/common_data/zzj_cloth/img/1.png'
img = cv2.imread(imgfile)
sized = cv2.resize(img, (m.width, m.height))
sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

# m.train()

# boxes = m(sized)

for i in range(2):
    start = time.time()
    boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
    finish = time.time()
    if i == 1:
        print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

plot_boxes_cv2(img, boxes[0], savename='predictions.jpg', class_names=class_names)