"""
Training code for Adversarial patch training using Faster RCNN based on mmdetection
Redo UPC in PyTorch

"""

import PIL
import cv2
# from load_data import *
import copy
from tqdm import tqdm



import torch
import gc
import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms

import subprocess
# from utils.utils import *
import numpy as np
import pickle

# import patch_config as patch_config
import sys
import time

# from brambox.io.parser.annotation import DarknetParser as anno_darknet_parse
# from utils import *
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
from skimage.segmentation import mark_boundaries
import os
from PIL import Image
import torch.optim as optim
import torch.nn.functional as F
import skimage.io as io
import math
from dataset.coco_train_1000_PERSON import CocoTrainPerson
from torch.utils.data import DataLoader,Dataset

from utils.iou import compute_iou_tensor

import PIL
import load_data
from tqdm import tqdm

from load_data import *
import gc
# import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
from tensorboardX import SummaryWriter
import subprocess
from utils.utils import *

import patch_config as patch_config
import sys
import time
import pickle

# from brambox.io.parser.annotation import DarknetParser as anno_darknet_parse
from utils import *
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
from skimage.segmentation import mark_boundaries
from fgsm import *
from train_patch_measure import measure_region_with_attack
import cv2
from dataset.coco_train_1000_PERSON import CocoTrainPerson
from torch.utils.data import DataLoader, Dataset


class PatchTrainer(object):
    def __init__(self, mode):
        # self.config = patch_config.patch_configs[mode]()

        # self.writer = self.init_tensorboard(mode)
        self.config = patch_config.patch_configs[mode]()

        self.darknet_model = Darknet(self.config.cfgfile)
        self.darknet_model.load_weights(self.config.weightfile)
        self.darknet_model = self.darknet_model.eval().cuda()  # TODO: Why eval?
        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer().cuda()
        self.prob_extractor = MaxProbExtractor(0, 80, self.config).cuda()
        self.top_prob_extractor = TopProbExtractor(0, 80, 'config').cuda()
        # self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        self.total_variation = TotalVariation().cuda()


    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """

        img_size = 800
        batch_size = 1
        n_epochs = 5000
        max_lab = 14

        # ATTACK_TASK = 'target'

        # TARGET_CLASS = 'dog'
        TARGET_CLASS = 16
        ATTACK_TASK = 'untarget'




        # Dataset prepare
        
        # data = CocoTrainPerson(dataType="train2017", num_use=500)
        data = CocoTrainPerson(dataType="train2017", num_use=100)
        dataloader = DataLoader(data, batch_size=1, shuffle=False) #使用DataLoader加载数据

        # attack_area_rate = 0.2
        ATTACK_AREA_RATE = 0.2
        decay_epoch = 100
        k = 0

        # # repeat name judge
        all_name = {}
        # for i_batch, batch_data in tqdm(enumerate(dataloader)):
        #     img, mask, bbox, class_label = batch_data[0][0], batch_data[1][0], batch_data[2][0], batch_data[3][0]
            # img  : 3,500,500
            # mask : 500,500
            # bbox : x1,y1,w,h
            # class_label : tensor[]
            # img_name = batch_data[4][0].split('_')
            # if all_name. __contains__(img_name):
            #     all_name[img_name] = all_name[img_name] + 1
            #     print(img_name)
            # else:
            #     all_name[img_name] = 1





        for i_batch, batch_data in enumerate(dataloader):
            img, mask, bbox, class_label = batch_data[0][0], batch_data[1][0], batch_data[2][0], batch_data[3][0]
            # img  : 3,500,500
            # mask : 500,500
            # bbox : x1,y1,w,h
            # class_label : tensor[]

            mask_area = torch.sum(mask)

            img_name = batch_data[4][0]

            print('---------------')
            print(img_name)
            print('---------------')

            record_dir = '/disk2/mycode/0511models/common_data/scatter_patch/disappear/area'

            success_dir = '/disk2/mycode/0511models/common_data/scatter_patch/disappear/img'

            record_path = os.path.join(record_dir, img_name.split('.')[0]+'.txt')
            # if os.path.exists(record_path):
            #     continue


            # if img_name != '000000131419.jpg':
                
            #     continue
            # else:
            #     if k == 0:
            #         k = k + 1
            #         continue
            

            # img_show = (img + mask.unsqueeze(0).repeat(3,1,1))/2
            # img_show = transforms.ToPILImage()(img_show)
            # img_show.show()

            # get attack region ---- attack_mask
            bbox_x1 = bbox[0]
            bbox_y1 = bbox[1]
            bbox_w = bbox[2]# - bbox_x1
            bbox_h = bbox[3]# - bbox_y1
            bbox_area = torch.sum(mask)
            # attack_area = bbox_area * ATTACK_AREA_RATE * 0.4
            # w_div_h = bbox_w/bbox_h
            # attack_h = torch.sqrt(attack_area/w_div_h)
            # attack_w = attack_h * w_div_h
            #
            # attack_x_c = bbox_x1+bbox_w/2
            # attack_y_c = bbox_y1+bbox_h/2
            #
            #
            # attack_x1 = int(attack_x_c - attack_w/2)
            # attack_x2 = attack_x1 + int(attack_w)
            # attack_y1 = int(attack_y_c - attack_h/2)
            # attack_y2 = attack_y1 + int(attack_h)

            ATTACK_AREA_RATE = 0.0
            for i_enlarge in range(10):
                ATTACK_AREA_RATE = ATTACK_AREA_RATE + 0.05

                # # define 4 pattern
                # # attack 4 square region
                # attack_area = mask_area * ATTACK_AREA_RATE
                # w_div_h = bbox_w / bbox_h
                # # zheng fang xing
                # w_div_h = 1
                #
                # basic_map = torch.zeros(10, 10)
                # basic_map[2, 2] = 1
                # basic_map[2, 7] = 1
                # basic_map[7, 2] = 1
                # basic_map[7, 7] = 1
                # basic_map = basic_map.unsqueeze(0).unsqueeze(0)
                # basic_map = F.interpolate(basic_map, (int(bbox_h), int(bbox_w))).squeeze()
                # # basic_map_pil = transforms.ToPILImage()(basic_map)
                # # basic_map_pil.show()
                # four_square_map = torch.zeros_like(mask)
                # four_square_map[int(bbox_y1):int(bbox_y1) + int(bbox_h),
                # int(bbox_x1):int(bbox_x1) + int(bbox_w)] = basic_map
                #
                # # basic_map_pil = transforms.ToPILImage()(four_square_map)
                # # basic_map_pil.show()
                #
                # four_square_map = four_square_map.cpu()
                #
                # for i in range(20):
                #     four_square_map_np = four_square_map.numpy()
                #     # erode
                #     kernel = np.ones((3, 3), np.uint8)
                #     four_square_map_np = cv2.dilate(four_square_map_np, kernel, iterations=1)
                #     four_square_map_tmp = torch.from_numpy(four_square_map_np)
                #     if torch.sum(four_square_map_tmp) < attack_area:
                #         four_square_map = four_square_map_tmp
                #     else:
                #         break
                #
                # attack_region_four_square = four_square_map.cuda()

                # shan dian
                # init grid

                densy = 5

                unit_w = 13 * densy
                unit_h = 13 * densy
                sandian = torch.zeros(unit_w, unit_h)

                '''
                log:
                10,5,10,5 : 0.04   work! at 700
                10,5,10,6 : 0.0333 work! at 2040


                '''
                # adv_mask_1_layer = adv_mask_1_layer.reshape(100,5,100,5)
                sandian = sandian.reshape(13, densy, 13, densy)
                # adv_mask_1_layer = adv_mask_1_layer.reshape(25,20,25,20)
                # adv_mask_1_layer = adv_mask_1_layer.reshape(20,25,20,25)
                # adv_mask_1_layer = adv_mask_1_layer.reshape(10,50,10,50)

                sandian[:, int((densy - 1) / 2), :, int((densy - 1) / 2)] = 1
                sandian = sandian.reshape(unit_w, unit_h)

                sandian = sandian.unsqueeze(0).unsqueeze(0)
                sandian = F.interpolate(sandian, (500, 500), mode='nearest').squeeze()

                sandian_region = sandian.cpu() * mask.cpu()

                sandian_region = sandian_region.cpu()

                attack_area = mask_area * ATTACK_AREA_RATE
                for i in range(20):
                    sandian_region_np = sandian_region.numpy()
                    # erode
                    kernel = np.ones((3, 3), np.uint8)
                    sandian_region_np = cv2.dilate(sandian_region_np, kernel, iterations=1)
                    sandian_region_tmp = torch.from_numpy(sandian_region_np)
                    if torch.sum(sandian_region_tmp) < attack_area:
                        sandian_region = sandian_region_tmp
                    else:
                        break

                sandian_region = sandian_region.cuda()





                # get adv pattern
                adv_patch = torch.rand_like(img).cuda()
                adv_patch.requires_grad_(True)
                optimizer = optim.Adam([
                    {'params': adv_patch, 'lr': 0.01}
                ], amsgrad=True)

                # hyper parameters
                lambda_rpn = 100.0
                lambda_balance1 = 0.00375 # 0.005
                lambda_balance2 = 0.002
                lambda_tv = 0.0001
                attack_epochs = 300

                # training start
                adv_patch_clamp = adv_patch

                min_max_iou_record = 10
                img = img.cuda()
                mask = mask.cuda()
                attack_mask = sandian_region
                attack_mask = attack_mask.cuda()

                print('use area', torch.sum(attack_mask)/mask_area)


                # refer_img_extend = refer_img_extend.cuda()
                for epoch in range(attack_epochs):
                    patched_img = torch.where(attack_mask > 0, adv_patch_clamp, img)
                    patched_img_255 = patched_img
                    patched_img_unsq = patched_img_255.unsqueeze(0)
                    patched_img_rsz = F.interpolate(patched_img_unsq, (608, 608), mode='bilinear').cuda()

                    # YOLOv4 output
                    output = self.darknet_model(patched_img_rsz)
                    obj_prob = self.prob_extractor(output)
                    top_prob = self.top_prob_extractor(output)[0]
                    top_prob_thr = top_prob[top_prob > 0.3]

                    list_boxes = output

                    nms_thresh = 0.4
                    conf_thresh = 0.5

                    anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
                    num_anchors = 9
                    anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
                    strides = [8, 16, 32]
                    anchor_step = len(anchors) // num_anchors
                    detect_result_list = []
                    for i in range(3):
                        masked_anchors = []
                        for m in anchor_masks[i]:
                            masked_anchors += anchors[m * anchor_step:(m + 1) * anchor_step]
                        masked_anchors = [anchor / strides[i] for anchor in masked_anchors]
                        # decode_result1 = get_region_boxes1(list_boxes[i].detach().cpu().numpy(), conf_thresh, 80, masked_anchors,
                        #                                len(anchor_masks[i]))
                        decode_result = get_region_boxes_tensor(list_boxes[i], conf_thresh, 80,
                                                          masked_anchors,
                                                          len(anchor_masks[i]))
                        # xs, ys, ws, hs, det_confs, cls_conf_logits
                        detect_result_list.append(decode_result)



                    detect_result = torch.cat(detect_result_list, dim=0)
                    proposals = detect_result[:, :5]  # x_c,y_c,w,h, det_confs
                    cls_conf_logits = detect_result[:, 5:]
                    cls_conf = torch.softmax(cls_conf_logits, dim=1)



                    img_cpu = img.cpu()



                    # for i in range(10):
                    #     max_c_index = torch.sort(cls_conf[:, 0] * proposals[:, 4], descending=True)[1][i]
                    #     max_c_bbox = proposals[max_c_index][:4]
                    #     x1 = int((max_c_bbox[0] - max_c_bbox[2] / 2) * 500)
                    #     x2 = int((max_c_bbox[0] + max_c_bbox[2] / 2) * 500)
                    #     y1 = max(int((max_c_bbox[1] - max_c_bbox[3] / 2) * 500),0)
                    #     y2 = int((max_c_bbox[1] + max_c_bbox[3] / 2) * 500)
                    #
                    #     mask_2 = torch.zeros_like((img_cpu))
                    #     mask_2[:, y1:y2, x1:x2] = 1
                    #     img_cpu2 = (img_cpu+mask_2)/2
                    #     img_cpu_pil = transforms.ToPILImage()(img_cpu2)
                    #     img_cpu_pil.show()
                    #     print()




                   # RPN Loss
                    # r1 : from score
                    # r2 : from x,y,w,h

                    # rpn score target is 0
                    rpn_score = proposals[:, 4]
                    loss_r1 = l2_norm(rpn_score - 0)

                    # rpn box target is smaller the boxes
                    rpn_ctx = proposals[:,0].unsqueeze(-1)
                    rpn_cty = proposals[:,1].unsqueeze(-1)
                    rpn_w = proposals[:, 2].unsqueeze(-1)
                    rpn_h = proposals[:, 3].unsqueeze(-1)
                    rpn_box_x1y1x2y2 = [rpn_ctx-rpn_w/2, rpn_cty-rpn_h/2, rpn_ctx+rpn_w/2, rpn_cty+rpn_h/2]
                    rpn_box_x1y1x2y2 = torch.cat(rpn_box_x1y1x2y2, dim=-1)
                    rpn_box_target = torch.cat([rpn_ctx,rpn_cty,rpn_ctx,rpn_cty], dim=-1)
                    loss_r2 = l1_norm(rpn_score.unsqueeze(-1).repeat(1,4)*(rpn_box_x1y1x2y2 - rpn_box_target))

                    test_bbox_reg = False # True
                    if test_bbox_reg:
                        rpn_delta_target = torch.zeros_like(rpn_box_x1y1x2y2).fill_(-0.1)
                        loss_r2_part2 = l1_norm(rpn_score.unsqueeze(-1).repeat(1,4)*(rpn_box_x1y1x2y2 - rpn_delta_target))
                        loss_r2 = (loss_r2 + loss_r2_part2)/2

                    rpn_dimension = proposals.shape[0] / 3
                    rpn_loss = lambda_rpn*(loss_r1 + lambda_balance1 * loss_r2) / rpn_dimension


                    # Regress Loss
                    attack_prob = cls_conf[:, class_label]
                    training_confidence_threshold = 0.38
                    ov_thrs_index = torch.where(attack_prob > training_confidence_threshold) # for certain class
                    ov_thrs_prob = attack_prob[ov_thrs_index]

                    # we can use final_roi here now !!!

                    final_roi = rpn_box_x1y1x2y2[ov_thrs_index]  # for certain class
                    final_roi_target = rpn_box_target[ov_thrs_index]


                    if epoch > decay_epoch:
                        lambda_reg = 10
                        lambda_cls = 100
                    else:
                        lambda_reg = 0
                        lambda_cls = 0

                    reg_loss = lambda_reg * l2_norm(final_roi - final_roi_target)

                    if test_bbox_reg:

                        pbbox = rpn_box_x1y1x2y2[ov_thrs_index]
                        tbbox = torch.zeros_like(pbbox).fill_(-0.1)
                        reg_loss_part2 = lambda_reg * l2_norm(pbbox-tbbox)
                        reg_loss = (reg_loss + reg_loss_part2)/2

                    mean_target_prob = torch.mean(cls_conf[:, class_label])
                    o_score = 0 # ~ logit

                    # Class Loss
                    assert ATTACK_TASK == 'target' or ATTACK_TASK == 'untarget'
                    if ATTACK_TASK == 'target':
                        classification_select = cls_conf[ov_thrs_index]
                        classification_select_log = torch.log(classification_select)
                        target_class = torch.ones(classification_select_log.shape[0]) * TARGET_CLASS
                        target_class = target_class.cuda().long()
                        object_xent = F.nll_loss(classification_select_log, target_class)

                        cls_loss = lambda_cls * (mean_target_prob + lambda_balance2 * object_xent)

                        target_cls_conf = torch.sum(torch.sort(det_labels[:, TARGET_CLASS],descending=True)[0][:10])

                        # cls_loss_new = torch.sum(torch.sort(det_labels[:, class_label],descending=True)[0][:10]) - target_cls_conf

                    elif ATTACK_TASK == 'untarget':
                        classification_select = cls_conf_logits[ov_thrs_index].cuda()
                        target_class = torch.ones(classification_select.shape[0]) * class_label
                        target_class = target_class.cuda().long()
                        object_xent = F.cross_entropy(classification_select, target_class)
                        cls_loss = lambda_cls * (mean_target_prob - lambda_balance2 * object_xent)

                        # cls_loss_new = torch.sum(torch.sort(det_labels[:, class_label],descending=True)[0][:10])


                    total_loss = rpn_loss + cls_loss + reg_loss

                    # if epoch > 100:
                    #     total_loss = cls_loss_new#rpn_loss + reg_loss

                    total_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    #  # adaptive epsilon
                    # target_image_epsilon = 1# 0.2
                    # ratio = epoch / (decay_epoch + 1e-3) * 2
                    # epsilon = target_image_epsilon + (1-target_image_epsilon) * np.exp(-ratio)
                    #
                    # # use epsilon to clamp the adv patch
                    # refer_img_lower_bound = refer_img_extend - epsilon
                    # refer_img_lower_bound = torch.max(refer_img_lower_bound, torch.zeros_like(refer_img_lower_bound))
                    # refer_img_upper_bound = refer_img_extend + epsilon
                    # refer_img_upper_bound = torch.min(refer_img_upper_bound, torch.ones_like(refer_img_upper_bound))
                    #
                    # adv_patch_clamp = torch.max(adv_patch, refer_img_lower_bound)
                    # adv_patch_clamp = torch.min(adv_patch_clamp, refer_img_upper_bound)

                    adv_patch_clamp = torch.clamp(adv_patch, 0, 1)


                    # ----------------------------------
                    # ------------------------
                    # early stop
                    #test
                    # establish gt bbox
                    ground_truth_bbox = [bbox_x1, bbox_y1, bbox_x1 + bbox_w, bbox_y1 + bbox_h]
                    ground_truth_bbox = torch.Tensor(ground_truth_bbox).unsqueeze(0).cuda() / 500
                    patched_img_cpu = patched_img.cpu()
                    test_confidence_threshold = 0.5
                    iou_threshold = 0.5
                    iou = compute_iou_tensor(rpn_box_x1y1x2y2, ground_truth_bbox.repeat(rpn_box_x1y1x2y2.shape[0], 1))
                    attack_prob_select_by_iou_ = (cls_conf[:, 0] * rpn_score)[iou > iou_threshold]
                    attack_prob_select_by_iou_ = attack_prob_select_by_iou_[attack_prob_select_by_iou_>test_confidence_threshold]

                    # stop if no such class found
                    if attack_prob_select_by_iou_.size()[0] == 0:
                        print('Break at', epoch, 'no bbox found')
                        # save image
                        out_file_path = os.path.join(success_dir, img_name)
                        patched_img_cpu_pil = transforms.ToPILImage()(patched_img_cpu)
                        patched_img_cpu_pil.save(out_file_path)

                        txt_save_path = os.path.join(record_dir, img_name.split('.')[0] + '.txt')
                        with open(txt_save_path, 'w') as f:
                            text = str(float(torch.sum(attack_mask)/mask_area))
                            f.write(text)
                        print()
                        break
                if attack_prob_select_by_iou_.size()[0] == 0:
                    break


            print('-------------------------')





            

        print(asdadasdadasdadasdadas)
        # ------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------



    def generate_patch(self, type):
        """
        Generate a random patch as a starting point for optimization.

        :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
        :return:
        """
        if type == 'gray':
            adv_patch_cpu = torch.full((3, 500, 500), 0.5)
        elif type == 'random':
            adv_patch_cpu = torch.rand((3, 500, 500))
        if type == 'trained_patch':
            patchfile = 'patches/object_score.png'
            patch_img = Image.open(patchfile).convert('RGB')
            patch_size = self.config.patch_size
            tf = transforms.Resize((patch_size, patch_size))
            patch_img = tf(patch_img)
            tf = transforms.ToTensor()
            adv_patch_cpu = tf(patch_img)

        return adv_patch_cpu

    def read_image(self, path):
        """
        Read an input image to be used as a patch

        :param path: Path to the image to be read.
        :return: Returns the transformed patch as a pytorch Tensor.
        """
        patch_img = Image.open(path).convert('RGB')
        tf = transforms.Resize((self.config.patch_size, self.config.patch_size))
        patch_img = tf(patch_img)
        tf = transforms.ToTensor()

        adv_patch_cpu = tf(patch_img)
        return adv_patch_cpu





def connected_domin_detect(input_img):
    from skimage import measure
    # detection
    if input_img.shape[0] == 3:
        input_img_new = (input_img[0] + input_img[1] + input_img[2])
    else:
        input_img_new = input_img
    ones = torch.Tensor(input_img_new.size()).fill_(1)
    zeros = torch.Tensor(input_img_new.size()).fill_(0)
    input_map_new = torch.where((input_img_new != 0), ones, zeros)
    # img = transforms.ToPILImage()(input_map_new.detach().cpu())
    # img.show()
    input_map_new = input_map_new.cpu()
    labels = measure.label(input_map_new[:, :], background=0, connectivity=2)
    label_max_number = np.max(labels)
    return float(label_max_number)


def get_obj_min_score(boxes):
    if type(boxes[0][0]) is list:
        min_score_list = []
        for i in range(len(boxes)):
            score_list = []
            for j in range(len(boxes[i])):
                score_list.append(boxes[i][j][4])
            min_score_list.append(min(score_list))
        return np.array(min_score_list)
    else:
        score_list = []
        for j in range(len(boxes)):
            score_list.append(boxes[j][4])
        return np.array(min(score_list))

def l2_norm(tensor):
    return torch.sqrt(torch.sum(torch.pow(tensor,2)))

def l1_norm(tensor):
    return torch.sum(torch.abs(tensor))


def main():


    trainer = PatchTrainer('paper_obj')
    trainer.train()


if __name__ == '__main__':
    main()


