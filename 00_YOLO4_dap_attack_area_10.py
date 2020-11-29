"""
Training code for Adversarial patch training using Faster RCNN based on mmdetection
Redo UPC in PyTorch

"""

import PIL
import cv2
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import argparse

"""hyper parameters"""
use_cuda = True


from tqdm import tqdm
import torch.nn.functional as F


from extend.coco_train_1000_PERSON import CocoTrainPerson
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from skimage.segmentation import slic
import PIL.Image as Image
from get_convex_env import get_conv_envl
from extend.iou import compute_iou_tensor

torch.cuda.set_device(2)

class PatchTrainer(object):
    def __init__(self, mode):
        
        cfgfile = './cfg/yolov4.cfg'
        weightfile = '../common_data/yolov4.weights'

        self.darknet_model = Darknet(cfgfile).cuda()
        self.darknet_model.print_network()
        self.darknet_model.load_weights(weightfile)
        self.darknet_model.eval()
        print('Loading weights from %s... Done!' % (weightfile))


    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """

        img_size = 800
        batch_size = 1
        n_epochs = 5000
        max_lab = 14




        # Dataset prepare

        data = CocoTrainPerson(dataType="train2017", num_use=100)
        dataloader = DataLoader(data, batch_size=1, shuffle=False) #使用DataLoader加载数据


        ATTACK_AREA_RATE = 0.1
        decay_epoch = 100
        k = 0



        for i_batch, batch_data in enumerate(dataloader):
            img, mask, bbox, class_label = batch_data[0][0], batch_data[1][0], batch_data[2][0], batch_data[3][0]



            ##############################################################################

            img_name = batch_data[4][0]
            mask_area = torch.sum(mask)

            print('---------------')
            print(img_name)
            print('---------------')

            record_dir = '../common_data/yolo4_dap_attack/disappear/area'


            record_path = os.path.join(record_dir, img_name.split('.')[0]+'.txt')


            
            # use segment SLIC
            base_SLIC_seed_num = 3000
            img_np = img.numpy().transpose(1,2,0)
            mask_np = mask.numpy()
            numSegments = int(base_SLIC_seed_num/(500*500)*torch.sum(mask))
            segments_np = slic(image=img_np, n_segments=numSegments, sigma=0, slic_zero=True, mask=mask_np)
            segments_tensor = torch.from_numpy(segments_np).float().cuda()
            segments_label = torch.unique(segments_tensor)
            segments_label = segments_label[1:]




            zero_layer = torch.zeros_like(segments_tensor)
            one_layer = torch.ones_like(segments_tensor)


            bbox_x1 = bbox[0]
            bbox_y1 = bbox[1]
            bbox_w = bbox[2]
            bbox_h = bbox[3]

            bbox_x_c = bbox_x1 + bbox_w/2
            bbox_y_c = bbox_y1 + bbox_h/2
            bbox_x_c_int = int(bbox_x_c)
            bbox_y_c_int = int(bbox_y_c)


            # 3 load attack region 
            load_patch_dir = '../common_data/NES_search_test_1107/'+img_name.split('_')[0]

            load_patch_list = os.listdir(load_patch_dir)
            load_patch_list.sort()
            wat_num_max = 0
            for i_name in load_patch_list:
                wat_num = int(i_name.split('_')[0])
                if wat_num > wat_num_max:
                    wat_num_max = wat_num
            for i_name in load_patch_list:
                wat_num = int(i_name.split('_')[0])
                if wat_num == wat_num_max:
                    max_name = i_name
                    break

            load_patch = os.path.join(load_patch_dir, max_name)

            load_img = Image.open(load_patch).convert('RGB')
            load_img = transforms.ToTensor()(load_img)
            region_mask = 2*load_img - img.cpu()
            region_mask = torch.sum(region_mask,dim=0)/3
            region_mask = torch.where(mask>0, region_mask,torch.zeros_like(region_mask))


            attack_region_tmp_pil = transforms.ToPILImage()(region_mask.cpu())
            attack_region_tmp_pil.save('013k.png')
            # process mask
            region_mask_new = torch.zeros_like(region_mask).cuda()
            for i in range(segments_label.shape[0]):
                sp =  segments_label[i]
                right_color = (torch.where(segments_tensor==sp,region_mask.cuda(),one_layer*(-10))).cpu()
                right_color = torch.mean(right_color[right_color!=-10])
                color_layer = torch.ones_like(segments_tensor).fill_(right_color)
                region_mask_new = torch.where(segments_tensor==sp, color_layer, region_mask_new)      
            region_mask_new = region_mask_new
            region_mask = region_mask_new
            region_mask_unique = torch.unique(region_mask)




            for i in range(region_mask_unique.shape[0]):
                thres = region_mask_unique[i]
                # region_mask_tmp = torch.zeros_like(region_mask)
                region_mask_tmp = torch.where(region_mask>thres, one_layer, zero_layer)
                pixel_num = torch.sum(region_mask_tmp)
                if pixel_num < mask_area * ATTACK_AREA_RATE:
                    break

            attack_region_search_top = region_mask_tmp
            attack_region_search_top = get_conv_envl(attack_region_search_top)

            attack_region_tmp_pil = transforms.ToPILImage()(attack_region_search_top.cpu())
            
            attack_region_tmp_pil.save('012k.png')


            attack_region_tmp = attack_region_search_top

            attack_region_tmp = attack_region_tmp.cuda()
            now_area =  float(torch.sum(attack_region_tmp)/mask_area)
            print('---------------')
            print('You have used ', now_area, 'area.')
            print('---------------')
            ## start at gray
            adv_patch_w = torch.zeros(3,500,500).cuda()

            adv_patch_w.requires_grad_(True)

            optimizer = torch.optim.Adam([
                {'params': adv_patch_w, 'lr': 0.1}
            ], amsgrad=True)

            t_op_num = 1500
            min_max_iou_record = 1
            for t_op_step in range(t_op_num):
                adv_patch = torch.sigmoid(adv_patch_w)
                patched_img = torch.where(attack_region_tmp>0, adv_patch, img.cuda()).unsqueeze(0)

                patched_img_rsz = F.interpolate(patched_img, (608, 608), mode='bilinear').cuda()


                yolov4_output = self.darknet_model(patched_img_rsz)

                bbox_pred, cls_pred, obj_pred = yolov4_output

                bbox_pred = bbox_pred.squeeze()

                total_loss = torch.max(obj_pred)

                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                person_conf = (cls_pred * obj_pred)[0,:,0]

                ground_truth_bbox = [bbox_x1, bbox_y1, bbox_x1 + bbox_w, bbox_y1 + bbox_h]
                ground_truth_bbox = torch.Tensor(ground_truth_bbox).unsqueeze(0).cuda() / 500

                ground_truth_bboxs = ground_truth_bbox.repeat(bbox_pred.shape[0], 1)

                iou = compute_iou_tensor(bbox_pred, ground_truth_bboxs)

                 # ----------------------------------
                # ------------------------
                # early stop

                #test
                patched_img_cpu = patched_img.cpu().squeeze()
                test_confidence_threshold = 0.45


                
                ov_test_thrs_index = torch.where(attack_prob > test_confidence_threshold)[0]

                final_pbbox = det_bboxes[:, class_label*4:(class_label+1)*4]
                ground_truth_bboxs_final = ground_truth_bbox.repeat(final_pbbox.shape[0],1)
                iou = compute_iou_tensor(final_pbbox, ground_truth_bboxs_final)
                attack_prob_select_by_iou_ = attack_prob[iou>0.05]
                attack_prob_select_by_iou_ = attack_prob_select_by_iou_[attack_prob_select_by_iou_>test_confidence_threshold]


                # stop if no such class found
                if attack_prob_select_by_iou_.shape[0] == 0:
                    print('Break at',t_op_step,'no bbox found')
                    # save image
                    patched_img_cpu_pil = transforms.ToPILImage()(patched_img_cpu)
                    out_file_path = os.path.join('../common_data/yolo4_dap_attack/success'+str(int(ATTACK_AREA_RATE*100)), img_name)
                    patched_img_cpu_pil.save(out_file_path)
                    break


                # max same-class object bounding box iou s
                final_pbbox = det_bboxes[ov_test_thrs_index][:, class_label*4:(class_label+1)*4]
                ground_truth_bboxs_final = ground_truth_bbox.repeat(final_pbbox.shape[0],1)
                iou = compute_iou_tensor(final_pbbox, ground_truth_bboxs_final)
                iou_max = torch.max(iou)
                if iou_max < 0.05:
                    print('Break at',t_op_step,'iou final max:', torch.max(iou))
                    # save image
                    patched_img_cpu_pil = transforms.ToPILImage()(patched_img_cpu)
                    out_file_path = os.path.join('../common_data/yolo4_dap_attack/success'+str(int(ATTACK_AREA_RATE*100)), img_name)
                    patched_img_cpu_pil.save(out_file_path)

                    break

                # report 
                ground_truth_bboxs = ground_truth_bbox.repeat(1000,1)
                final_pbbox = det_bboxes[ov_test_thrs_index][:, class_label*4:(class_label+1)*4]
                ground_truth_bboxs_final = ground_truth_bbox.repeat(final_pbbox.shape[0],1)
                iou = compute_iou_tensor(final_pbbox, ground_truth_bboxs_final)
                
                max_iou = torch.max(iou)
                if max_iou < min_max_iou_record:
                    min_max_iou_record = max_iou
                    txt_save_dir =  '../common_data/yolo4_dap_attack/iou'+str(int(ATTACK_AREA_RATE*100))
                    txt_save_path = os.path.join(txt_save_dir, img_name.split('.')[0]+'.txt')
                    with open(txt_save_path,'w') as f:
                        text = str(float(max_iou))
                        f.write(text)

                if t_op_step % 100 == 0:

                    iou_sort = torch.sort(iou,descending=True)[0][:6].detach().clone().cpu()

                    print(t_op_step, 'iou t-cls  :', iou_sort)

                    # iou over 0.5, confidence print
                    final_pbbox = det_bboxes[:, class_label*4:(class_label+1)*4]
                    iou = compute_iou_tensor(final_pbbox, ground_truth_bbox.repeat(final_pbbox.shape[0],1))
                    attack_prob = det_labels[:,class_label]
                    attack_prob_select_by_iou_ = attack_prob[iou>0.05]

                    attack_prob_select_by_iou_sort = torch.sort(attack_prob_select_by_iou_,descending=True)[0][:6].detach().cpu()
                    print(t_op_step, 'right cls cf:', attack_prob_select_by_iou_sort)


                    print()

                    

     



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


