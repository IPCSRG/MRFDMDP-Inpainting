import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
 
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
from PIL import Image
import glob
import torchvision.transforms.functional as F
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

torch.autograd.set_detect_anomaly(True)

def load_flist(flist):
    # np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
    if isinstance(flist, list):
        return flist
    # flist: image file path, image directory path, text file flist path
    if isinstance(flist, str):
        if os.path.isdir(flist):
            flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
            flist.sort()
            return flist

        if os.path.isfile(flist):
            try:
                return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
            except:
                return [flist]
    return []

def calculate_psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))

def postprocess(img):
    img = (img + 1) / 2 * 255
    img = img.permute(0, 2, 3, 1)
    img = img.int().cpu().numpy().astype(np.uint8)
    return img

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    # opt.save_latest_freq = 12971
    opt.checkpoints_dir = './checkpoints'
    # opt.save_latest_freq = 28000
    # opt.save_latest_freq = 40000
    # opt.train_imgroot = 'D:/Documents/Dataset/CelebA-HQ-img-28000_256'
    # opt.train_maskroot = 'D:/Documents/Dataset/train_mask_28000_256'
    opt.save_latest_freq = 6
    opt.train_imgroot = 'D:/Documents/Dataset/111/1111'
    opt.train_maskroot = 'D:/Documents/Dataset/111/56'
    val_image = '/root/code/mymethod/datasets/celea-hq/celeba-hq-256-test-2000-fillname'
    val_pre = '/root/code/mymethod/datasets/testing_mask_256_cvresize/'
    
    # opt.train_imgroot = '/root/code/mymethod/datasets/places2/places2_train_image_40000-256'
    # opt.train_maskroot = '/root/code/mymethod/datasets/places2/places2_train_mask_40000-256'
    # val_image = '/root/code/mymethod/datasets/places2/places2_test_image_2000-256'
    # val_pre = '/root/code/mymethod/datasets/testing_mask_256_cvresize/'
    
    
    # opt.train_imgroot = 'D:/Documents/Dataset/Piarstreet/paris_train_original-256'
    # opt.train_maskroot = 'D:/Documents/Dataset/Piarstreet/paris_train_mask-256'
    # val_image = '/root/code/mymethod/datasets/Piarstreet/paris_eval_gt-256'
    # val_pre = '/root/code/mymethod/datasets/Piarstreet/paris_test_irrmask-256_cvresize/'
    
  
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
   
    total_iters = 0                # the total number of training iterations

    model.train()
    # opt.continue_train = True
    if opt.continue_train:
        # opt.epoch_count=95
        print(opt.epoch_count)
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>

       

        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing

            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            # model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

           

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, epoch_iter, t_comp, t_data)
                for k, v in losses.items():
                    message += '%s: %.3f ' % (k, v)
                print(message)  # print the message 

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.


#         if epoch%1 == 0:
#             model.eval()
           
#             val_mask_suffix = ['mask0-10', 'mask10-20', 'mask20-30', 'mask30-40', 'mask40-50', 'mask50-60']
#             save_dir_suffix = ['010', '1020', '2030', '3040', '4050', '5060']
#             mean_psnr = 0
#             for suffix_idx in range(6):
#                 val_mask = val_pre + val_mask_suffix[suffix_idx]
#                 test_image_flist = load_flist(val_image)
#                 print(len(test_image_flist))
#                 test_mask_flist = load_flist(val_mask)
#                 print(len(test_mask_flist))
#                 psnr_lg = []
#                 psnr_sk = []
#                 mask_num = len(test_mask_flist)
#                 # iteration through datasets
#                 for idx in range(len(test_image_flist)):
#                     img = Image.open(test_image_flist[idx]).convert('RGB')
#                     # mask = Image.open(test_mask_flist[idx%mask_num]).convert('L')
#                     mask = Image.open(test_mask_flist[idx]).convert('L')
#                     masks = F.to_tensor(mask)
#                     images = F.to_tensor(img) * 2 - 1.
#                     images = images.unsqueeze(0)
#                     masks = masks.unsqueeze(0)
#                     data = {'A': images, 'B': masks, 'A_paths': ''}

#                     with torch.no_grad():
#                         model.set_input(data)
#                         model.forward()

#                     orig_imgs = postprocess(model.images)
#                     mask_imgs = postprocess(model.masked_images1)
#                     comp_imgs = postprocess(model.merged_images1)
#                     comp_imgs1 = postprocess(model.merged_images1)

#                     psnr_tmp = calculate_psnr(orig_imgs, comp_imgs)
#                     psnr_lg.append(psnr_tmp)
#                     psnr_score = psnr(orig_imgs, comp_imgs, data_range=255)
#                     psnr_sk.append(psnr_score)
#                 print('===============val_epoch: ' + str(opt.epoch)+'===================')
#                 print('The avg psnr is', np.mean(np.array(psnr_lg)))
#                 print('The avg psnr_sk is', np.mean(np.array(psnr_sk)))

    os.system("shutdown")