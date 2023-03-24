import os
import torch
from torch.autograd import Variable
import argparse
import time
import datetime
import sys
import math
from skimage import io
from model_3DUnet import Network_3D_Unet
import numpy as np
from data_process import shuffle_datasets_lessMemory, test_preprocess_lessMemoryNoTail_cell256_S2, \
    test_preprocess_lessMemoryNoTail_cell256, Cell_namelst,\
    train_preprocess_lessMemoryMulStacks_Cellim_S1

import glob
from model_3DUnet import RCAN
root = '/mnt/home/user1/'
device = 'cuda:0'

#############################################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--datasets_folder', type=str, default='cell/',  # '20120502_cell1/',
                    help="A folder containing files to be tested")

parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--GPU', type=int, default=0, help="the index of GPU you will use for computation")
parser.add_argument('--batch_size', type=int, default=1, help="batch size")

# # NB2NB
parser.add_argument('--denoise_model', type=str, default='S1_3D')
parser.add_argument('--img_w', type=int, default=256, help="the width of image sequence")
parser.add_argument('--img_h', type=int, default=256, help="the height of image sequence")
parser.add_argument('--img_s', type=int, default=20, help="the slices of image sequence")
parser.add_argument('--gap_w', type=int, default=90, help='the width of image gap')
parser.add_argument('--gap_h', type=int, default=90, help='the height of image gap')
parser.add_argument('--gap_s', type=int, default=10, help='the slices of image gap')


# parser.add_argument('--denoise_model', type=str, default='S2_3D')
# parser.add_argument('--img_w', type=int, default=256, help="the width of image sequence")
# parser.add_argument('--img_h', type=int, default=256, help="the height of image sequence")
# parser.add_argument('--img_s', type=int, default=11, help="train slices = 42")
# parser.add_argument('--gap_w', type=int, default=90, help='the width of image gap')
# parser.add_argument('--gap_h', type=int, default=90, help='the height of image gap')
# parser.add_argument('--gap_s', type=int, default=20, help='the slices of image gap')

parser.add_argument('--in_channels', type=int, default=1, help='1')
parser.add_argument('--out_channels', type=int, default=1, help='1')

parser.add_argument('--test_datasizeh', type=int, default=256, help='512 or -1')
parser.add_argument('--datasets_path', type=str, default=root[:-len('DeepCAD_pytorch/')] + 'dataset/',
                        help=" '/datasets' ")
parser.add_argument("--n_epochs", type=int, default=100, help="number of training epochs")
parser.add_argument('--train_datasets_size', type=int, default=-1, help='datasets size for training')
parser.add_argument('--test_datasize', type=int, default=150, help='-1 dataset size to be tested')

parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument("--b1", type=float, default=0.5, help="Adam: bata1")
parser.add_argument("--b2", type=float, default=0.999, help="Adam: bata2")
parser.add_argument('--normalize_factor', type=int, default=65535, help='normalize factor')
parser.add_argument('--output_dir', type=str, default=root+'/results', help="output directory")
parser.add_argument('--pth_path', type=str, default=root+'/pth', help="pth file root path")


opt = parser.parse_args()

############################----------------  start  ------------###############################


# # 2/3D [150,150,150]->[1,150,150]
def generateTrainDataS2(noise_img, init_s, end_s, init_h, end_h, init_w, end_w, threeD=False):
    patch = noise_img[init_s:end_s, init_h:end_h, init_w:end_w]  # 150x150x150
    t, h, w = patch.shape  # [42, 256, 256]
    noise_patch1 = np.zeros([t-2, h, w], dtype=np.float32)
    # 中心帧
    noise_patch2 = patch[t // 2:t // 2 + 1, init_h:end_h, init_w:end_w]
    ## 中心帧移除
    noise_patch1[0:t // 2-1, :, :] = patch[1:t // 2, :, :]
    noise_patch1[t // 2:, :, :] = patch[t // 2 + 1, :, :]
    # print('noise_patch1[t//2-1,:,:]', noise_patch1[t//2-1,:,:])
    
    if threeD:
        real_A = torch.from_numpy(np.expand_dims(np.expand_dims(noise_patch1, 0), 0)).to(device)  # 1x1x40x256x256
        real_B = torch.from_numpy(np.expand_dims(np.expand_dims(noise_patch2, 0), 0)).to(device)  # 1x1x1x256x256
    else:
        real_A = torch.from_numpy(np.expand_dims(noise_patch1, 0)).to(device)  # 1x40x256x256
        real_B = torch.from_numpy(np.expand_dims(noise_patch2, 0)).to(device)  # 1x1x256x256
    
    return real_A, real_B


############################----------------  start of NB2NB  ------------###############################
operation_seed_counter = 0


# 2D [40,256,256]->[40,1,256,256]
def generateTrainDataS1(noise_img, init_s, end_s, init_h, end_h, init_w, end_w):
    def generate_mask_pair_torch(img):
        # prepare masks (N x C x H/2 x W/2)
        def get_generator():
            global operation_seed_counter
            operation_seed_counter += 1
            g_cuda_generator = torch.Generator(device=device)
            g_cuda_generator.manual_seed(operation_seed_counter)
            return g_cuda_generator
        
        n, c, h, w = img.shape  # [2,4,64,64]
        mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4,),
                            dtype=torch.bool,
                            device=img.device)
        mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4,),
                            dtype=torch.bool,
                            device=img.device)
        # prepare random mask pairs
        idx_pair = torch.tensor(
            [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
            dtype=torch.int64,
            device=img.device)  # [8, 2]
        rd_idx = torch.zeros(size=(n * h // 2 * w // 2,),
                             dtype=torch.int64,
                             device=img.device)  # [2048=h*w*batch/4]
        torch.randint(low=0,
                      high=8,
                      size=(n * h // 2 * w // 2,),
                      generator=get_generator(),
                      out=rd_idx)
        rd_pair_idx = idx_pair[rd_idx]  # [2048,2]
        rd_pair_idx += torch.arange(start=0, end=n * h // 2 * w // 2 * 4,
                                    step=4, dtype=torch.int64, device=img.device).reshape(-1, 1)
        # get masks
        mask1[rd_pair_idx[:, 0]] = 1
        mask2[rd_pair_idx[:, 1]] = 1
        return mask1, mask2
    
    def generate_subimages_torch(img, mask):
        def space_to_depth(x, block_size):
            n, c, h, w = x.size()  # 2,1,64,64
            unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)  # 2,4,1024
            return unfolded_x.view(n, c * block_size ** 2, h // block_size, w // block_size)  # 2,4,32,32
        
        n, c, h, w = img.shape
        subimage = torch.zeros(n, c, h // 2, w // 2, dtype=img.dtype, layout=img.layout, device=img.device)
        # per channel
        for i in range(c):
            img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)  # n, 4, h, w
            img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)  # n, h, w, 4
            subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(n, h // 2, w // 2, 1).permute(0, 3, 1,
                                                                                                     2)  # n, 1, h/2, w/2
        return subimage
    
    noise = noise_img[init_s:end_s, init_h:end_h, init_w:end_w]
    noise = torch.from_numpy(np.expand_dims(noise, 0)).to(device)  # 1x40x256x256
    
    mask1, mask2 = generate_mask_pair_torch(noise)
    real_A = generate_subimages_torch(noise, mask1)
    real_B = generate_subimages_torch(noise, mask2)

    real_A = real_A.permute([1, 0, 2, 3])  # 40x1x256x256
    real_B = real_B.permute([1, 0, 2, 3])  # 40x1x256x256
    
    return real_A, real_B  # 40x1x128x128


##  only squeeze 0/last axis for 3D image  [40,1,1,256,256] or [1,1,40,256,256]
def generateTrainDataS1_3D(noise_img, init_s, end_s, init_h, end_h, init_w, end_w):
    def generate_mask_pair_torch(img):
        # prepare masks (N x C x H/2 x W/2)
        def get_generator():
            global operation_seed_counter
            operation_seed_counter += 1
            g_cuda_generator = torch.Generator(device=device)
            g_cuda_generator.manual_seed(operation_seed_counter)
            return g_cuda_generator
        
        n, c, h, w = img.shape  # [2,4,64,64]
        mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4,),
                            dtype=torch.bool,
                            device=img.device)
        mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4,),
                            dtype=torch.bool,
                            device=img.device)
        # prepare random mask pairs
        idx_pair = torch.tensor(
            [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
            dtype=torch.int64,
            device=img.device)  # [8, 2]
        rd_idx = torch.zeros(size=(n * h // 2 * w // 2,),
                             dtype=torch.int64,
                             device=img.device)  # [2048=h*w*batch/4]
        torch.randint(low=0,
                      high=8,
                      size=(n * h // 2 * w // 2,),
                      generator=get_generator(),
                      out=rd_idx)
        rd_pair_idx = idx_pair[rd_idx]  # [2048,2]
        rd_pair_idx += torch.arange(start=0, end=n * h // 2 * w // 2 * 4,
                                    step=4, dtype=torch.int64, device=img.device).reshape(-1, 1)
        # get masks
        mask1[rd_pair_idx[:, 0]] = 1
        mask2[rd_pair_idx[:, 1]] = 1
        return mask1, mask2
    
    def generate_subimages_torch(img, mask):
        def space_to_depth(x, block_size):
            n, c, h, w = x.size()  # 2,1,64,64
            unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)  # 2,4,1024
            return unfolded_x.view(n, c * block_size ** 2, h // block_size, w // block_size)  # 2,4,32,32
        
        n, c, h, w = img.shape
        subimage = torch.zeros(n, c, h // 2, w // 2, dtype=img.dtype, layout=img.layout, device=img.device)
        # per channel
        for i in range(c):
            img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)  # n, 4, h, w
            img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)  # n, h, w, 4
            subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(n, h // 2, w // 2, 1).permute(0, 3, 1,
                                                                                                     2)  # n, 1, h/2, w/2
        return subimage
    
    noise = noise_img[init_s:end_s, init_h:end_h, init_w:end_w]
    noise = torch.from_numpy(np.expand_dims(noise, 0)).to(device)
    
    mask1, mask2 = generate_mask_pair_torch(noise)
    real_A = generate_subimages_torch(noise, mask1)
    real_B = generate_subimages_torch(noise, mask2)  # 1x40x256x256

    real_A = real_A.permute([1, 0, 2, 3])  # 40x1x128x128
    real_B = real_B.permute([1, 0, 2, 3])
    real_A = real_A.unsqueeze(0)
    real_B = real_B.unsqueeze(0)  # 1x40x1x128x128
    real_A = real_A.permute([1, 0, 2, 3, 4])  # 40x1x1x128x128
    real_B = real_B.permute([1, 0, 2, 3, 4])
    
    return real_A, real_B


############################----------  end pf NB2NB  ------------#################################
def train():
    saveinput = False  # True  #
    scale_factor = 4
    output_path = pth_path + '/valid/'
    os.makedirs(output_path, exist_ok=True)
    
    ########################################################################################################################
    L1_pixelwise = torch.nn.L1Loss()
    L2_pixelwise = torch.nn.MSELoss()
    if three_Dmodel:
        if opt.out_channels == 1:
            denoise_generator = Network_3D_Unet(in_channels=1, out_channels=1, channelstride=1,
                                    final_sigmoid=True, layer_order='crg', pool_kernel_size=(1, 2, 2))
        else:
            denoise_generator = Network_3D_Unet(in_channels=1, out_channels=1, channelstride=1,
                                        final_sigmoid=True, layer_order='crg')
    else:
        if 'RCAN' in opt.denoise_model:
            denoise_generator = RCAN(in_channels=opt.in_channels, out_channels=opt.out_channels)
        else:
            denoise_generator = Network_3D_Unet(UNet_type='2DUNet', in_channels=opt.in_channels,  # opt.img_s * 2,
                                                out_channels=opt.out_channels, final_sigmoid=True, layer_order='trg')

    resume = False
    if resume:
        modellst = glob.glob(pth_path + '/G_*.pth')
        if len(modellst) > 0:
            denoise_generator.load_state_dict(torch.load(modellst[-1], map_location=torch.device(device)))
            print('Resume Model from', modellst[-1])
            opt.epoch = int(modellst[-1][len(pth_path) + 3:-4])
    else:
        print('train from epoch 0')

    if torch.cuda.is_available():
        print('Using GPU.')
        denoise_generator.to(device)
        L2_pixelwise.to(device)
        L1_pixelwise.to(device)
        
    nameall_list, subdirlst = Cell_namelst(opt)
    for epoch in range(opt.epoch, opt.n_epochs + 1):
        for di in range(len(subdirlst)):
            dir = subdirlst[di]
            name_list = nameall_list[di]
            for ni in range(len(name_list)):
                name = name_list[ni]
                patchname_list, noise_img, coordinate_list = \
                        train_preprocess_lessMemoryMulStacks_Cellim_S1(opt, name, dir, NB='NB2NB' in opt.denoise_model)
                print('len(name_list), len(patchname_list) -----> ', len(name_list), len(patchname_list))  # 1,118

                optimizer_G = torch.optim.Adam(denoise_generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
                ########################################################################################################################
                prev_time = time.time()
                time_start = time.time()
                patchname_list = shuffle_datasets_lessMemory(patchname_list)
                for index in range(len(patchname_list)):
                    single_coordinate = coordinate_list[patchname_list[index]]
                    init_h = single_coordinate['init_h']
                    end_h = single_coordinate['end_h']
                    init_w = single_coordinate['init_w']
                    end_w = single_coordinate['end_w']
                    init_s = single_coordinate['init_s']
                    end_s = single_coordinate['end_s']

                    if three_Dmodel:
                        real_A, real_B = generateTrainDataS1_3D(noise_img, init_s, end_s, init_h, end_h, init_w,
                                              end_w)
                    else:
                        real_A, real_B = generateTrainDataS1(noise_img, init_s, end_s, init_h, end_h, init_w,
                                            end_w)
                    
                    
                    if saveinput:
                        im = (noise_img[10] * 255 * scale_factor * 10).astype('uint8')
                        io.imsave(output_path + 'noise_img_%d.png' % scale_factor, im)
                        savecolorim(output_path + 'noise_imgC_%d.png' % scale_factor, im, norm=False)
    
                        real_A1 = real_A.cpu().numpy().squeeze().astype(np.float32)
                        im = (real_A1[-1] * 255 * scale_factor * 10).astype('uint8')
                        io.imsave(output_path + 'realA_%d.png' % scale_factor, im)
                        savecolorim(output_path + 'realAC_%d.png' % scale_factor, im, norm=False)
    
                        real_B1 = real_B.cpu().numpy().squeeze().astype(np.float32)
                        im = (real_B1[-1] * 255 * scale_factor * 10).astype('uint8')
                        io.imsave(output_path + 'realB_%d.png' % scale_factor, im)
                        savecolorim(output_path + 'realBC_%d.png' % scale_factor, im, norm=False)
                        exit()
                    
                    real_A = Variable(real_A)
                    fake_B = denoise_generator(real_A)
                    
                    # Pixel-wise loss
                    L1_loss = L1_pixelwise(fake_B, real_B)
                    L2_loss = L2_pixelwise(fake_B, real_B)
                    ################################################################################################################
                    optimizer_G.zero_grad()
                    
                    # Total loss
                    Total_loss = 0.5 * L1_loss + 0.5 * L2_loss
                    Total_loss.backward()
                    optimizer_G.step()
                    ################################################################################################################
                    batches_done = epoch * len(name_list) + index
                    batches_left = opt.n_epochs * len(name_list) - batches_done
                    time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                    prev_time = time.time()
                    ################################################################################################################
                    if index % 50 == 0:
                        time_end = time.time()
                        print('time cost', time_end - time_start, 's \n')
                        sys.stdout.write(
                            "\r[Epoch %d/%d] [Batch %d/%d] [Total loss: %f, L1 Loss: %f, L2 Loss: %f] ETA: %s"
                            % (epoch, opt.n_epochs, index + ni * len(patchname_list), len(name_list) * len(patchname_list),
                               Total_loss.item(), L1_loss.item(), L2_loss.item(), time_left,))
                               
        # ######### Valid  ##################
        if (epoch + 1) % 1 == 0:
            torch.save(denoise_generator.state_dict(), pth_path + '//G_' + str(epoch) + '.pth')
            if 'S1' in opt.denoise_model:
                testS1(pth_name=pth_path + '//G_' + str(epoch) + '.pth',
                     output_path=output_path + '/model%d_20120502_cell1/' % epoch)
            elif 'S2' in opt.denoise_model:
                testS2(pth_name=pth_path + '//G_' + str(epoch) + '.pth',
                     output_path=output_path + '/model%d_20120502_cell1/' % epoch)
            

def testS1(pth_name=None, output_path=None):
    scale_factor = 40  # 20  #
    save = True  # False
    saveInput = False  # True  #
    print('model_path: ', pth_path)
    for modelid in [85]:
        if not pth_name: pth_name = pth_path + '/G_%d.pth' % modelid
        if not output_path:
            output_path = pth_path + '/result_Model%d' % modelid + '_' + opt.datasets_folder + '/20120502_cell1/'
        os.makedirs(output_path, exist_ok=True)
        if (not three_Dmodel) and opt.out_channels == 1:
            name_list, noise_img, coordinate_list = \
                test_preprocess_lessMemoryNoTail_cell256(opt, '/20120502_cell1/', img_s2=5)
        else:
            name_list, noise_img, coordinate_list = \
                test_preprocess_lessMemoryNoTail_cell256(opt, '/20120502_cell1/', img_s2=opt.img_s)

        num_h = (math.floor((noise_img.shape[1] - opt.img_h) / opt.gap_h) + 1)  # [6000,512,512]
        num_w = (math.floor((noise_img.shape[2] - opt.img_w) / opt.gap_w) + 1)
        num_s = (math.floor((noise_img.shape[0] - opt.img_s) / opt.gap_s) + 1)
        print(num_h, num_w, num_s)
        
        if three_Dmodel:
            if 'Df' in opt.denoise_model:
                denoise_generator = Network_3D_Unet(in_channels=1, out_channels=1, channelstride=1,
                               final_sigmoid=True, layer_order='crg')
            else:
                denoise_generator = Network_3D_Unet(in_channels=1, out_channels=1, channelstride=1,
                           final_sigmoid=True, layer_order='crg', pool_kernel_size=(1, 2, 2))
        else:
            if 'RCAN' in opt.denoise_model:
                denoise_generator = RCAN(in_channels=opt.in_channels, out_channels=opt.out_channels)
            else:
                denoise_generator = Network_3D_Unet(UNet_type='2DUNet', in_channels=opt.in_channels,  # opt.img_s * 2,
                                                    out_channels=opt.out_channels, final_sigmoid=True,
                                                    layer_order='trg')
                        
        if os.path.exists(pth_name):
            denoise_generator.load_state_dict(torch.load(pth_name, map_location=torch.device(device)))
            print('Load Model from', opt.pth_path + '//' + opt.denoise_model + '//' + pth_name)
        denoise_generator.to(device)
        denoise_img = np.zeros(noise_img.shape)
        input_img = np.zeros(noise_img.shape)
        
        for index in range(len(name_list)):
            single_coordinate = coordinate_list[name_list[index]]
            init_h = single_coordinate['init_h']
            end_h = single_coordinate['end_h']
            init_w = single_coordinate['init_w']
            end_w = single_coordinate['end_w']
            init_s = single_coordinate['init_s']
            end_s = single_coordinate['end_s']
            
            # S1
            noise_patch = noise_img[init_s:end_s, init_h:end_h, init_w:end_w]
            if three_Dmodel:
                if 'Df' in opt.denoise_model:
                    # [1, 1, 40, 256, 256]
                    real_A = torch.from_numpy(np.expand_dims(np.expand_dims(noise_patch, 0), 0)).to(device)
                else:
                    real_A = torch.from_numpy(np.expand_dims(noise_patch, 0)).to(device)  # 1x40x256x256
                    real_A = real_A.permute([1, 0, 2, 3])  # 40x1x256x256
                    real_A = real_A.unsqueeze(0)  # 1x40x1x256x256
                    real_A = real_A.permute([1, 0, 2, 3, 4])  # 40x1x1x128x128
                    # real_A = real_A.permute([2, 0, 1, 3, 4])  # [40, 1, 1, 256, 256]
            else:
                real_A = torch.from_numpy(np.expand_dims(noise_patch, 0)).to(device)  # [1, 20, 256, 256]
                if not ('Df' in opt.denoise_model):
                    real_A = real_A.permute([1, 0, 2, 3])  # [20, 1, 256, 256]
            # real_A1 = real_A[-1].cpu().numpy().squeeze().astype(np.float32)
            # im = (real_A1 * 255 * scale_factor).astype('uint8')
            # io.imsave(output_path + 'realA_%d.png' % scale_factor, im)
            # savecolorim(output_path + 'realAC_%d.png' % scale_factor, im, norm=False)
            
            input_name = name_list[index]
            print('input_name -----> ', input_name)  # cell1_001_001_x0_y0_z0
            print('single_coordinate -----> ', single_coordinate)
            print('real_A -----> ', real_A.shape)  # [40, 1, 1, 150, 150]
            real_A = Variable(real_A)
            fake_B = denoise_generator(real_A)

            # fake_B1 = fake_B[-1].detach().cpu().numpy().squeeze().astype(np.float32)
            # im = (fake_B1 * 255 * scale_factor).astype('uint8')
            # io.imsave(output_path + '/fakeB_%d.png' % scale_factor, im)
            # savecolorim(output_path + '/fakeBC_%d.png' % scale_factor, im, norm=False)
            # exit()
            
            ################################################################################################################
            output_image = np.squeeze(fake_B.cpu().detach().numpy())
            raw_image = np.squeeze(real_A.cpu().detach().numpy())
            stack_start_w = int(single_coordinate['stack_start_w'])
            stack_end_w = int(single_coordinate['stack_end_w'])
            patch_start_w = int(single_coordinate['patch_start_w'])
            patch_end_w = int(single_coordinate['patch_end_w'])
            
            stack_start_h = int(single_coordinate['stack_start_h'])
            stack_end_h = int(single_coordinate['stack_end_h'])
            patch_start_h = int(single_coordinate['patch_start_h'])
            patch_end_h = int(single_coordinate['patch_end_h'])
            
            stack_start_s = int(single_coordinate['stack_start_s'])
            stack_end_s = int(single_coordinate['stack_end_s'])
            patch_start_s = int(single_coordinate['patch_start_s'])
            patch_end_s = int(single_coordinate['patch_end_s'])
            
            aaaa = output_image[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]
            bbbb = raw_image[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]
            
            denoise_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                = output_image[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w] * (
                    np.sum(bbbb) / np.sum(aaaa)) ** 0.5
            input_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                = raw_image[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]

        output_img0 = denoise_img.squeeze().astype(np.float32)
        output_img255 = np.clip(output_img0 * 255 * scale_factor, 0, 255).astype('uint8')
        print('output_img0.max(), min() = ', output_img0.max(), output_img0.min())
        del noise_img
        del denoise_img
        
        input_img = input_img.squeeze().astype(np.float32)
        input_img255 = np.clip(input_img.squeeze().astype(np.float32) * 255 * scale_factor, 0, 255).astype('uint8')
        
        if save:
            outname = output_path + '//' + name_list[0][:-3]
            io.imsave(outname + '_output.tif', np.clip(output_img0 * opt.normalize_factor, 0, 65535).astype('uint16'))
            if saveInput: io.imsave(outname + '_input.tif',
                        np.clip(input_img * opt.normalize_factor, 0, 65535).astype('uint16'))
            for i in range(0, input_img.shape[0], 10):
                io.imsave(outname + '_outT%d.png' % i, output_img255[i])
                oname = outname + '_outT%dcolor.png' % i
                savecolorim(oname, output_img255[i], norm=False)
                print('\n save to ', oname)
                if saveInput:
                    io.imsave(outname + '_inT%d.png' % i, input_img255[i])
                    savecolorim(outname + '_inT%dcolor.png' % i, input_img255[i], norm=False)


def testS2(pth_name=None, output_path=None):
    scale_factor = 40  # 20  #
    save = True  # False
    saveInput = False  # True  #
    print('model_path: ', pth_path)
    modelid = 1
    if not pth_name: pth_name = pth_path + '/G_%d.pth' % modelid
    if not output_path: output_path = pth_path + '/result_Model%d' % modelid + '_' + opt.datasets_folder + '/20120502_cell1/'
    os.makedirs(output_path, exist_ok=True)
   
    name_list, noise_img, coordinate_list = test_preprocess_lessMemoryNoTail_cell256_S2(opt, subdir='/20120502_cell1/')

    num_h = (math.floor((noise_img.shape[1] - opt.img_h) / opt.gap_h) + 1)  # [6000,512,512]
    num_w = (math.floor((noise_img.shape[2] - opt.img_w) / opt.gap_w) + 1)
    num_s = (math.floor((noise_img.shape[0] - opt.img_s) / opt.gap_s) + 1)
    print(num_h, num_w, num_s)
    
    if three_Dmodel:
        denoise_generator = Network_3D_Unet(in_channels=1, out_channels=1,
                            channelstride=(opt.img_s*2-2)//opt.out_channels, final_sigmoid=True, layer_order='crg')
    else:
        if 'RCAN' in opt.denoise_model:
            denoise_generator = RCAN(in_channels=opt.in_channels, out_channels=opt.out_channels)
        else:
            denoise_generator = Network_3D_Unet(UNet_type='2DUNet', in_channels=opt.in_channels,  # opt.img_s * 2,
                                                out_channels=opt.out_channels, final_sigmoid=True, layer_order='trg')
    
    if torch.cuda.is_available():
        print('Using GPU.')
    
    if os.path.exists(pth_name):
        denoise_generator.load_state_dict(torch.load(pth_name, map_location=torch.device(device)))
        print('Load Model from', opt.pth_path + '//' + opt.denoise_model + '//' + pth_name)
    denoise_generator.to(device)
    denoise_img = np.zeros(noise_img.shape)
    input_img = np.zeros(noise_img.shape)
    
    for index in range(len(name_list)):
        single_coordinate = coordinate_list[name_list[index]]
        init_s = single_coordinate['init_s']
        
        ## [1, 40, 256, 256]
        noise_patch = np.zeros([opt.in_channels, opt.img_h, opt.img_w], dtype=np.float32)
        if init_s < opt.in_channels//2:
            noise_patch[0:opt.in_channels//2, :, :] = noise_img[0, :, :]
            noise_patch[opt.in_channels//2 - init_s:opt.in_channels//2, :, :] = noise_img[0:init_s, :, :]
            noise_patch[opt.in_channels//2:, :, :] = noise_img[init_s + 1:init_s + opt.in_channels//2 + 1, :, :]
        elif init_s >= noise_img.shape[0] - opt.in_channels//2:
            noise_patch[0:opt.in_channels//2, :, :] = noise_img[init_s - opt.in_channels//2:init_s, :, :]
            noise_patch[opt.in_channels // 2:opt.in_channels // 2 + noise_img.shape[0] - init_s - 1, :, :] \
                = noise_img[noise_img.shape[0] + (init_s + 1 - noise_img.shape[0]):noise_img.shape[0], :, :]
            noise_patch[opt.in_channels // 2 + noise_img.shape[0] - init_s - 1:, :, :] \
                = noise_img[noise_img.shape[0] - 1:, :, :]
        else:
            noise_patch[0:opt.in_channels//2, :, :] = noise_img[init_s - opt.in_channels//2:init_s, :, :]
            noise_patch[opt.in_channels//2:, :, :] = noise_img[init_s + 1:init_s + opt.in_channels//2 + 1, :, :]

        
        if three_Dmodel:
            real_A = torch.from_numpy(np.expand_dims(np.expand_dims(noise_patch, 0), 0)).to(device)
        else:
            real_A = torch.from_numpy(np.expand_dims(noise_patch, 0)).to(device)
        # real_A1 = real_A[-1].cpu().numpy().squeeze().astype(np.float32) * opt.normalize_factor  # 0~65535
        # real_A1 = np.clip((real_A1 - real_A1.min()) / real_A1.max() * 65535, 0, 65535).astype('uint16')
        # io.imsave(root + '/pth/S1_2D/real_A.png', (real_A1 / 65535 * 255).astype('uint8'))
        # exit()
        
        input_name = name_list[index]
        print('input_name -----> ', input_name)  # cell1_001_001_x0_y0_z0
        print('single_coordinate -----> ', single_coordinate)
        print('real_A -----> ', real_A.shape)  # [1, 1, 20, 150, 150]
        real_A = Variable(real_A)
        fake_B = denoise_generator(real_A)
        
        ################################################################################################################
        output_image = np.squeeze(fake_B.cpu().detach().numpy())
        raw_image = np.squeeze(real_A.cpu().detach().numpy())
        
        stack_start_s = int(single_coordinate['stack_start_s'])
        patch_start_s = int(single_coordinate['patch_start_s'])
        
        aaaa = output_image[:, :]
        bbbb = raw_image[patch_start_s:patch_start_s+1, :, :]
        
        denoise_img[stack_start_s:stack_start_s+1, :, :] \
            = output_image[:, :] * (np.sum(bbbb) / np.sum(aaaa)) ** 0.5
        input_img[stack_start_s:stack_start_s+1, :, :] = raw_image[patch_start_s:patch_start_s+1, :, :]
    
    output_img0 = denoise_img.squeeze().astype(np.float32)
    output_img255 = np.clip(output_img0 * 255 * scale_factor, 0, 255).astype('uint8')
    output_img = np.clip(output_img0 * opt.normalize_factor, 0, 65535).astype('uint16')
    print('output_img.max(), output_img.min() = ', output_img.max(), output_img.min())
    del noise_img
    del denoise_img
    
    input_img = input_img.squeeze().astype(np.float32)
    input_img255 = np.clip(input_img.squeeze().astype(np.float32) * 255 * scale_factor, 0, 255).astype('uint8')
    input_img = np.clip(input_img * opt.normalize_factor, 0, 65535).astype('uint16')
    
    if save:
        outname = output_path + '//' + name_list[0][:-3]
        if saveInput: io.imsave(outname + '_input.tif', input_img)
        io.imsave(outname + '_output.tif', output_img)
        for i in range(0, input_img.shape[0], 10):
            io.imsave(outname + '_outT%d.png' % i, output_img255[i])
            oname = outname + '_outT%dcolor.png' % i
            savecolorim(oname, output_img255[i], norm=False)
            print('\n save to ', oname)
            if saveInput:
                io.imsave(outname + '_inT%d.png' % i, input_img255[i])
                savecolorim(outname + '_inT%dcolor.png' % i, input_img255[i], norm=False)


########################################################################################################################
import matplotlib.pyplot as plt


def to_color(arr, pmin=1, pmax=99.8, colors=((0, 1, 0), (1, 0, 1), (0, 1, 1))):
    """Converts a 2D or 3D stack to a colored image (maximal 3 channels).

    Parameters
    ----------
    arr : numpy.ndarray
        2D or 3D input data
    pmin : float
        lower percentile, pass -1 if no lower normalization is required
    pmax : float
        upper percentile, pass -1 if no upper normalization is required
    gamma : float
        gamma correction
    colors : list
        list of colors (r,g,b) for each channel of the input

    Returns
    -------
    numpy.ndarray
        colored image
    """
    if not arr.ndim in (2, 3):
        raise ValueError("only 2d or 3d arrays supported")
    
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    
    ind_min = np.argmin(arr.shape)
    arr = np.moveaxis(arr, ind_min, 0).astype(np.float32)
    
    out = np.zeros(arr.shape[1:] + (3,))
    
    eps = 1.e-20
    if pmin >= 0:
        mi = np.percentile(arr, pmin, axis=(1, 2), keepdims=True)
    else:
        mi = 0
    
    if pmax >= 0:
        ma = np.percentile(arr, pmax, axis=(1, 2), keepdims=True)
    else:
        ma = 1. + eps
    
    arr_norm = (1. * arr - mi) / (ma - mi + eps)
    
    for i_stack, col_stack in enumerate(colors):
        if i_stack >= len(arr):
            break
        for j, c in enumerate(col_stack):
            out[..., j] += c * arr_norm[i_stack]
    
    return np.clip(out, 0, 1)


def savecolorim(save, im, norm=True, **imshow_kwargs):
    # im: Uint8
    imshow_kwargs['cmap'] = 'magma'
    if not norm:  # 不对当前图片归一化处理，直接保存
        imshow_kwargs['vmin'] = 0
        imshow_kwargs['vmax'] = 255
    
    im = np.asarray(im)
    im = np.stack(map(to_color, im)) if 1 < im.shape[-1] <= 3 else im
    ndim_allowed = 2 + int(1 <= im.shape[-1] <= 3)
    proj_axis = tuple(range(1, 1 + max(0, im[0].ndim - ndim_allowed)))
    im = np.max(im, axis=proj_axis)
    
    # plt.imshow(im, **imshow_kwargs)
    # cb = plt.colorbar(fraction=0.05, pad=0.05)
    # cb.ax.tick_params(labelsize=23)  # 设置色标刻度字体大小。
    # # font = {'size': 16}
    # # cb.set_label('colorbar_title', fontdict=font)
    # plt.show()
    
    plt.imsave(save, im, **imshow_kwargs)



if __name__ == '__main__':
    method = opt.denoise_model
    os.makedirs(opt.output_dir, exist_ok=True)
    
    pth_path = opt.pth_path + '/' + method
    os.makedirs(pth_path, exist_ok=True)
    if '3D' in opt.denoise_model:
        three_Dmodel = True  #
    else:
        three_Dmodel = False
    
    
    train()
    testS1()
    # testS2()
