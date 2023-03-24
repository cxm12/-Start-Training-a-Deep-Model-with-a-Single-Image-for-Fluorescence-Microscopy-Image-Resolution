import numpy as np
import os
import tifffile as tiff
import random
import math


def shuffle_datasets(train_raw, train_GT, name_list):
    index_list = list(range(0, len(name_list)))
    # print('index_list -----> ',index_list)
    random.shuffle(index_list)
    random_index_list = index_list
    # print('index_list -----> ',index_list)
    new_name_list = list(range(0, len(name_list)))
    train_raw = np.array(train_raw)
    # print('train_raw shape -----> ',train_raw.shape)
    train_GT = np.array(train_GT)
    # print('train_GT shape -----> ',train_GT.shape)
    new_train_raw = train_raw
    new_train_GT = train_GT
    for i in range(0, len(random_index_list)):
        # print('i -----> ',i)
        new_train_raw[i, :, :, :] = train_raw[random_index_list[i], :, :, :]
        new_train_GT[i, :, :, :] = train_GT[random_index_list[i], :, :, :]
        new_name_list[i] = name_list[random_index_list[i]]
    # new_train_raw = np.expand_dims(new_train_raw, 4)
    # new_train_GT = np.expand_dims(new_train_GT, 4)
    return new_train_raw, new_train_GT, new_name_list



def get_gap_s(args, img, stack_num):
    whole_w = img.shape[2]
    whole_h = img.shape[1]
    whole_s = img.shape[0]
    print('whole_w -----> ', whole_w)
    print('whole_h -----> ', whole_h)
    print('whole_s -----> ', whole_s)
    w_num = math.floor((whole_w - args.img_w) / args.gap_w) + 1
    h_num = math.floor((whole_h - args.img_h) / args.gap_h) + 1
    s_num = math.ceil(args.train_datasets_size / w_num / h_num / stack_num)
    print('w_num -----> ', w_num)
    print('h_num -----> ', h_num)
    print('s_num -----> ', s_num)
    gap_s = math.floor((whole_s - args.img_s * 2) / (s_num - 1))
    print('gap_s -----> ', gap_s)
    return gap_s



def Cell_namelst(args):
    im_folder = args.datasets_path + '//' + args.datasets_folder
    name_list = []
    subdirlst = []
    flst = list(os.walk(im_folder, topdown=False))
    print('list(os.walk(im_folder, topdown=False)) -----> ', flst)
    print('len(flst)', len(flst))
    for subdir in flst[:-1]:
        # print('subdir', subdir, 'subdir[0]', subdir[0])
        if not ('20120502_cell1' in subdir[0]):
            continue
        subdirlst.append(subdir[0])
        nam_dirlst = []
        for im_name in subdir[-1]:
            # if not ('cell1_001_001' in im_name):
            #     continue
            if not ('tif' in im_name):
                continue
            nam_dirlst.append(im_name)
        name_list.append(nam_dirlst)
    
    return name_list, subdirlst


def train_preprocess_lessMemoryMulStacks_Cellim_S1(args, im_name, subdir, NB=False):
    img_h = args.img_h
    img_w = args.img_w
    if NB:
        img_s2 = args.img_s
    else:
        img_s2 = args.img_s * 2
    gap_s2 = args.gap_s
    gap_h = args.gap_h
    gap_w = args.gap_w

    nameim_list = []
    coordinateim_list = {}

    noise_im = tiff.imread(subdir + '//' + im_name)

    # gap_s2 = get_gap_s(args, noise_im, stack_num=1)
    print('noise_im.max() -----> ', noise_im.min(), noise_im.max())
    noise_im = (noise_im - noise_im.min()).astype(np.float32) / args.normalize_factor
    whole_w = noise_im.shape[2]
    whole_h = noise_im.shape[1]
    if args.train_datasets_size > 0:
        whole_s = args.train_datasets_size
    else:
        whole_s = noise_im.shape[0]

    for x in range(0, int((whole_h - img_h + gap_h) / gap_h)):
        for y in range(0, int((whole_w - img_w + gap_w) / gap_w)):
            for z in range(0, int((whole_s - img_s2 + gap_s2) / gap_s2)):
                single_coordinate = {'init_h': 0, 'end_h': 0, 'init_w': 0, 'end_w': 0, 'init_s': 0, 'end_s': 0}
                init_h = gap_h * x
                end_h = gap_h * x + img_h
                init_w = gap_w * y
                end_w = gap_w * y + img_w
                init_s = gap_s2 * z
                end_s = gap_s2 * z + img_s2
                single_coordinate['init_h'] = init_h
                single_coordinate['end_h'] = end_h
                single_coordinate['init_w'] = init_w
                single_coordinate['end_w'] = end_w
                single_coordinate['init_s'] = init_s
                single_coordinate['end_s'] = end_s
                # noise_patch1 = noise_im[init_s:end_s,init_h:end_h,init_w:end_w]
                patch_name = im_name[:-4] + '_x' + str(x) + '_y' + str(y) + '_z' + str(z)
                nameim_list.append(patch_name)
                coordinateim_list[patch_name] = single_coordinate

    print('noise_im.max() -----> ', noise_im.min(), noise_im.max())
    return nameim_list, noise_im, coordinateim_list



def shuffle_datasets_lessMemory(name_list):
    index_list = list(range(0, len(name_list)))
    # print('index_list -----> ',index_list)
    random.shuffle(index_list)
    random_index_list = index_list
    # print('index_list -----> ',index_list)
    new_name_list = list(range(0, len(name_list)))
    for i in range(0, len(random_index_list)):
        new_name_list[i] = name_list[random_index_list[i]]
    return new_name_list


#################################################################################
def test_preprocess_lessMemoryNoTail_cell256(args, subdir='/20120502_cell1/', img_s2=0):
    img_h = args.img_h
    img_w = args.img_w
    if img_s2 == 0:
        img_s2 = args.img_s
    gap_s2 = args.gap_s
    cut_s = 0
    im_folder = args.datasets_path + '//' + args.datasets_folder + subdir
    filelst = [im_folder + '/cell1_001_001.tif']
    # filelst = glob.glob(im_folder + '*.tif')[:1]

    name_list = []
    coordinate_list = {}
    for im_dir in filelst:
        im_name = im_dir[len(im_folder)+1:]
        noise_im = tiff.imread(im_dir)
        print('noise_im shape -----> ', noise_im.shape)
        print('noise_im max -----> ', noise_im.max())
        print('noise_im min -----> ', noise_im.min())
        if args.test_datasize > 0: noise_im = noise_im[0:args.test_datasize, :args.test_datasizeh, :args.test_datasizeh]
        noise_im = (noise_im - noise_im.min()).astype(np.float32) / args.normalize_factor
        whole_s = noise_im.shape[0]

        num_s = math.ceil((whole_s - img_s2 + gap_s2) / gap_s2)

        for z in range(0, num_s):
            single_coordinate = {'init_h': 0, 'end_h': 0, 'init_w': 0, 'end_w': 0, 'init_s': 0, 'end_s': 0}
    
            end_h = img_h
            end_w = img_w
    
            if z != (num_s - 1):
                init_s = gap_s2 * z
                end_s = gap_s2 * z + img_s2
            else:
                init_s = whole_s - img_s2
                end_s = whole_s
            single_coordinate['end_h'] = end_h
            single_coordinate['end_w'] = end_w
            single_coordinate['init_s'] = init_s
            single_coordinate['end_s'] = end_s
    
            single_coordinate['stack_start_w'] = single_coordinate['patch_start_w'] = 0
            single_coordinate['stack_end_w'] = single_coordinate['patch_end_w'] = img_w
    
            single_coordinate['stack_start_h'] = single_coordinate['patch_start_h'] = 0
            single_coordinate['stack_end_h'] = single_coordinate['patch_end_h'] = img_h
    
            if z == 0:
                single_coordinate['stack_start_s'] = z * gap_s2
                single_coordinate['stack_end_s'] = z * gap_s2 + img_s2 - cut_s
                single_coordinate['patch_start_s'] = 0
                single_coordinate['patch_end_s'] = img_s2 - cut_s
            elif z == num_s - 1:
                single_coordinate['stack_start_s'] = whole_s - img_s2 + cut_s
                single_coordinate['stack_end_s'] = whole_s
                single_coordinate['patch_start_s'] = cut_s
                single_coordinate['patch_end_s'] = img_s2
            else:
                single_coordinate['stack_start_s'] = z * gap_s2 + cut_s
                single_coordinate['stack_end_s'] = z * gap_s2 + img_s2 - cut_s
                single_coordinate['patch_start_s'] = cut_s
                single_coordinate['patch_end_s'] = img_s2 - cut_s
    
            patch_name = im_name[:-4] + '_z' + str(z)
            name_list.append(patch_name)
            coordinate_list[patch_name] = single_coordinate
            
    return name_list, noise_im, coordinate_list


def test_preprocess_lessMemoryNoTail_cell256_S2(args, subdir='/20120502_cell1/'):
    img_h = args.img_h
    img_w = args.img_w
    im_folder = args.datasets_path + '//' + args.datasets_folder + subdir
    filelst = [im_folder + '/cell1_001_001.tif']
    # filelst = glob.glob(im_folder + '*.tif')[:1]
    
    name_list = []
    coordinate_list = {}
    for im_dir in filelst:
        im_name = im_dir[len(im_folder) + 1:]
        noise_im = tiff.imread(im_dir)
        print('noise_im shape -----> ', noise_im.shape)
        print('noise_im max -----> ', noise_im.max())
        print('noise_im min -----> ', noise_im.min())
        if args.test_datasize > 0: noise_im = noise_im[0:args.test_datasize, :args.test_datasizeh, :args.test_datasizeh]
        noise_im = (noise_im - noise_im.min()).astype(np.float32) / args.normalize_factor
        whole_s = noise_im.shape[0]
        num_s = whole_s
        
        for z in range(0, num_s):
            single_coordinate = {'init_h': 0, 'end_h': 0, 'init_w': 0, 'end_w': 0, 'init_s': 0, 'end_s': 0}
            
            end_h = img_h
            end_w = img_w
            
            if z != (num_s - 1):
                init_s = z
            else:
                init_s = whole_s - 1
            single_coordinate['end_h'] = end_h
            single_coordinate['end_w'] = end_w
            single_coordinate['init_s'] = init_s
            
            single_coordinate['stack_start_w'] = single_coordinate['patch_start_w'] = 0
            single_coordinate['stack_end_w'] = single_coordinate['patch_end_w'] = img_w
            
            single_coordinate['stack_start_h'] = single_coordinate['patch_start_h'] = 0
            single_coordinate['stack_end_h'] = single_coordinate['patch_end_h'] = img_h
            
            if z == 0:
                single_coordinate['stack_start_s'] = z
                single_coordinate['patch_start_s'] = 0
            elif z == num_s - 1:
                single_coordinate['stack_start_s'] = whole_s - 1
                single_coordinate['patch_start_s'] = 0
            else:
                single_coordinate['stack_start_s'] = z
                single_coordinate['patch_start_s'] = 0
            
            patch_name = im_name[:-4] + '_z' + str(z)
            name_list.append(patch_name)
            coordinate_list[patch_name] = single_coordinate
    
    return name_list, noise_im, coordinate_list
