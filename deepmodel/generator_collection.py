# Class to generate data for training
import numpy as np
import tensorflow.keras as keras
from deepmodel.generic import JsonLoader
import tifffile
import nibabel as nib


class DeepGenerator(keras.utils.Sequence):
    """
    This class instantiante the basic Generator Sequence object
    from which all Deep Interpolation generator should be generated.

    Parameters:
    json_path: a path to the json file used to parametrize the generator

    Returns:
    None
    """
    
    def __init__(self, json_path):
        local_json_loader = JsonLoader(json_path)
        local_json_loader.load_json()
        self.json_data = local_json_loader.json_data
        self.local_mean = 1
        self.local_std = 1
    
    def get_input_size(self):
        """
        This function returns the input size of the
        generator, excluding the batching dimension

        Parameters:
        None

        Returns:
        tuple: list of integer size of input array,
        excluding the batching dimension
        """
        local_obj = self.__getitem__(0)[0]
        
        return local_obj.shape[1:]
    
    def get_output_size(self):
        """
        This function returns the output size of
        the generator, excluding the batching dimension

        Parameters:
        None

        Returns:
        tuple: list of integer size of output array,
        excluding the batching dimension
        """
        local_obj = self.__getitem__(0)[1]
        
        return local_obj.shape[1:]
    
    def __len__(self):
        return 0
    
    def __getitem__(self, idx):
        return [np.array([]), np.array([])]
    
    def __get_norm_parameters__(self, idx):
        """
        This function returns the normalization parameters
        of the generator. This can potentially be different
        for each data sample

        Parameters:
        idx index of the sample

        Returns:
        local_mean
        local_std
        """
        local_mean = self.local_mean
        local_std = self.local_std
        
        return local_mean, local_std


class FmriGeneratorS3(DeepGenerator):
    
    def __init__(self, json_path):
        super().__init__(json_path)
        self.raw_data_file = self.json_data["train_path"]
        self.batch_size = self.json_data["batch_size"]
        self.pre_post_x = self.json_data["pre_post_x"]
        self.pre_post_y = self.json_data["pre_post_y"]
        self.pre_post_z = self.json_data["pre_post_z"]
        self.pre_post_t = self.json_data["pre_post_t"]
        
        self.start_frame = self.json_data["start_frame"]
        self.end_frame = self.json_data["end_frame"]
        self.total_nb_block = self.json_data["total_nb_block"]
        self.steps_per_epoch = self.json_data["steps_per_epoch"]
        
        if "center_omission_size" in self.json_data.keys():
            self.center_omission_size = self.json_data["center_omission_size"]
        else:
            self.center_omission_size = 1
        
        if "single_voxel_output_single" in self.json_data.keys():
            self.single_voxel_output_single = self.json_data[
                "single_voxel_output_single"]
        else:
            self.single_voxel_output_single = True
        if "initialize_list" in self.json_data.keys():
            self.initialize_list = self.json_data["initialize_list"]
        else:
            self.initialize_list = 1
            
        # We load the entire data as it fits into memory
        self.raw_data = nib.load(self.raw_data_file).get_fdata()
        if "smallIm" in self.json_data.keys():
            # self.raw_data = self.raw_data[:16, :16, :5, :6]
            self.raw_data = self.raw_data[:, :, :, :10]
        if "NeedGT" in self.json_data.keys():
            self.NeedGT = self.json_data["NeedGT"]
        else:
            self.NeedGT = True
        if "Pad0" in self.json_data.keys():  # 中间帧置为0
            self.Pad0 = self.json_data["Pad0"]
        else:
            self.Pad0 = False
            
        if self.NeedGT:
            msk1, msk2 = self.generate_mask_pair(self.raw_data)
            if self.Pad0:  # [64,64,50,175]
                self.raw_datam1 = self.generate_subimages_nodown(self.raw_data, msk1, start=0)
                self.raw_datam2 = self.generate_subimages_nodown(self.raw_data, msk2, start=0)  # start=1)  #
            else:  # [64//2,64//2,50,175]
                self.raw_datam1 = self.generate_subimages(self.raw_data, msk1)
                self.raw_datam2 = self.generate_subimages(self.raw_data, msk2)
        else:  # save test data for visualization
            self.raw_datam1 = self.raw_datam2 = self.raw_data

        self.data_shape = self.raw_datam1.shape
        print('self.data_shape', self.data_shape)
        if self.data_shape[3] < self.end_frame: self.end_frame = self.data_shape[3]
        middle_vol = np.round(np.array(self.data_shape) / 2).astype("int")
        range_middle = np.round(np.array(self.data_shape) / 4).astype("int")
        
        # We take the middle of the volume and time for range estimation to avoid edge effects
        local_center_data = self.raw_data[
                            middle_vol[0] - range_middle[0]: middle_vol[0] + range_middle[0],
                            middle_vol[1] - range_middle[1]: middle_vol[1] + range_middle[1],
                            middle_vol[2] - range_middle[2]: middle_vol[2] + range_middle[2],
                            middle_vol[3] - range_middle[3]: middle_vol[3] + range_middle[3],]
        # print(local_center_data[:,:,0,0])
        self.local_mean = np.mean(local_center_data.flatten())
        self.local_std = np.std(local_center_data.flatten())
        self.epoch_index = 0
        
        if self.initialize_list == 1:
            self.x_list = []
            self.y_list = []
            self.z_list = []
            self.t_list = []
            filling_array = np.zeros(self.data_shape, dtype=bool)  # [64//2,64//2,50,175]=8,960,000
            for index, value in enumerate(range(self.total_nb_block)):  # 9,000,000
                retake = True
                # print(index)
                while retake:
                    x_local, y_local, z_local, t_local = self.get_random_xyzt()
                    retake = False
                    if filling_array[x_local, y_local, z_local, t_local]:
                        retake = True
                filling_array[x_local, y_local, z_local, t_local] = True
                
                self.x_list.append(x_local)
                self.y_list.append(y_local)
                self.z_list.append(z_local)
                self.t_list.append(t_local)

    def get_random_xyzt(self):
        x_center = np.random.randint(0, self.data_shape[0])
        y_center = np.random.randint(0, self.data_shape[1])
        z_center = np.random.randint(0, self.data_shape[2])
        t_center = np.random.randint(self.start_frame, self.end_frame)
    
        return x_center, y_center, z_center, t_center

    def __len__(self):
        "Denotes the total number of batches"
        return int(np.floor(float(len(self.x_list) / self.batch_size)))

    def on_epoch_end(self):
        if self.steps_per_epoch * (self.epoch_index + 2) < self.__len__():
            self.epoch_index = self.epoch_index + 1
        else:
            # if we reach the end of the data, we roll over
            self.epoch_index = 0

    def generate_mask_pair(self, img):
        # prepare masks (N x C x H/2 x W/2)  #  (B x T x C x H/2 x W/2)
        h, w, c, t = img.shape  # [64,64,50,175]
        img = np.transpose(img, (3, 2, 0, 1))  # [175,50,64,64] t, c, h, w  #
        n, c, h, w = img.shape
        mask1 = np.zeros((n * h // 2 * w // 2 * 4,), dtype=np.bool)
        mask2 = np.zeros((n * h // 2 * w // 2 * 4,), dtype=np.bool)
        
        # prepare random mask pairs
        idx_pair = np.array([[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]])
        # rd_idx = np.zeros((t * h // 2 * w // 2,), dtype=np.int64)
        rd_idx = np.random.randint(0, 8, (n * h // 2 * w // 2,), dtype=np.int64)
        
        rd_pair_idx = idx_pair[rd_idx]
        rd_pair_idx += np.reshape(np.arange(start=0, stop=t * h // 2 * w // 2 * 4,
                                            step=4, dtype=np.int64), (-1, 1))
        # get masks
        mask1[rd_pair_idx[:, 0]] = 1
        mask2[rd_pair_idx[:, 1]] = 1
        return mask1, mask2
    
    def space_to_depth(self, x, block_size=2):
        # n, c, h, w = x.shape
        x = np.transpose(x, (0, 2, 3, 1))
        n, h, w, c = x.shape
        unfolded_x = x.reshape(n, h // block_size, block_size,
                               w // block_size, block_size, c)
        z = np.swapaxes(unfolded_x, 2, 3).reshape(n, h // block_size, w // block_size, -1)
        y = np.transpose(z, (0, 3, 1, 2))
        return y
    
    def generate_subimages(self, img, mask):
        """ x2 downscale input (64x64->32x32)"""
        h, w, c, t = img.shape  # [64,64,50,175]
        img = np.transpose(img, (3, 2, 0, 1))  # [175,50,64,64] t, c, h, w  #
        n, c, h, w = img.shape
        subimage = np.zeros((t, c, h // 2, w // 2), dtype=img.dtype)
        # per channel
        for i in range(c):
            img_per_channel = self.space_to_depth(img[:, i:i + 1, :, :], block_size=2)
            img_per_channel = np.reshape(np.transpose(img_per_channel, (0, 2, 3, 1)), -1)  # n, h, w, c
            # img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
            subimage[:, i:i + 1, :, :] = np.transpose(np.reshape(img_per_channel[mask], (n, h // 2, w // 2, 1)), (0, 3, 1, 2))
            # subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
        subimage = np.transpose(subimage, (2, 3, 1, 0))  # [32, 32, 50, 175]
        return subimage

    def generate_subimages_nodown(self, img, mask, start=0):
        """ Not downscale input (64x64->32x32) zero padding"""
        # [64,64,50,175]
        img = np.transpose(img, (3, 2, 0, 1))  # [175,50,64,64] t, c, h, w  #
        n, c, h, w = img.shape
        subimage = np.zeros((n, c, h, w), dtype=img.dtype)
        # per channel
        for i in range(c):
            img_per_channel = self.space_to_depth(img[:, i:i + 1, :, :], block_size=2)  # n, c, h, w
            img_per_channel = np.transpose(img_per_channel, (0, 2, 3, 1))  # n, h/2, w/2, c*4
            img_per_channel = np.reshape(img_per_channel, -1)  # n, h/2,w/2, c*4
            x = np.reshape(img_per_channel[mask], (n, h // 2, w // 2, 1))  # 716800->179200
            x = np.transpose(x, (0, 3, 1, 2))
            subimage[:, i, start:h:2, start:h:2] = x[:, 0, :, :]  # [175,50,64,64]
        subimage = np.transpose(subimage, (2, 3, 1, 0))  # [64, 64, 50, 175]
        return subimage

    def __getitem__(self, index):
        # This is to ensure we are going through the
        # entire data when steps_per_epoch<self.__len__
        index = index + self.steps_per_epoch * self.epoch_index
        # Generate indexes of the batch
        indexes = np.arange(index * self.batch_size, (index + 1) * self.batch_size)
        
        if self.json_data["InputF5"]:
            input_full = np.zeros(
                [self.batch_size, self.pre_post_x * 2 + 1,
                 self.pre_post_y * 2 + 1,
                 self.pre_post_z * 2 + 1,
                 self.pre_post_t * 2 + 1,], dtype="float32", )  # [Batch, 7, 7, 7, 1]
        else:
            input_full = np.zeros(
                [self.batch_size, self.pre_post_x * 2 + 1,
                 self.pre_post_y * 2 + 1,
                 self.pre_post_z * 2 + 1,
                 1, ], dtype="float32", )  # [Batch, 7, 7, 7, 1]

        # if not self.single_voxel_output_single:
        output_full = np.zeros(
                [self.batch_size,
                    self.pre_post_x * 2 + 1,
                    self.pre_post_y * 2 + 1,
                    self.pre_post_z * 2 + 1,
                    1,], dtype="float32")  # [Batch, 7, 7, 7, 1]
        
        for batch_index, sample_index in enumerate(indexes):
            # print('batch indexes=', indexes)
            local_x = self.x_list[sample_index]
            local_y = self.y_list[sample_index]
            local_z = self.z_list[sample_index]
            local_t = self.t_list[sample_index]  # [local_x, y, z, t] 中心像素坐标 从其前后左右上下各取3个像素位置
            # local_x = local_y = 2
            input, output = self.__data_generation__(
                local_x, local_y, local_z, local_t)
            input_full[batch_index, :, :, :, :] = input
            output_full[batch_index, :, :, :, :] = output
        return input_full, output_full
    
    def __data_generation__(self, local_x, local_y, local_z, local_t):
        " Generates data containing batch_size samples "
        if self.json_data["InputF5"]:
            input_full = np.zeros([1,
                 self.pre_post_x * 2 + 1,
                 self.pre_post_y * 2 + 1,
                 self.pre_post_z * 2 + 1,
                 self.pre_post_t * 2 + 1,], dtype="float32",)  # [1, 7, 7, 7, 1]
        else:
            input_full = np.zeros(
                [1,
                 self.pre_post_x * 2 + 1,
                 self.pre_post_y * 2 + 1,
                 self.pre_post_z * 2 + 1,
                 1,], dtype="float32",)  # [1, 7, 7, 7, 1]
        
        output_full = np.zeros([1,
                    self.pre_post_x * 2 + 1,
                    self.pre_post_y * 2 + 1,
                    self.pre_post_z * 2 + 1,
                    1,],dtype="float32")  # [1, 7, 7, 7, 1]
        
        # We cap the x y z t axis when touching the limit of the volume
        pre_x = min(local_x, self.pre_post_x)
        post_x = min(self.data_shape[0] - 1 - local_x, self.pre_post_x)
        pre_y = min(local_y, self.pre_post_y)
        post_y = min(self.data_shape[1] - 1 - local_y, self.pre_post_y)
        pre_z = min(local_z, self.pre_post_z)
        post_z = min(self.data_shape[2] - 1 - local_z, self.pre_post_z)
        pre_t = min(local_t, self.pre_post_t)
        post_t = min(self.data_shape[3] - 1 - local_t, self.pre_post_t)
       
        # 非0元素矩阵尺寸 [0~7, 0~7, 0~7, 0]
        if self.json_data["InputF5"]:
            input_full[0,
            (self.pre_post_x - pre_x): (self.pre_post_x + post_x + 1),  # 0~3:3~7
            (self.pre_post_y - pre_y): (self.pre_post_y + post_y + 1),
            (self.pre_post_z - pre_z): (self.pre_post_z + post_z + 1),
            (self.pre_post_t - pre_t): (self.pre_post_t + post_t + 1),  # 0~2:5
            ] = self.raw_datam1[
                (local_x - pre_x): (local_x + post_x + 1),  # 11:18  0~local_x-3: local_x+1~local_x+4
                (local_y - pre_y): (local_y + post_y + 1),  # 27:34
                (local_z - pre_z): (local_z + post_z + 1),  # 18:25
                (local_t - pre_t): (local_t + post_t + 1),]  # 83:86
        else:
            input_full[0,
            (self.pre_post_x - pre_x): (self.pre_post_x + post_x + 1),  # 0~3:3~7
            (self.pre_post_y - pre_y): (self.pre_post_y + post_y + 1),
            (self.pre_post_z - pre_z): (self.pre_post_z + post_z + 1),
            0,] = self.raw_datam1[
                (local_x - pre_x): (local_x + post_x + 1),  # 11:18  0~local_x-3: local_x+1~local_x+4
                (local_y - pre_y): (local_y + post_y + 1),  # 27:34
                (local_z - pre_z): (local_z + post_z + 1),  # 18:25
                local_t, ]  # 83:86
        input_full = (input_full.astype("float32") - self.local_mean) / self.local_std  # [1, 7, 7, 7, 5]

        if self.NeedGT:
            output_full[0,
            (self.pre_post_x - pre_x): (self.pre_post_x + post_x + 1),  # 0~3:3~7
            (self.pre_post_y - pre_y): (self.pre_post_y + post_y + 1),
            (self.pre_post_z - pre_z): (self.pre_post_z + post_z + 1),
            0, ] = self.raw_datam2[
                   (local_x - pre_x): (local_x + post_x + 1),  # 11:18  0~local_x-3: local_x+1~local_x+4
                   (local_y - pre_y): (local_y + post_y + 1),  # 27:34
                   (local_z - pre_z): (local_z + post_z + 1),  # 18:25
                   local_t,  # 83:86
                   ]  # [0, :, :, :, 0] = input_full[0, :, :, :, self.pre_post_t]
            output_full = (output_full.astype("float32") - self.local_mean) / self.local_std  # [1, 7, 7, 7, 1]
        else:
            output_full = input_full[:, :, :, :, 0:1]
        return input_full, output_full


class SingleTifGeneratorS1(DeepGenerator):
    
    def __init__(self, json_path):
        "Initialization"
        super().__init__(json_path)
        self.raw_data_file = self.json_data["train_path"]
        self.batch_size = self.json_data["batch_size"]
        if "pre_post_frame" in self.json_data.keys():
            self.pre_frame = self.json_data["pre_post_frame"]
            self.post_frame = self.json_data["pre_post_frame"]
        else:
            self.pre_frame = self.json_data["pre_frame"]
            self.post_frame = self.json_data["post_frame"]
        self.pre_post_omission = self.json_data["pre_post_omission"]
        self.start_frame = self.json_data["start_frame"]
        self.steps_per_epoch = self.json_data["steps_per_epoch"]
        if "randomize" in self.json_data.keys():
            self.randomize = self.json_data["randomize"]
        else:
            self.randomize = 1
            
        # This is compatible with negative frames
        self.end_frame = self.json_data["end_frame"]
        
        with tifffile.TiffFile(self.raw_data_file) as tif:
            self.raw_data = tif.asarray()  # 【100，512，512】 uint16
        self.Pad0 = False
            
        if "NeedGT" in self.json_data.keys():
            self.NeedGT = self.json_data["NeedGT"]
        else:
            self.NeedGT = True
        if self.NeedGT:   # save test data for visualization
            msk1, msk2 = self.generate_mask_pair(self.raw_data)
            self.raw_datam1 = self.generate_subimages(self.raw_data, msk1)
            self.raw_datam2 = self.generate_subimages(self.raw_data, msk2)
        else:
            self.raw_datam1 = self.raw_datam2 = self.raw_data
            
        self.total_frame_per_movie = self.raw_datam1.shape[0]
        if self.end_frame < 0 or self.total_frame_per_movie < self.end_frame:
            self.img_per_movie = self.total_frame_per_movie - self.start_frame
        else:
            self.img_per_movie = self.end_frame + 1 - self.start_frame
        
        average_nb_samples = 1000
        local_data = self.raw_data[0:average_nb_samples, :, :].flatten()
        local_data = local_data.astype("float32")
        self.local_mean = np.mean(local_data)
        self.local_std = np.std(local_data)
        self.epoch_index = 0
        
        self.list_samples = np.arange(
            self.pre_frame + self.pre_post_omission + self.start_frame,
            self.start_frame + self.img_per_movie - self.post_frame - self.pre_post_omission, )  # （41,）
        if self.randomize:
            np.random.shuffle(self.list_samples)
    
    def __getitem__(self, index):
        if self.steps_per_epoch > 0:
            index = index + self.steps_per_epoch * self.epoch_index
        
        # Generate indexes of the batch
        indexes = np.arange(index * self.batch_size,
                            (index + 1) * self.batch_size)
        shuffle_indexes = self.list_samples[indexes]
        
        if self.NeedGT:  # train 512
            if self.Pad0:  # train 512
                shape = [1, 2*self.raw_datam1.shape[1], 2*self.raw_datam1.shape[2], 1]
            else:  # train 256
                shape = [1, self.raw_datam1.shape[1], self.raw_datam1.shape[2], 1]
        else:  # test 512
            shape = [1, self.raw_datam1.shape[1], self.raw_datam1.shape[2], 1]
        
        if self.pre_frame > 0:
            input_full = np.zeros([self.batch_size, shape[1], shape[2], self.pre_frame + self.post_frame,
                ], dtype="float32", )  # [B,512/2,512/2,1]
        else:
            input_full = np.zeros([self.batch_size, shape[1], shape[2], 1, ], dtype="float32", )  # [B,512/2,512/2,1]
        output_full = np.zeros([self.batch_size, shape[1], shape[2], 1], dtype="float32", )  # [B,512,512,1]
        
        for batch_index, frame_index in enumerate(shuffle_indexes):
            X, Y = self.__data_generation__(frame_index)
            
            input_full[batch_index, :, :, :] = X
            output_full[batch_index, :, :, :] = Y
            
        if (not self.NeedGT) and (not self.Pad0):
            input_full = self.space_to_batch(input_full, block_size=2)
        return input_full, output_full  # make sure input_full[256x256]
    
    def space_to_batch(self, x, block_size=2):  # [100,512,512,1]
        x = np.transpose(x, (3, 1, 2, 0))  # [1,512,512,100]
        c, h, w, n = x.shape
        unfolded_x = x.reshape(c, h // block_size, block_size, w // block_size, block_size, n)
        z = np.swapaxes(unfolded_x, 2, 3).reshape(c, h // block_size, w // block_size, -1)  # [c,512/2,512/2,4n]
        z = np.transpose(z, (3,1,2,0))  # [4n,512/2,512/2,1]
        return z
    
    def __data_generation__(self, index_frame):
        "Generates data containing batch_size samples"
        if self.Pad0 and self.NeedGT:  # train 512
            shape = [1, 2*self.raw_datam1.shape[1], 2*self.raw_datam1.shape[2], 1]
        elif not self.NeedGT:  # test 512
            shape = [1, self.raw_datam1.shape[1], self.raw_datam1.shape[2], 1]
        elif (not self.Pad0) and self.NeedGT:  # train 256
            shape = [1, self.raw_datam1.shape[1], self.raw_datam1.shape[2], 1]
        output_full = np.zeros(
            [1, shape[1], shape[2], 1], dtype="float32")

        if self.pre_frame > 0:
            input_full = np.zeros([1, shape[1], shape[2], self.pre_frame + self.post_frame,], dtype="float32", )
            
            input_index = np.arange(
                index_frame - self.pre_frame - self.pre_post_omission,
                index_frame + self.post_frame + self.pre_post_omission + 1, )
            input_index = input_index[input_index != index_frame]
            for index_padding in np.arange(self.pre_post_omission + 1):
                input_index = input_index[input_index != index_frame - index_padding]
                input_index = input_index[input_index != index_frame + index_padding]
                
            data_img_input = self.raw_datam1[input_index, :, :]
            data_img_input = np.swapaxes(data_img_input, 1, 2)  # 60 h w -> 60 w h
            data_img_input = np.swapaxes(data_img_input, 0, 2)  # 60 w h -> h w 60
            img_in_shape = data_img_input.shape
        else:
            input_full = np.zeros([1, shape[1], shape[2], 1,], dtype="float32", )
            data_img_input = self.raw_datam1[index_frame, :, :]
            img_in_shape = data_img_input.shape
            data_img_input = np.expand_dims(data_img_input, -1)
        
        data_img_input = (data_img_input.astype("float32") - self.local_mean) / self.local_std
        data_img_output = self.raw_datam2[index_frame, :, :]
        img_out_shape = data_img_output.shape
        data_img_output = (data_img_output.astype("float32") - self.local_mean) / self.local_std
        if self.Pad0 and self.NeedGT:
            input_full[0, 0:img_in_shape[0]*2:2, 0:img_in_shape[1]*2:2, :] = data_img_input
            output_full[0, 0:img_out_shape[0]*2:2, 0:img_out_shape[1]*2:2, 0] = data_img_output
        else:
            input_full[0, : img_in_shape[0], : img_in_shape[1], :] = data_img_input
            output_full[0, : img_out_shape[0], : img_out_shape[1], 0] = data_img_output

        return input_full, output_full

    def generate_mask_pair(self, img):
        # prepare masks (N x C x H/2 x W/2)  #  [100.512.512]
        img = np.expand_dims(img, 1)  # [100,1,512.512]
        n, c, h, w = img.shape
        mask1 = np.zeros((n * h // 2 * w // 2 * 4,), dtype=np.bool)
        mask2 = np.zeros((n * h // 2 * w // 2 * 4,), dtype=np.bool)
    
        # prepare random mask pairs
        idx_pair = np.array([[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]])
        rd_idx = np.random.randint(0, 8, (n * h // 2 * w // 2,), dtype=np.int64)
        rd_pair_idx = idx_pair[rd_idx]
        rd_pair_idx += np.reshape(np.arange(start=0, stop=n * h // 2 * w // 2 * 4,
                                            step=4, dtype=np.int64), (-1, 1))
        # get masks
        mask1[rd_pair_idx[:, 0]] = 1
        mask2[rd_pair_idx[:, 1]] = 1
        return mask1, mask2

    def generate_subimages(self, img, mask):
        """ x2 downscale input (512->256)"""
        n, h, w = img.shape
        img_per_channel = self.space_to_depth(img, block_size=2)  # n, h, w, 4
        img_per_channel = np.reshape(np.transpose(img_per_channel, (0, 2, 3, 1)), -1)  # n, w, 4, h
        subimage = np.reshape(img_per_channel[mask], (n, h // 2, w // 2))  # n, h, w
        return subimage  # [100,256,256]

    def space_to_depth(self, x, block_size=2):
        x = np.expand_dims(x, -1)  # [100,512,512,1]
        n, h, w, c = x.shape
        unfolded_x = x.reshape(n, h // block_size, block_size, w // block_size, block_size, c)
        z = np.swapaxes(unfolded_x, 2, 3).reshape(n, h // block_size, w // block_size, -1)  # [100,512/2,512/2,4]
        return z

    def get_output_size(self):
        local_obj = self.__getitem__(0)[1]
        return local_obj.shape[1:]

    def __len__(self):
        "Denotes the total number of batches"
        return int(np.floor(float(len(self.list_samples)) / self.batch_size))

    def on_epoch_end(self):
        # We only increase index if steps_per_epoch is set
        # to positive value. -1 will force the generator
        # to not iterate at the end of each epoch
        if self.steps_per_epoch > 0:
            if self.steps_per_epoch * (self.epoch_index + 2) < self.__len__():
                self.epoch_index = self.epoch_index + 1
            else:
                # if we reach the end of the data, we roll over
                self.epoch_index = 0


class SingleTifGeneratorS2(DeepGenerator):
    
    def __init__(self, json_path):
        super().__init__(json_path)
        self.raw_data_file = self.json_data["train_path"]
        self.batch_size = self.json_data["batch_size"]
        self.pre_frame = self.post_frame = self.json_data["pre_post_frame"]
        self.start_frame = self.json_data["start_frame"]
        self.steps_per_epoch = self.json_data["steps_per_epoch"]
        self.end_frame = self.json_data["end_frame"]
        
        with tifffile.TiffFile(self.raw_data_file) as tif:
            self.raw_data = tif.asarray()
        self.total_frame_per_movie = self.raw_data.shape[0]
        
        if self.end_frame < 0 or self.total_frame_per_movie < self.end_frame:
            self.img_per_movie = self.total_frame_per_movie - self.start_frame
        else:
            self.img_per_movie = self.end_frame + 1 - self.start_frame
        
        average_nb_samples = 1000
        local_data = self.raw_data[0:average_nb_samples, :, :].flatten()
        local_data = local_data.astype("float32")
        self.local_mean = np.mean(local_data)
        self.local_std = np.std(local_data)
        self.raw_data = (self.raw_data.astype("float32") - self.local_mean) / self.local_std
        self.epoch_index = 0
        self.list_samples = np.arange(self.pre_frame + self.start_frame,
            self.start_frame + self.img_per_movie - self.post_frame, )
        np.random.shuffle(self.list_samples)
        
        self.testinput_msk = True  # False  #
        #  'GTall_ch1_Inmsk0_Inmf0' GTmsk_ch1_Inmsk0_Inmf0
        self.Input_mid_frame0 = True  # False  # input的中间帧删除or置0
        if self.Input_mid_frame0:  # (_Inmf)
            self.removeMdFram = False  # True  #
            if self.removeMdFram:  # (_Inmf) input的中间帧删除（**不是置0**）
                self.inchannel = self.pre_frame + self.post_frame
            else:  # (_Inmf0)
                self.inchannel = self.pre_frame + self.post_frame + 1
        else:
            self.inchannel = self.pre_frame + self.post_frame + 1
        self.Inputmask0 = True  # False  # Input 像素置0 (_Inmsk0) or 相邻像素

        self.GT0 = True  # False  # GT基于0矩阵(GT0) or Input(GTmsk)
        if self.Input_mid_frame0:
            self.GT0 = False  # GTall
        self.GT_catMsk = False  # True  # concat(GT,mask) 2D output (_ch1/2)
        if self.GT_catMsk:
            self.gtchannel = 2
        else:
            self.gtchannel = 1  # _ch1
    
    def __len__(self):
        "Denotes the total number of batches"
        return int(np.floor(float(len(self.list_samples)) / self.batch_size))
    
    def on_epoch_end(self):
        if self.steps_per_epoch > 0:
            if self.steps_per_epoch * (self.epoch_index + 2) < self.__len__():
                self.epoch_index = self.epoch_index + 1
            else:
                # if we reach the end of the data, we roll over
                self.epoch_index = 0
    
    def __getitem__(self, index):
        if self.steps_per_epoch > 0:
            index = index + self.steps_per_epoch * self.epoch_index
        # Generate indexes of the batch
        indexes = np.arange(index * self.batch_size, (index + 1) * self.batch_size)
        shuffle_indexes = self.list_samples[indexes]
        input_full = np.zeros([
            self.batch_size,
            self.raw_data.shape[1],
            self.raw_data.shape[2],
            self.inchannel, ], dtype="float32", )  # 【5，512，512，60】
        output_full = np.zeros(
            [self.batch_size, self.raw_data.shape[1],
             self.raw_data.shape[2], self.gtchannel], dtype="float32", )  # 【5，512，512，2】

        for batch_index, frame_index in enumerate(shuffle_indexes):
            X, Y = self.__data_generation__(frame_index)
            input_full[batch_index, :, :, :] = X
            output_full[batch_index, :, :, :] = Y
        return input_full, output_full
    
    def __data_generation__(self, index_frame):
        " Generates data containing batch_size samples "
        input_full = np.zeros([1, self.raw_data.shape[1],
                               self.raw_data.shape[2], self.inchannel, ], dtype="float32", )
        output_full = np.zeros([1, self.raw_data.shape[1], self.raw_data.shape[2], self.gtchannel], dtype="float32")
        
        input_index = np.arange(index_frame - self.pre_frame, index_frame + self.post_frame + 1, )
        if self.Input_mid_frame0:
            if self.removeMdFram:  # input丢弃中间(2 * self.pre_post_omission + 1)帧
                input_index = input_index[input_index != index_frame]
                # input_index = input_index[input_index != index_frame - 1] # [input_index != index_frame + 1]
                data_img_input = self.raw_data[input_index, :, :]
            else:  # input中间帧=0
                data_img_input = self.raw_data[input_index, :, :]
                data_img_input[self.post_frame, :, :] = np.zeros_like(data_img_input[0, :, :])
        else:  # input保留中间帧
            data_img_input = self.raw_data[input_index, :, :]
        
        data_img_input = np.swapaxes(np.swapaxes(data_img_input, 1, 2), 0, 2)  # 60 h w -> 60 w h -> h w 60
        data_img_input_midframe = self.raw_data[index_frame:index_frame+1, :, :]
        data_img_input_midframe = np.swapaxes(np.swapaxes(data_img_input_midframe, 1, 2), 0, 2)  # 60 h w -> 60 w h -> h w 60

        img_in_shape = data_img_input.shape

        if self.json_data["NeedGT"]:  # gtmsk
            data_img_inputmsk, mask, data_img_output = self.generate_InputGT(data_img_input, data_img_input_midframe)  # mask=0: 图像像素置为相邻的随机像素值
            input_full[0, : img_in_shape[0], : img_in_shape[1], :] = data_img_inputmsk
            if self.GT_catMsk:
                output_full[0, : img_in_shape[0], : img_in_shape[1], 0] = data_img_output
                output_full[0, : img_in_shape[0], : img_in_shape[1], 1] = mask
            else:
                output_full[0, : img_in_shape[0], : img_in_shape[1], :] = data_img_output
        else:  # test
            if self.testinput_msk:
                data_img_input, _, _ = self.generate_InputGT(data_img_input, data_img_input_midframe)  # mask=0: 图像像素置为相邻的随机像素值
            input_full[0, : img_in_shape[0], : img_in_shape[1], :] = data_img_input
            
        return input_full, output_full

    def generate_InputGT(self, input, midframe):
        if self.Inputmask0:
            return self.generate_mask0(input, midframe)
        else:
            return self.generate_mask(input)
        
    def generate_mask0(self, input, midframe):  # mask=0: 图像像素置0
        ratio = 0.1  # 0.9  #
        size_data = input.shape  # [512,512,1]
        num_sample = int(size_data[0] * size_data[1] * (1 - ratio))
        mask = np.ones(size_data)
        output = input
        if self.GT0:
            gt = np.zeros([size_data[0], size_data[1], 1])
        else:
            gt = midframe
        
        for ich in range(size_data[2]):
            idy_msk = np.random.randint(0, size_data[0], num_sample)  # (26214,)
            idx_msk = np.random.randint(0, size_data[1], num_sample)
            id_msk = (idy_msk, idx_msk, ich)
            output[id_msk] = 0.0
            mask[id_msk] = 0.0
            if self.GT0 and (ich == self.pre_frame):
                id_msk_gt = (idy_msk, idx_msk, 0)
                gt[id_msk_gt] = midframe[id_msk_gt]
                assert self.Input_mid_frame0 == True
        return output, mask, gt

    def generate_mask(self, input):  # mask=0: 图像像素置为相邻的随机像素值
        ratio = 0.9
        size_window = (5, 5)
        size_data = input.shape  # [512,512,1]
        num_sample = int(size_data[0] * size_data[1] * (1 - ratio))
    
        mask = np.ones(size_data)
        output = input
        gt = np.zeros(input.shape)
    
        for ich in range(size_data[2]):
            idy_msk = np.random.randint(0, size_data[0], num_sample)  # (26214,)
            idx_msk = np.random.randint(0, size_data[1], num_sample)
        
            idy_neigh = np.random.randint(-size_window[0] // 2 + size_window[0] % 2,
                                          size_window[0] // 2 + size_window[0] % 2, num_sample)
            idx_neigh = np.random.randint(-size_window[1] // 2 + size_window[1] % 2,
                                          size_window[1] // 2 + size_window[1] % 2, num_sample)
        
            idy_msk_neigh = idy_msk + idy_neigh
            idx_msk_neigh = idx_msk + idx_neigh
        
            idy_msk_neigh = idy_msk_neigh + (idy_msk_neigh < 0) * size_data[0] - (idy_msk_neigh >= size_data[0]) * \
                            size_data[0]
            idx_msk_neigh = idx_msk_neigh + (idx_msk_neigh < 0) * size_data[1] - (idx_msk_neigh >= size_data[1]) * \
                            size_data[1]
        
            id_msk = (idy_msk, idx_msk, ich)
            id_msk_neigh = (idy_msk_neigh, idx_msk_neigh, ich)
        
            output[id_msk] = input[id_msk_neigh]
            mask[id_msk] = 0.0
            gt[id_msk] = input[id_msk]
    
        return output, mask, gt
