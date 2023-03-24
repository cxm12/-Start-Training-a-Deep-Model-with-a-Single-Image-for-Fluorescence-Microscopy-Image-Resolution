import keras.backend as K
import cv2
import os
import mlflow
import warnings
import h5py
import numpy as np
from deepinterpolation.generic import JsonLoader
from tensorflow.keras.models import load_model
import deepinterpolation.loss_collection as lc
from keras.models import Model
from keras.layers import Input
import matplotlib.pyplot as plt


def to_color(arr, pmin=1, pmax=99.8, gamma=1., colors=((0, 1, 0), (1, 0, 1), (0, 1, 1))):
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


class fmriS3_inferrence:
    # This inferrence is specific to fMRI which is raster scanning for denoising
    def __init__(self, inferrence_json_path, generator_obj, network_callback):
        self.inferrence_json_path = inferrence_json_path
        self.generator_obj = generator_obj
        self.network_obj = network_callback
        self.SNR = True
        local_json_loader = JsonLoader(inferrence_json_path)
        local_json_loader.load_json()
        self.json_data = local_json_loader.json_data
        self.output_file = self.json_data["output_file"]
        self.model_path = self.json_data["model_path"]

        # when output is a full volume to select only the center currently only set to true. Future implementation could make smarter
        # scanning of the volume and leverage more than just the center pixel
        if "single_voxel_output_single" in self.json_data.keys():
            self.single_voxel_output_single = self.json_data[
                "single_voxel_output_single"]
        else:
            self.single_voxel_output_single = True

        self.model_path = self.json_data["model_path"]
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            print('**************** Load Model ', self.model_path, '***************')
        else:
            self.local_generator = generator_obj
            local_size = self.local_generator.get_input_size()
            input_img = Input(shape=local_size)
            self.model = Model(input_img, self.network_obj(input_img))
            print('************************** No Model *************************')
        self.input_data_size = self.generator_obj.data_shape

    def run(self):
        chunk_size = list(self.generator_obj.data_shape)
        # Time is where we chunk the h5 file
        chunk_size[-1] = 1
        file_handlein = h5py.File(self.output_file[:-3] + '_Input.h5', "w")
        dset_in = file_handlein.create_dataset("data", shape=tuple(self.generator_obj.data_shape),
                                               chunks=tuple(chunk_size), dtype="float32", )
        gtmap = np.zeros(tuple(self.generator_obj.data_shape))
        with h5py.File(self.output_file, "w") as file_handle:
            dset_out = file_handle.create_dataset("data", shape=tuple(self.generator_obj.data_shape),
                                                  chunks=tuple(chunk_size), dtype="float32", )
            all_z_values = np.arange(0, self.input_data_size[2])
            all_y_values = np.arange(0, self.input_data_size[1])
            input_full = np.zeros([
                all_y_values.shape[0] * all_z_values.shape[0] * self.input_data_size[3],
                self.generator_obj.pre_post_x * 2 + 1,
                self.generator_obj.pre_post_y * 2 + 1,
                self.generator_obj.pre_post_z * 2 + 1,
                self.generator_obj.pre_post_t * 2 + 1, ], dtype="float32", )  # [340800, 7, 7, 7, 5]
            gt_full = np.zeros([
                all_y_values.shape[0] * all_z_values.shape[0] * self.input_data_size[3],
                self.generator_obj.pre_post_x * 2 + 1,
                self.generator_obj.pre_post_y * 2 + 1,
                self.generator_obj.pre_post_z * 2 + 1,
                1, ], dtype="float32", )  # [340800, 7, 7, 7, 1]
            # We are looping across the volume
            for local_x in np.arange(0, self.input_data_size[0]):
                print("x=" + str(local_x))
                for index_y, local_y in enumerate(all_y_values):  # print("y=" + str(local_y))
                    for index_z, local_z in enumerate(all_z_values):
                        for local_t in np.arange(0, self.input_data_size[3]):
                            (input_tmp, output_tmp,) = self.generator_obj.__data_generation__(local_x, local_y, local_z, local_t)
                            input_full[local_t + index_z * self.input_data_size[3] +
                                       index_y * self.input_data_size[3] * all_z_values.shape[0],
                            :, :, :, :, ] = input_tmp
                            gt_full[local_t + index_z * self.input_data_size[3] +
                                    index_y * self.input_data_size[3] * all_z_values.shape[0],
                            :, :, :, 0, ] = output_tmp[:, :, :, :, 0]  # self.generator_obj.pre_post_t]
                predictions_data = self.model.predict(input_full)  # gt_full  # [32000,7,7,7,3]
                # predictions_data = gt_full  # [32000,7,7,7,1]
                input = input_full * self.generator_obj.local_std + self.generator_obj.local_mean  # [340800, 7, 7, 7, 1]
                gt = gt_full * self.generator_obj.local_std + self.generator_obj.local_mean  # [340800, 7, 7, 7, 1]
                corrected_data = predictions_data * self.generator_obj.local_std + self.generator_obj.local_mean
                for index_y, local_y in enumerate(all_y_values):
                    for index_z, local_z in enumerate(all_z_values):
                        for local_t in np.arange(0, self.input_data_size[3]):
                            dset_out[local_x, local_y, local_z, local_t] = \
                                    corrected_data[local_t + index_z * self.input_data_size[3]
                                                   + index_y * self.input_data_size[3] * all_z_values.shape[0],
                                                   self.generator_obj.pre_post_x, self.generator_obj.pre_post_y,
                                                   self.generator_obj.pre_post_z, 0,]
                            dset_in[local_x, local_y, local_z, local_t] = \
                                input[local_t + index_z * self.input_data_size[3] + index_y * self.input_data_size[3] *
                                      all_z_values.shape[0],
                                      self.generator_obj.pre_post_x, self.generator_obj.pre_post_y,
                                      self.generator_obj.pre_post_z, self.generator_obj.pre_post_t,]  # [32, 32, 50, 213]
                            gtmap[local_x, local_y, local_z, local_t] = \
                                gt[local_t + index_z * self.input_data_size[3] +
                                   index_y * self.input_data_size[3] * all_z_values.shape[0],
                                   self.generator_obj.pre_post_x, self.generator_obj.pre_post_y,
                                   self.generator_obj.pre_post_z, 0,]
        
            for c in range(0, self.input_data_size[2], self.input_data_size[2] // 5):  # [64,64,50,175]
                for t in range(0, self.input_data_size[3], max(self.input_data_size[3] // 5, 1)):
            
                    try:
                        im = dset_out.value
                    except:
                        im = dset_out[()]
                    print(im.shape)
                    a = im[:, :, c, t]
                    cv2.imwrite(self.output_file[:-3]+'_out_c%dt%d.png' % (c, t), a)
                    a = dset_in.value[:, :, c, t]
                    cv2.imwrite(self.output_file[:-3] + '_in_c%dt%d.png' % (c, t), a)
                    a = gtmap[:, :, c, t]
                    cv2.imwrite(self.output_file[:-3] + '_gt_c%dt%d.png' % (c, t), a)
                    print('save to', self.output_file[:-3] + '_out_c%dt%d.png' % (c, t))
        
            if self.SNR:
                im = dset_out.value  # [64,64,50,175]
                mean = np.mean(im, axis=(0, 1, 2))  # [175,]
                std = np.std(im, axis=(0, 1, 2))  # [175,]
                temporalSNR = mean / std
                im1 = dset_in.value  # [64,64,50,175]
                mean1 = np.mean(im1, axis=(0, 1, 2))  # [175,]
                std1 = np.std(im1, axis=(0, 1, 2))  # [175,]
                temporalSNR1 = mean1 / std1
                print('temporalSNR = ', temporalSNR, 'Input temporalSNR = ', temporalSNR1, '\n Mean temporalSNR = ',
                      np.mean(temporalSNR), '\n Mean Input temporalSNR = ', np.mean(temporalSNR1))
                with open(self.output_file[:-3] + 'TemporalSRN.txt', 'w') as f:
                    f.write('temporalSNR = ' + str(temporalSNR))
                    f.write('\n Mean temporalSNR = ' + str(np.mean(temporalSNR)))
                    f.write('\n Input temporalSNR = ' + str(temporalSNR1))
                    f.write('\n Mean Input temporalSNR = ' + str(np.mean(temporalSNR1)))


def size(model):  # Compute number of params in a model (the actual number of floats)
    return sum([np.prod(K.get_value(w).shape) for w in model.trainable_weights])


class core_inferrence:
    # This is the generic inferrence class
    def __init__(self, inferrence_json_path, generator_obj, network_obj):
        self.inferrence_json_path = inferrence_json_path
        self.generator_obj = generator_obj
        self.network_obj = network_obj

        local_json_loader = JsonLoader(inferrence_json_path)
        local_json_loader.load_json()
        self.json_data = local_json_loader.json_data

        self.output_file = self.json_data["output_file"]

        if "save_raw" in self.json_data.keys():
            self.save_raw = self.json_data["save_raw"]
        else:
            self.save_raw = False

        if "rescale" in self.json_data.keys():
            self.rescale = self.json_data["rescale"]
        else:
            self.rescale = True

        self.batch_size = self.generator_obj.batch_size
        self.nb_datasets = len(self.generator_obj)  # = self.batchnum
        self.indiv_shape = self.generator_obj.get_output_size()  #
        self.savename = self.json_data["savename"]
        self.SNR = True

        self.__load_model()
        # print(size(self.model))
        # exit()

    def __load_model(self):
        local_model_path = self.__get_local_model_path()
        if os.path.exists(local_model_path):
            self.__load_local_model(path=local_model_path)
            print('****** Load Model', local_model_path, '********')
        else:
            print('****** Not Load Model ********')
            print('local_model_path = ', local_model_path)
            # self.__load_model_from_mlflow()
            local_size = self.generator_obj.get_input_size()
            input_img = Input(shape=local_size)
            self.model = Model(input_img, self.network_obj(input_img))

    def __get_local_model_path(self):
        try:
            model_path = self.json_data['model_path']
            warnings.warn('Loading model from model_path will be deprecated '
                          'in a future release')
        except KeyError:
            model_path = self.json_data['model_source']['local_path']
        return model_path

    def __load_local_model(self, path: str):
        self.model = load_model(
            path,
            custom_objects={
                "msk_mean_squareroot_error": lc.loss_selector("msk_mean_squareroot_error")},
                # "annealed_loss": lc.loss_selector("annealed_loss")},
        )

    def __load_model_from_mlflow(self):

        mlflow_registry_params = \
            self.json_data['model_source']['mlflow_registry']

        model_name = mlflow_registry_params['model_name']
        model_version = mlflow_registry_params.get('model_version')
        model_stage = mlflow_registry_params.get('model_stage')

        mlflow.set_tracking_uri(mlflow_registry_params['tracking_uri'])

        if model_version is not None:
            model_uri = f"models:/{model_name}/{model_version}"
        elif model_stage:
            model_uri = f"models:/{model_name}/{model_stage}"
        else:
            # Gets the latest version without any stage
            model_uri = f"models:/{model_name}/None"

        self.model = mlflow.keras.load_model(
            model_uri=model_uri
        )
    
    def run_Ophy(self):
        self.NeedGT = self.json_data["NeedGT"]
        self.Pad0 = self.json_data["Pad0"]
        chunk_size = [1]
        # self.nb_datasets = 2
        final_shape = [self.nb_datasets * self.batch_size]
        (h, w, c) = self.indiv_shape
        final_shape.extend((h, w, 1))  # [80, 512, 512,1]
        chunk_size.extend((h, w, 1))
        gtmap = np.zeros(tuple(final_shape))
        Inputmap = np.zeros(tuple(final_shape))
        with h5py.File(self.output_file, "w") as file_handle:
            dset_out = file_handle.create_dataset("data", shape=tuple(final_shape),
                              chunks=tuple(chunk_size), dtype="float32", )  # [80, 512, 512,1]
        
            if self.save_raw:
                raw_out = file_handle.create_dataset("raw",
                    shape=tuple(final_shape),
                    chunks=tuple(chunk_size),
                    dtype="float32",)
        
            for index_dataset in np.arange(0, self.nb_datasets, 1):
                local_data = self.generator_obj.__getitem__(index_dataset)
                # predictions_data = local_data[1]
                gtdata = local_data[1]
                indata = local_data[0]
                # predictions_data = gtdata
                predictions_data = self.model.predict(local_data[0])
                local_mean, local_std = self.generator_obj.__get_norm_parameters__(index_dataset)
            
                if self.rescale:
                    corrected_data = predictions_data * local_std + local_mean
                    gtdata = gtdata * local_std + local_mean
                    indata = indata * local_std + local_mean
                else:
                    corrected_data = predictions_data
                if corrected_data.shape[-1] != 1:
                    corrected_data = corrected_data[:, :, :, corrected_data.shape[-1]//2:corrected_data.shape[-1]//2+1]
                
                if ("NB2NB" in self.json_data["model_path"]) and (not self.NeedGT) and (not self.Pad0):  # #
                    print('| depth to space |')
                    n, h, w, c = corrected_data.shape  # [B*4,512/2,512/2,1]
                    corrected_data = np.squeeze(corrected_data)
                    unfolded_x = corrected_data.reshape(n // 4, 2, 2, h, w, )
                    z = np.transpose(unfolded_x, (0, 3, 1, 4, 2))  # n // 4, h, 2, w, 2
                    z = z.reshape(n // 4, 2 * h, 2 * w)  # [n // 4, 2h, 2w]
                    corrected_data = np.expand_dims(z, -1)

                    n, h, w, c = indata.shape  # [B*4,512/2,512/2,20]
                    unfolded_xin = indata.reshape(n // 4, 2, 2, h, w, c)
                    zin = np.transpose(unfolded_xin, (0, 5, 1, 2, 3, 4))  # n // 4, c, 2, 2, h, w
                    zin = np.transpose(zin, (0, 1, 4, 2, 5, 3))  # n // 4, c, h, 2, w, 2
                    zin = zin.reshape([n // 4, c, 2 * h, 2 * w])  # [n // 4, c, 2h, 2w]
                    indata = np.transpose(zin, (0, 2, 3, 1))  # n // 4, h*2, w*2, c
                local_size = corrected_data.shape[0]

                if self.save_raw:
                    if self.rescale:
                        corrected_raw = local_data[1] * local_std + local_mean
                    else:
                        corrected_raw = local_data[1]
                    if ("NB2NB" in self.json_data["model_path"]) and (not self.NeedGT) and (not self.Pad0):  #depth to space
                        n, h, w, c = corrected_raw.shape  # [B*4,512/2,512/2,1]
                        corrected_raw = np.squeeze(corrected_raw)
                        unfolded_x = corrected_raw.reshape(n // 4, 2, 2, h, w, )
                        z = np.transpose(unfolded_x, (0, 3, 1, 4, 2))  # n // 4, h, 2, w, 2
                        z = z.reshape(n // 4, 2 * h, 2 * w)  # [n // 4, 2h, 2w]
                        corrected_raw = np.expand_dims(z, -1)
                    raw_out[index_dataset * self.batch_size:index_dataset * self.batch_size + local_size, :,] = corrected_raw
            
                start = index_dataset * self.batch_size
                end = index_dataset * self.batch_size + local_size
                dset_out[start:end, :] = corrected_data
                if gtdata.shape[-1] != 1:
                    gtdata = gtdata[:, :, :, gtdata.shape[-1] // 2:gtdata.shape[-1] // 2 + 1]
                gtmap[start:end, :] = gtdata
                Inputmap[start:end, :] = indata[:, :, :, self.generator_obj.pre_frame:self.generator_obj.pre_frame+1]
                
            for c in range(0, final_shape[0], 1):  # [80, 512,512,1]
            # for c in range(0, 20, 5):  # [80, 512,512,1]
                im = dset_out.value[c, :, :, 0]
                cv2.imwrite(self.savename + '_B%d.png' % c, im)
                # cv2.imwrite(self.savename + 'Gt_B%d.png' % c, gtmap[c, :, :, 0])
                # cv2.imwrite(self.savename + 'In_B%d.png' % c, Inputmap[c, :, :, 0])
                print('save to', self.savename + '_B%d.png' % c)
                
            if self.SNR:
                im = dset_out.value  # [64,64,50,175]
                mean = np.mean(im, axis=(1, 2))  # [175,]
                std = np.std(im, axis=(1, 2))  # [175,]
                temporalSNR = mean / std
                im1 = Inputmap  # [64,64,50,175]
                mean1 = np.mean(im1, axis=(1, 2))  # [175,]
                std1 = np.std(im1, axis=(1, 2))  # [175,]
                temporalSNR1 = mean1 / std1
                print('temporalSNR = ', temporalSNR, 'Input temporalSNR = ', temporalSNR1, '\n Mean temporalSNR = ',
                      np.mean(temporalSNR), '\n Mean Input temporalSNR = ', np.mean(temporalSNR1))
                with open(self.output_file[:-3] + 'TemporalSRN.txt', 'w') as f:
                    f.write('temporalSNR = ' + str(temporalSNR))
                    f.write('\n Mean temporalSNR = ' + str(np.mean(temporalSNR)))
                    f.write('\n Input temporalSNR = ' + str(temporalSNR1))
                    f.write('\n Mean Input temporalSNR = ' + str(np.mean(temporalSNR1)))
