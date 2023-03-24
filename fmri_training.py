import deepmodel as de
import os
from deepinterpolation.generic import JsonSaver, ClassLoader, JsonLoader
import datetime


now = datetime.datetime.now()
rootpath = os.getcwd()

training_param = {}
generator_param = {}
network_param = {}
generator_test_param = {}
generator_param_name = "FmriGeneratorS3"


steps_per_epoch = 6000

generator_test_param["type"] = "generator"
generator_test_param["name"] = generator_param_name
generator_test_param["Pad0"] = True
generator_test_param["InputF5"] = True
generator_test_param["gtchannel"] = 1  # output channel
generator_test_param["pre_post_t"] = 2
generator_test_param["pre_post_x"] = 3
generator_test_param["pre_post_y"] = 3
generator_test_param["pre_post_z"] = 3
generator_test_param['center_omission_size'] = 4
generator_test_param["train_path"] =\
rootpath + "/sample_data/ds001246-master/sub-01/ses-perceptionTest01/func/sub-01_ses-perceptionTest01_task-perception_run-01_bold.nii.gz"
generator_test_param["batch_size"] = 1000
generator_test_param["total_nb_block"] = 450000
generator_test_param["start_frame"] = 5
generator_test_param["end_frame"] = 160
generator_test_param["steps_per_epoch"] = steps_per_epoch
generator_test_param["single_voxel_output_single"] = False
local_train_path = rootpath + '/sample_data/ds001246-master/fmri_data/training'  # '/home/ec2-user/fmri_data/training'  # /fmri_imagenet_sub_{}_percept_{}_run_{}.nii.gz
train_paths = os.listdir(local_train_path)


generator_param_list = []
for indiv_path in train_paths:
    generator_param = {}
    generator_param["type"] = "generator"
    generator_param["name"] = generator_param_name
    generator_param["Pad0"] = generator_test_param["Pad0"]
    generator_param["InputF5"] = generator_test_param["InputF5"]
    generator_param["movingWindow"] = generator_test_param["movingWindow"]
    generator_param["N2Vmsk0"] = generator_test_param["N2Vmsk0"]
    generator_param["gtchannel"] = generator_test_param["gtchannel"]  # output channel
    generator_param["pre_post_t"] = generator_test_param["pre_post_t"]
    generator_param["pre_post_x"] = 3
    generator_param["pre_post_y"] = 3
    generator_param["pre_post_z"] = 3
    generator_param["train_path"] = os.path.join(local_train_path, indiv_path)
    generator_param["start_frame"] = 5
    generator_param["end_frame"] = 160
    generator_param["batch_size"] = 1000
    generator_param["total_nb_block"] = 9000000
    generator_param["steps_per_epoch"] = steps_per_epoch
    generator_param["center_omission_size"] = 4
    generator_param["single_voxel_output_single"] = False
    generator_param_list.append(generator_param)

network_param["type"] = "network"
network_param["name"] = "fmri_unet_denoiser"

training_param["type"] = "trainer"
training_param["name"] = "core_trainer"
training_param["batch_size"] = generator_test_param["batch_size"]
training_param["steps_per_epoch"] = steps_per_epoch
training_param["period_save"] = 5
training_param["nb_gpus"] = 1
training_param["apply_learning_decay"] = 1
training_param["initial_learning_rate"] = 0.001
training_param["epochs_drop"] = 50
training_param["nb_times_through_data"] = 1
training_param["learning_rate"] = 0.001
training_param["nb_workers"] = 0
training_param["use_multiprocessing"] = False
training_param["loss"] = "mean_absolute_error"
training_param["model_string"] = (network_param["name"] + "_" + training_param["loss"])  #
jobdir = rootpath + "/Checkpoint/%s/%s/" % (generator_param["name"], training_param["model_string"])


training_param["output_dir"] = jobdir
os.makedirs(jobdir, exist_ok=True)
training_param["model_path"] = jobdir + "fmri_unet_denoiser.h5"


path_training = os.path.join(jobdir, "training.json")
json_obj = JsonSaver(training_param)
json_obj.save_json(path_training)

list_train_generator = []
for local_index, indiv_generator in enumerate(generator_param_list):
    print('local_index: ', local_index)
    if local_index == 0:
        indiv_generator["initialize_list"] = 1
    else:
        indiv_generator["initialize_list"] = 0
    
    path_generator = os.path.join(jobdir, "generator" + str(local_index) + ".json")
    if os.path.exists(path_generator):
        json_obj = JsonLoader(path_generator)
        json_obj.load_json()
    else:
        json_obj = JsonSaver(indiv_generator)
        json_obj.save_json(path_generator)
    
    generator_obj = ClassLoader(path_generator)
    train_generator = generator_obj.find_and_build()(path_generator)

    # we don't need to set a random set of points for all 100 or so
    if local_index == 0:
        keep_generator = train_generator
    else:
        train_generator.x_list = keep_generator.x_list
        train_generator.y_list = keep_generator.y_list
        train_generator.z_list = keep_generator.z_list
        train_generator.t_list = keep_generator.t_list

    list_train_generator.append(train_generator)


path_test_generator = os.path.join(jobdir, "valid_generator.json")
json_obj = JsonSaver(generator_test_param)
json_obj.save_json(path_test_generator)

path_network = os.path.join(jobdir, "network.json")
json_obj = JsonSaver(network_param)
json_obj.save_json(path_network)

generator_obj = ClassLoader(path_generator)
generator_test_obj = ClassLoader(path_test_generator)

network_obj = ClassLoader(path_network)
trainer_obj = ClassLoader(path_training)

train_generator = generator_obj.find_and_build()(path_generator)

global_train_generator = de.generator_collection.CollectorGenerator(
    list_train_generator)

test_generator = generator_test_obj.find_and_build()(path_test_generator)

network_callback = network_obj.find_and_build()(path_network)

training_class = trainer_obj.find_and_build()(
    global_train_generator, test_generator, network_callback, path_training)

training_class.run()

training_class.finalize()
