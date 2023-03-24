import os
from deepinterpolation.generic import JsonSaver, ClassLoader
import datetime

now = datetime.datetime.now()
run_uid = now.strftime("%Y_%m_%d_%H_%M")
rootpath = os.getcwd()
generator_param = {}
inferrence_param = {}
steps_per_epoch = 10


generator_param["name"] = "FmriGeneratorS3"
inferrence_param["name"] = "fmriS3_inferrence"
NeedGT = False
generator_param["NeedGT"] = NeedGT
generator_param["type"] = "generator"
generator_param["Pad0"] = True
generator_param["InputF5"] = True
generator_param["gtchannel"] = 1  # output channel
generator_param["pre_post_t"] = 2
generator_param["pre_post_x"] = 3
generator_param["pre_post_y"] = 3
generator_param["pre_post_z"] = 3
generator_param["center_omission_size"] = 4
generator_param["train_path"] =\
rootpath + "/sample_data/ds001246-master/sub-02/ses-imageryTest02/func/sub-02_ses-imageryTest02_task-imagery_run-01_bold.nii.gz"
generator_param["smallIm"] = True
generator_param["end_frame"] = 100
generator_param["batch_size"] = 10
generator_param["total_nb_block"] = 10
generator_param["start_frame"] = 0
generator_param["steps_per_epoch"] = steps_per_epoch


inferrence_param["type"] = "inferrence"
inferrence_param["InputF5"] = generator_param["InputF5"]
inferrence_param["single_voxel_output_single"] = True  # allow in the future to change scanning mode for inference

jobdir = rootpath + "/Checkpoint/%s/" % generator_param["name"]
savedir = jobdir + 'fMRI'
os.makedirs(jobdir, exist_ok=True)
os.makedirs(savedir, exist_ok=True)
inferrence_param["model_path"] = jobdir + "fmri_unet_denoiser.h5"
inferrence_param["output_file"] = savedir + "/sub-02_ses-imageryTest02_task-imagery_run-01_bold_full.h5"


path_generator = os.path.join(jobdir, "test-generator.json")
json_obj = JsonSaver(generator_param)
json_obj.save_json(path_generator)

path_infer = os.path.join(jobdir, "inferrence.json")
json_obj = JsonSaver(inferrence_param)
json_obj.save_json(path_infer)

generator_obj = ClassLoader(path_generator)
data_generator = generator_obj.find_and_build()(path_generator)

# Those are parameters used for the network topology
inferrence_obj = ClassLoader(path_infer)
inferrence_class = inferrence_obj.find_and_build()(path_infer, data_generator, None)

inferrence_class.run()
