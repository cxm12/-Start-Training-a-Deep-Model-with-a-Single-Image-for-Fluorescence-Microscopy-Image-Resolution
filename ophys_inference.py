import os
from deepmodel.generic import JsonSaver, ClassLoader
import pathlib


generator_param = {}
inferrence_param = {}

# We are reusing the data generator for training here.
generator_param["NeedGT"] = False
generator_param["name"] = "SingleTifGeneratorS2"
generator_param["Pad0"] = True
generator_param["pre_post_frame"] = 10

# generator_param["name"] = "SingleTifGeneratorS1"
# generator_param["pre_post_frame"] = 5
# generator_param["Pad0"] = False


generator_param["InputF20"] = True
generator_param["pre_post_omission"] = 0
generator_param["steps_per_epoch"] = -1  # No steps necessary for inference as epochs are not relevant. -1 deactivate it.

generator_param["train_path"] = os.path.join(pathlib.Path(__file__).parent.absolute(),
    "./sample_data/ophys_tiny_761605196.tif",)

generator_param["batch_size"] = 1
generator_param["start_frame"] = 20
generator_param["end_frame"] = 99
generator_param["randomize"] = 0  # This is important to keep the order and avoid the randomization used during training


inferrence_param["type"] = "inferrence"
inferrence_param["name"] = "core_inferrence"
inferrence_param["NeedGT"] = False
inferrence_param["Pad0"] = generator_param["Pad0"]

methodname = '%s' % (generator_param["name"])
network = 'unet_single_1024'
loss = 'mean_absolute_error'
jobdir = "./Checkpoint/%s/%s_%s/" % (methodname, network, loss)
os.makedirs(jobdir, exist_ok=True)
inferrence_param["model_path"] = jobdir + "%s_%s_model.h5" % (network, loss)
inferrence_param["output_file"] = jobdir + "/ophys_tiny_761605196.h5"
inferrence_param["savename"] = inferrence_param["output_file"][:-3]


path_generator = os.path.join(jobdir, "generator.json")
json_obj = JsonSaver(generator_param)
json_obj.save_json(path_generator)

path_infer = os.path.join(jobdir, "inferrence.json")
json_obj = JsonSaver(inferrence_param)
json_obj.save_json(path_infer)

generator_obj = ClassLoader(path_generator)
data_generator = generator_obj.find_and_build()(path_generator)

##
path_network = os.path.join(jobdir, "network.json")
network_param = {}
network_param["type"] = "network"
network_param["name"] = "unet_single_1024"
json_obj = JsonSaver(network_param)
json_obj.save_json(path_network)
network_obj = ClassLoader(path_network)
network_callback = network_obj.find_and_build()(path_network)
##
inferrence_obj = ClassLoader(path_infer)
inferrence_class = inferrence_obj.find_and_build()(path_infer, data_generator, network_callback)

# Except this to be slow on a laptop without GPU. Inference needs parallelization to be effective.
inferrence_class.run_Ophy()
