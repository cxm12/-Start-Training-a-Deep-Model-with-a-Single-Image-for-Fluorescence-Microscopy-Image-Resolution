import os
from deepinterpolation.generic import JsonSaver, ClassLoader
import datetime
import pathlib


# This is used for record-keeping

# Initialize meta-parameters objects
training_param = {}
generator_param = {}
network_param = {}
generator_test_param = {}

# An epoch is defined as the number of batches pulled from the dataset. Because our datasets are VERY large. Often, we cannot
# go through the entirity of the data so we define an epoch slightly differently than is usual.
steps_per_epoch = 5

# Those are parameters used for the Validation test generator. Here the test is done on the beginning of the data but
# this can be a separate file
generator_test_param["name"] = "SingleTifGeneratorS1"
generator_test_param["pre_post_frame"] = 10  # Number of frame provided before and after the predicted frame
generator_test_param["InputF20"] = True
generator_test_param["batch_size"] = 2

# generator_test_param["name"] = "SingleTifGeneratorS2"
# generator_test_param["pre_post_frame"] = 5
# generator_test_param["InputF20"] = True
# generator_test_param["Pad0"] = True  #
# generator_test_param["batch_size"] = 1

generator_test_param["start_frame"] = 0
generator_test_param["end_frame"] = 199
generator_test_param["pre_post_omission"] = 1  # Number of frame omitted before and after the predicted frame
generator_test_param["NeedGT"] = True
generator_test_param["steps_per_epoch"] = -1  # No step necessary for testing as epochs are not relevant. -1 deactivate it.
generator_test_param["train_path"] = os.path.join(pathlib.Path(__file__).parent.absolute(),
    "./sample_data/ophys_tiny_761605196.tif",)

# Those are parameters used for the main data generator
generator_param["type"] = "generator"
generator_param["stype"] = "generator"
generator_param["steps_per_epoch"] = steps_per_epoch
generator_param["name"] = generator_test_param["name"]
generator_param["pre_post_frame"] = generator_test_param["pre_post_frame"]
generator_param["InputF20"] = generator_test_param["InputF20"]
generator_param["Pad0"] = generator_test_param["Pad0"]
generator_param["NeedGT"] = generator_test_param["NeedGT"]
generator_param["train_path"] = os.path.join(pathlib.Path(__file__).parent.absolute(), "./sample_data/ophys_tiny_761605196.tif",)
generator_param["batch_size"] = generator_test_param["batch_size"]
generator_param["start_frame"] = generator_test_param["start_frame"]
generator_param["end_frame"] = generator_test_param["end_frame"]
generator_param["pre_post_omission"] = generator_test_param["pre_post_omission"]

# Those are parameters used for the network topology
network_param["type"] = "network"
network_param["name"] = "unet_single_1024"

# Those are parameters used for the training process
training_param["type"] = "trainer"
training_param["name"] = "core_trainer"
training_param["batch_size"] = generator_test_param["batch_size"]
training_param["steps_per_epoch"] = steps_per_epoch
training_param["period_save"] = 25  # network model is potentially saved during training between a regular nb epochs
training_param["nb_gpus"] = 1
training_param["apply_learning_decay"] = 0
training_param["nb_times_through_data"] = 1  # epoch number: if you want to cycle through the entire data. Two many iterations will cause noise overfitting
training_param["learning_rate"] = 0.0001
training_param["pre_post_frame"] = generator_test_param["pre_post_frame"]
training_param["loss"] = "msk_mean_squareroot_error"  # "mean_absolute_error"  #
training_param["nb_workers"] = 0  # this is to enable multiple threads for data generator loading. Useful when this is slower than training
training_param["model_string"] = (network_param["name"] + "_" + training_param["loss"])

# Where do you store ongoing training progress
rootp = os.getcwd()
jobdir = rootp + "/Checkpoint/%s/" % (generator_param["name"])
training_param["output_dir"] = jobdir
os.makedirs(jobdir, exist_ok=True)


# Here we create all json files that are fed to the training. This is used for recording purposes as well as input to the
# training process
path_training = os.path.join(jobdir, "training.json")
json_obj = JsonSaver(training_param)
json_obj.save_json(path_training)

path_generator = os.path.join(jobdir, "generator.json")
json_obj = JsonSaver(generator_param)
json_obj.save_json(path_generator)

path_test_generator = os.path.join(jobdir, "test_generator.json")
json_obj = JsonSaver(generator_test_param)
json_obj.save_json(path_test_generator)

path_network = os.path.join(jobdir, "network.json")
json_obj = JsonSaver(network_param)
json_obj.save_json(path_network)

# We find the generator obj in the collection using the json file
generator_obj = ClassLoader(path_generator)
generator_test_obj = ClassLoader(path_test_generator)

# We find the network obj in the collection using the json file
network_obj = ClassLoader(path_network)

# We find the training obj in the collection using the json file
trainer_obj = ClassLoader(path_training)

# We build the generators object. This will, among other things, calculate normalizing parameters.
train_generator = generator_obj.find_and_build()(path_generator)
test_generator = generator_test_obj.find_and_build()(path_test_generator)

# We build the network object. This will, among other things, calculate normalizing parameters.
network_callback = network_obj.find_and_build()(path_network)

# We build the training object.
training_class = trainer_obj.find_and_build()(
    train_generator, test_generator, network_callback, path_training)

# Start training. This can take very long time.
training_class.run()

# Finalize and save output of the training.
training_class.finalize()
