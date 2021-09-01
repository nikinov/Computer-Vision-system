from train import AI
from utils import primary
from networks.pt_efficient_net import PtEfficientNet


# define my parameters
data_dir = "../../../resources/Datasets/NumberData"
model_output_path = "../../../resources/models/"
model_name = "my_favourite_model"
save_type = "jit_trace"

# define train object
tr = AI(model=PtEfficientNet())

# prepare for the training
tr.prep(dataset_path=data_dir, model_output_path=model_output_path)

# train our model
model = tr.train()

# save the model
primary.save_model(model, my_type="jit_trace", output_path=model_output_path, model_name=model_name)


# define my parameters
data_dir = "../../TestAssets"
model_path = "models/my_favourite_model.pt"
model_type = "jit"

# load my model
model = primary.load_model(model_path, my_type=model_type)

# define my AI object with a model the type of efficient_net
ai = AI(model=PtEfficientNet(model=model))

# test our model a folder adn get the number of correct and incorrect images
num_correct, num_incorrect = ai.predict_folder(folder_path=data_dir)
print("number of correctly predicted images: " + num_correct)
print("number of incorrectly predicted images: " + num_incorrect)
