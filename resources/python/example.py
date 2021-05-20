from train_c import train

dataset_path = "../assets"
out_path = "/models"

# make an instance train and feed in optional the dataset and output paths
tr = train(dataset_path=dataset_path, model_output_path=out_path)

tr.model_prep()

