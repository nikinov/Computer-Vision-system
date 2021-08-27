import argparse as ap
from ..number_train_resnet import train
from ..data_loading.data_save import csv_save

def main():
    """
    Function to make and train a model using train and argparse
    :return: None
    """
    parser = ap.ArgumentParser()
    parser.add_argument("ds", help="root dataset directory", type=str)
    parser.add_argument("mp", help="model path", type=str)
    parser.add_argument("mn", help="model name", type=str)
    parser.add_argument("sc", help="create csv file", type=bool)

    args = parser.parse_args()

    tr = train(model_name=args.mn)
    tr.prep(dataset_path=args.ds, model_output_path=args.mp)
    tr.train(save_type="jit")
    if args.sc:
        csv_save(file_save_name="config", data_dir=args.ds)




"""
    if args.dp != args.op or args.dp == "def":
        if (args.dp and args.op) == "def":
            tr = train(save_config=args.sc, use_config=args.uc)
        else:
            tr = train(args.dp, args.op)
        tr.model_prep()
        print("Started training")
        if args.ep == 0:
            tr.train_and_validate(epochs=25)
        else:
            tr.train_and_validate(epochs=args.epoch)
        print("Finished training")
"""