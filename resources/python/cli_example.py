from train import train
import argparse as ap

def main():
    """
    Function to make and train a model using train and argparse
    :return: None
    """
    parser = ap.ArgumentParser()
    parser.add_argument("-dp","--data_path", help="make model", type=str)
    parser.add_argument("-op", "--out_path", help="define output path", type=str)

    args = parser.parse_args()

    if args.data_path != args.out_path:
        tr = train(args.data_path, args.out_path)
        tr.model_prep()

        tr.train_and_validate(epochs=50)
