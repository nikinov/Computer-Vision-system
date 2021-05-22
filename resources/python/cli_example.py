from train import train
import argparse as ap

def main():
    """
    Function to make and train a model using train and argparse
    :return: None
    """
    parser = ap.ArgumentParser()
    parser.add_argument("dp", help="make model", type=str)
    parser.add_argument("op", help="define output path", type=str)

    args = parser.parse_args()

    if args.dp != args.op or args.dp == "def":
        if (args.dp and args.op) == "def":
            tr = train()
        else:
            tr = train(args.dp, args.op)
        tr.model_prep()
        tr.train_and_validate(epochs=5)

main()