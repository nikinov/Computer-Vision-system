from train import train
import argparse as ap

def make_model():
    tr = train()



def main():
    parser = ap.ArgumentParser()
    parser.add_argument("data_path", help="make model", type=str)
    parser.add_argument("out_path", help="define output path", type=str)


