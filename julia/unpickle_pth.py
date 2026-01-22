import argparse
import torch
import json
import numpy as np


def unpickle(path):
    model = torch.load(path)
    #print(model)

    with open(path.replace('.pth', '.txt'), "w") as f:
        f.write(str(model))
        #json.dump(model, f, ensure_ascii=False, indent=2)
    print("Converted pickle file to txt and saved as:", path.replace('.pth', '.txt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Unpickle a pth file and save it as a json')
    parser.add_argument('pth_file', type=str, help='path and name of pth file')
    args = parser.parse_args()

    unpickle(args.pth_file)