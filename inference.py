import argparse
from dataloader import get_loader
from network import InferNetwork

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', default='/data')
parser.add_argument('--checkpoint_dir', default='/data/model')
parser.add_argument('--output_dir', default='/data/output')

opt = parser.parse_args()


if __name__ == '__main__':
    dia_infernet = InferNetwork(opt, 'CHD', 'dia', crop_range=(12,100,100,12), hu_range=(-1000,1000), down_size=(208,256,256))
    dia_infernet.infer()

    sys_infernet = InferNetwork(opt, 'CHD', 'sys', crop_range=(12,100,100,12), hu_range=(-1000,1000), down_size=(256,224,224))
    sys_infernet.infer()

    a_infernet = InferNetwork(opt, 'HCMP', 'A', crop_range=(60,72,102,30), hu_range=(-400,1000), down_size=(208,256,256))
    a_infernet.infer()

    n_infernet = InferNetwork(opt, 'HCMP', 'N', crop_range=(86,86,160,12), hu_range=(-200,600), down_size=(208,256,256))
    n_infernet.infer()
