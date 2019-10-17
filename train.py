import argparse
from network import Network

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', default='/data')
parser.add_argument('--checkpoint_dir', default='/data/model')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_splits', default=5)

opt = parser.parse_args()
print(opt)


if __name__ == '__main__':
    # CHD: 0-background, 1-LVM, 2-LV, 3-RV
    # HCMP 0-background, 1-LVM, 2-APM, 3-PPM

    dia_net = Network(opt, 'CHD', 'dia', class_weights=(0, 1/3, 1/3, 1/3), crop_range=(12,100,100,12), hu_range=(-1000,1000),
                      down_size=(208,256,256), n_epochs=40, loss_change=10, milestones=(20,30))
    dia_net.train()

    sys_net = Network(opt, 'CHD', 'sys', class_weights=(0, 1/3, 1/3, 1/3), crop_range=(12,100,100,12), hu_range=(-1000,1000),
                      down_size=(256,224,224), n_epochs=40, loss_change=10, milestones=(20,30))
    sys_net.train()

    a_net = Network(opt, 'HCMP', 'A', class_weights=(0, 1/2, 1/4, 1/4), crop_range=(60,72,102,30), hu_range=(-400,1000),
                    down_size=(208,256,256), n_epochs=60, loss_change=5, milestones=(20,40))
    a_net.train()

    n_net = Network(opt, 'HCMP', 'N', class_weights=(0, 1/2, 1/4, 1/4), crop_range=(86,86,160,12), hu_range=(-200,600),
                    down_size=(208,256,256), n_epochs=60, loss_change=15, milestones=(30,45))
    n_net.train()
