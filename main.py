import argparse

from utils.train import train
from utils.utils import seed_everything

def parser():
    parser = argparse.ArgumentParser(description='DACON ARTIST')
    parser.add_argument('--seed', type=int, default=1997)

    # model
    parser.add_argument('--model', type=str, default='convnext_xlarge_384_in22ft1k')
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--num_classes', type=int, default=50)
    parser.add_argument('--drop_rate', type=float, default=0.3)
    # dataloader
    parser.add_argument('--train_df', type=str, default='./train.csv')
    parser.add_argument('--valid_df', type=str, default='./test.csv')
    parser.add_argument('--num_workers', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=64)
    # train
    # parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=3e-4)


    args = parser.parse_args()

    return args



def main():
    parser()

    args = parser()

    seed_everything(args.seed)

    train(args=args)

if __name__ == '__main__':
    main()