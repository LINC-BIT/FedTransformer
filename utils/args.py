import os


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    # parser.add_argument('--model_type', type=str, default='cnn', choices=["cnn", "vit"],
    #                     help='neural network used in training')

    parser.add_argument('--model', type=str, default='simplecnn', metavar='N',
                        choices=["resnet50", "efficient_b4", "cct", "levit", "regnety_4g", "simplecnn", "t2t"],
                        help='neural network used in training')

    parser.add_argument('--dataset_name', type=str, default='cifar10', metavar='N', choices=["cifar10", "fmnist"],
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='/data/zxj/dataset/cifar10', help='data directory')

    parser.add_argument('--partition', type=str, default='iid', metavar='N', choices=["iid", "split-1", "split-2"],
                        help='how to partition the dataset on local workers, split-1(4-classes), split-2(2-classes)')

    parser.add_argument('--lr_scheduler', type=str, default='none', metavar='N', choices=['cosine', 'none'],
                        help='lr scheduler')

    parser.add_argument('--balance', action='store_true', default=False, help="Whether use pretrained or not")

    parser.add_argument('--train_bs', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--img_size', type=int, default=32, metavar='N',
                        help='input image size (default: 32)')

    parser.add_argument('--in_channels', type=int, default=3, metavar='N',
                        help='input in channels (default: 3)')

    parser.add_argument('--client_optimizer', type=str, default='sgd',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.03, metavar='LR',
                        help='learning rate (default: 0.03)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.0)

    parser.add_argument('--epochs', type=int, default=1, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--client_num_in_total', type=int, default=10, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=10, metavar='NN',
                        help='number of workers')

    parser.add_argument('--max_round', type=int, default=100,
                        help='how many round of communications we shoud use')

    parser.add_argument('--num_classes', type=int, default=10,
                        help='how many classes for classification')

    parser.add_argument('--test_freq', type=int, default=1,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    # parser.add_argument('--num_shards', type=int, default=-1, metavar='N',
    #                     help='shards number when partition method is shards (default: -1)')

    # parser.add_argument('--diri_alpha', type=float, default=0.1, metavar='N',
    #                     help='alpha of dirichlet')

    # parser.add_argument('--warmup_step', type=int, default=-1, metavar='N',
    #                     help='shards number when partition method is shards (default: -1)')

    # parser.add_argument('--Pretrained', action='store_true', default=False, help="Whether use pretrained or not")

    return parser


def show_setting(args):
    message = ''
    message += "================ FL training with total model parameters: %2.1fM  ================\n" % (args.model_quantity)
    message += '++++++++++++++++ Other Train related parameters ++++++++++++++++ \n'

    for k, v in sorted(vars(args).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '++++++++++++++++  End of show parameters ++++++++++++++++ '
    print(message)
    return message


def save_setting(message, output_dir):
    setting_file = os.path.join(output_dir, 'fl_setting.txt')
    with open(setting_file, 'wt') as args_file:
        args_file.write(message)
        args_file.write('\n')

    # data distribution
    from .data_utils import NET_DATA_STATE
    content = ''
    for k, v in NET_DATA_STATE.items():
        content += f"client-{k} : {v} \n"
    data_dist_file = os.path.join(output_dir, 'net_data_state.txt')
    with open(data_dist_file, 'wt') as f:
        f.write(content)
        f.write('\n')
