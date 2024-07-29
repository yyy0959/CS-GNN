def parser_add_main_args(parser):
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--sub_dataset', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--runs', type=int, default=5,
                        help='number of distinct runs')
    parser.add_argument('--directed', action='store_true',
                        help='set to not symmetrize adjacency')
    parser.add_argument('--train_prop', type=float, default=1.0,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')
    parser.add_argument('--protocol', type=str, default='semi',
                        help='protocol for cora datasets, semi or supervised')
    parser.add_argument('--rand_split', action='store_true', help='use random splits')
    parser.add_argument('--rand_split_class', action='store_true',
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--tr_num_per_class', type=int, default=20, help='training nodes randomly selected')
    parser.add_argument('--val_num_per_class', type=int, default=30, help='valid nodes randomly selected')
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc', 'f1'],
                        help='evaluation metric')

    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--print_prop', action='store_true',
                        help='print proportions of predicted class')

    # -------------------------------------------------------------------------------------            
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2, help='number of feed-forward layers')
    parser.add_argument('--nparts', type=int, default=8)
    parser.add_argument('--ego_hop', type=int, default=2)

    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.2)

    parser.add_argument('--method', default='gcn')
    parser.add_argument('--trans', action='store_true')
    parser.add_argument('--induc', action='store_true')
    parser.add_argument('--downstream_num', type=int, default=0, help='number of downstream tasks')
    # -------------------------------------------------------------------------------------            