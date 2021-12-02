from argparse import ArgumentParser

def make_args(input = None):
    parser = ArgumentParser()
    
    parser.add_argument('--comment', dest='comment', default='', type=str,
                        help='comment')

    parser.add_argument('--dataset', dest='dataset', default='dblp', type=str,
                        help='name of the dataset')

    parser.add_argument('--gpu', dest='gpu', action='store_true',
                        help='whether use gpu')

    parser.add_argument('--cpu', dest='gpu', action='store_false',
                        help='whether use cpu')

    parser.add_argument('--cuda', dest='cuda', default='0', type=str)

    parser.add_argument('--res_path', dest='res_path', default='result/res.csv', type=str)

    parser.add_argument('--model_dir', dest='model_dir', default='saved_model', type=str)

    parser.add_argument('--ps', dest='ps', default='', type=str)

    parser.add_argument('--lr', type=float, default=1e-2,help='learning rate')

    parser.add_argument('--sigma', type=float, default=0.5,help='sigma of tf loss (default 1)')

    parser.add_argument('--beta', type=float, default=1e-1,help='beta of tf loss (default 1e-1)')

    parser.add_argument('--gamma', type=float, default=1e-1,help='gamma of sim loss (default 1e-1)')

    parser.add_argument('--alpha', type=float, default=0.95,help='alpha of message passing')

    parser.add_argument('--drop_out', type=float, default=0.5, help = 'dropout rate')

    parser.add_argument('--eval_time', type=int, default=1,help='epochs for evaluation')

    parser.add_argument('--layer_num', type=int, default=2, help='The number of layers')

    parser.add_argument('--conv_num', type=int, default=3, help='The number of conv layers')

    parser.add_argument('--hid_d', type=int, default=512, help='The dimension of hidden layers')

    parser.add_argument('--d', type=int, default=64, help='The dimension of output')

    parser.add_argument('--lam', type=float, default=0.5,help='The initial value of lambda')  

    parser.add_argument('--m', type=float, default=0.,help='The initial value of m')  

    parser.add_argument('--topk', type=int, default=100, help='The initial value of topk')

    parser.add_argument('--re', type=int, dest='he', default=10, help='The initial value of hidden edge percentage')

    parser.add_argument('--rw', type=int, dest='hw', default=0, help='The initial value of hidden keywords percentage')

    parser.add_argument('--kw', type=int, default=2, help='The initial value of kw_num')

    parser.add_argument('--repeat', type=int, default=10, help='The times of repeat')    

    parser.add_argument('--epochs_num', dest='epochs_num', default=500, type=int,
                        help='Number of epochs')

    parser.set_defaults(

                dataset= 'citeseer', 

                alpha = 0.95,

                beta = 1,

                sigma = 0.5,

                gamma = 1,

                topk = 100,

                kw = 3,

                he = 0,

                hw = 30,

                pca_verbal = True,
        )

    if input is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input)

    # if args.saved_model:

    return args

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
