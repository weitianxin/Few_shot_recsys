import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='Training HiCE on WikiText-103')

    '''
        Dataset arguments
    '''
    parser.add_argument('--w2v_dir', type=str, default='./data/base_w2v/wiki_all.sent.split.model',
                        help='location of the default node embedding')
    parser.add_argument('--data_dir', type=str, default='./data/yelp_dataset/yelp_dataset',
                        help='location of the training data')
    parser.add_argument('--freq_lbound', type=int, default=16,
                        help='Lower bound of word frequency in w2v for selecting target words')
    parser.add_argument('--freq_ubound', type=int, default=2 ** 16,
                        help='Upper bound of word frequency in w2v for selecting target words')
    parser.add_argument('--cxt_lbound', type=int, default=2,
                        help='Lower bound of word frequency in corpus for selecting target words')
    parser.add_argument('--chimera_dir', type=str, default='./data/chimeras/',
                        help='location of the testing corpus (Chimeras)')
    parser.add_argument('--cuda', type=str, default='0',
                        help='Avaiable GPU ID')
    '''
        Model hyperparameters
    '''
    parser.add_argument('--inner_batch', type=int, default=4,
                        help='inner batch of a task')
    parser.add_argument('--embed_dim',type=int,default=64,
                        help='ID embedding dim')
    parser.add_argument('--input_dim', type=int, default=64,
                        help='ID embedding dim')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='ID embedding dim')
    parser.add_argument('--output_dim', type=int, default=64,
                        help='ID embedding dim')

    parser.add_argument('--n_head', type=int, default=10,
                        help='number of hidden units per layer')
    parser.add_argument('--n_layer', type=int, default=2,
                        help='number of gcn layers')
    parser.add_argument('--n_epoch', type=int, default=100,
                        help='upper bound of training epochs')
    parser.add_argument('--n_batch', type=int, default=256,
                        help='batch size')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size')
    parser.add_argument('--lr_init', type=float, default=1e-3,
                        help='initial learning rate for Adam')
    parser.add_argument('--n_shot', type=int, default=6,
                        help='upper bound of training K-shot')
    '''
        Validation & Test arguments
    '''
    parser.add_argument('--test_interval', type=int, default=1,
                        help='report interval')
    parser.add_argument('--save_dir', type=str, default='./save/',
                        help='location for saving the best model')
    parser.add_argument('--lr_decay', type=float, default=0.5,
                        help='Learning Rate Decay using ReduceLROnPlateau Scheduler')
    parser.add_argument('--threshold', type=float, default=1e-3,
                        help='Learning Rate Decay using ReduceLROnPlateau Scheduler')
    parser.add_argument('--patience', type=int, default=4,
                        help='Patience for lr Scheduler judgement')
    parser.add_argument('--lr_early_stop', type=float, default=1e-5,
                        help='the lower bound of training lr. Early stop after lr is below it.')


    '''
        Adaptation with First-Order few_shot_recsys arguments
    '''
    parser.add_argument('--adapt', action='store_true',
                        help='adapt to target dataset with 1-st order few_shot_recsys')
    parser.add_argument('--inner_batch_size', type=int, default=4,
                        help='batch for updating using source corpus')
    parser.add_argument('--meta_batch_size', type=int, default=16,
                        help='batch for accumulating meta gradients')

    args = parser.parse_args()
    return args
