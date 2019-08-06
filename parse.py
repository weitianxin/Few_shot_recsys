import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='Training HiCE on WikiText-103')

    '''
        Dataset arguments
    '''
    parser.add_argument('--data_dir', type=str, default='./data/yelp_dataset/yelp_dataset',
                        help='location of the training data')
    parser.add_argument('--cuda', type=str, default='3',
                        help='Avaiable GPU ID')
    parser.add_argument('--test_val_file', type=str, default='test_val/test_val.txt',
                        help='test val performance')
    '''
        Model hyperparameters
    '''

    parser.add_argument('--model', type=str, default="maml",
                        help='choose one of the models: maml,gnn,gnn_few_shot')

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
    parser.add_argument('--n_batch', type=int, default=64,
                        help='number of batches in a epoch')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='meta batch size:number of tasks in a batch')
    parser.add_argument('--task_size', type=int, default=1,
                        help='task size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate for Adam')
    parser.add_argument('--inner_lr', type=float, default=1e-3,
                        help='initial learning rate for Adam')
    parser.add_argument('--n_shot', type=int, default=6,
                        help='upper bound of training K-shot')

    '''
        Validation & Test arguments
    '''
    parser.add_argument('--early_stop_epoches', type=int, default=5,
                        help='early_stop_epoches')
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

    args = parser.parse_args()
    return args
