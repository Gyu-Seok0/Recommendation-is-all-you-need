import argparse
import os

def create_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data",
                        type = str,
                        default = os.path.join('./preprocessed',"cand_ml-100k.pickle"),
                        help = "File path for data")

    parser.add_argument("--seed",
                        type = int,
                        default = 0,
                        help = "Seed (For reproducability")

    parser.add_argument("--num_ns",
                        type = int,
                        default = 1,
                        help = "The number of negative samples")

    parser.add_argument("--batch_size",
                        type = int,
                        default = 1024,
                        help = "The number of dataset in one batch")

    parser.add_argument("--model",
                        type = str,
                        default = "BPR")

    parser.add_argument("--dim",
                        type = int,
                        default = 60,
                        help = "Dimension for embedding")

    parser.add_argument('--lr',
                        type = float,
                        default = 0.001,
                        help = 'learning rate')

    parser.add_argument('--reg',
                        type = float,
                        default = 0.0001,
                        help = 'for L2 regularization')

    parser.add_argument("--max_epoch",
                        type = int,
                        default = 1000)

    parser.add_argument("--early_stop",
                        type = int,
                        default = 20)

    parser.add_argument("--es_epoch",
                        type = int,
                        default = 0,
                        help = "evaluation start epoch")
    
    parser.add_argument("--save",
                        type = int,
                        default = 0,
                        help = '0:false, 1:true') 
    
    args = parser.parse_args()
    
    return args
