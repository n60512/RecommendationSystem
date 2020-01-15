import argparse
import json
import os
import warnings
import datetime

class GatherOptions():
    def __init__(self):
        parser = argparse.ArgumentParser(description="train or test CAGAN-v2")
        
        parser.add_argument("--mode", default="train", choices=["train", "test", 'both', 'showAttn'],
                            help="train or test the model" )
        
        current_time = datetime.datetime.now()
        parser.add_argument("--save_dir", default=("HNAE/log/origin/{:%Y%m%d_%H_%M}".format(current_time)), help="path for saving model")
        # parser.add_argument("--model_dir", help="path to load model for test(the largest step or use --step to specify)")
        
        parser.add_argument('--sqlfile', default='', help="loacl sql cmd file")

        parser.add_argument('--save_model_freq', type=int, default=1, help="frequency of saving model")

        parser.add_argument("--having_interactions", type=int, default=15, help="num of user interactions")        
        parser.add_argument("--epoch", type=int, default=30, help="num of eopch for training")        
        parser.add_argument('--num_of_reviews', type=int, default=4, help="number of every user's reviews")
        parser.add_argument("--batchsize", type=int, default=40, help="input batch size")
        parser.add_argument("--num_of_rating", type=int, default=3, help="number of rating")
        parser.add_argument("--num_of_validate", type=int, default=3, help="number of validate")
        parser.add_argument("--latentK", type=int, default=32, help="latenK")
        parser.add_argument('--lr', type=float, default=0.00005, help="initial learning rate for adam")
        # parser.add_argument('--dropout', type=float, default=0, help="dropout")        

        parser.add_argument('--selectTable', default='clothing_', help="select db table")

        parser.add_argument('--intra_attn_method', default='dualFC', help="intra attention method")
        parser.add_argument('--inter_attn_method', default='general', help="inter attention method")

        parser.add_argument("--user_pretrain_wordVec", default="Y", 
                    choices=["Y", "N"], help="Wheather using pretrain embedding" )        

        parser.add_argument('--selectAttnModel', default='', help="Select model that wanna to show attn weight")
        parser.add_argument("--visulize_attn_epoch", type=int, default=0, help="No. of epoch that you like to show attention weight")

        parser.add_argument("--use_nltk_stopword", default="Y", choices=["Y", "N"], 
            help="Using NLTK stopword")

        self.parser = parser

    def parse(self, argv=None):
        if argv == None:
            opt = self.parser.parse_args(argv) # for running in jupyter notebook    
        else:
            opt = self.parser.parse_args()
        self.opt = opt
        self.config_path = os.path.join(opt.save_dir, 'opt.json')

        if opt.mode == "test":
            self.parser.add_argument("--model_dir", help="path to load model for test(the largest step or use --step to specify)")
            
        if opt.mode == "train" or opt.mode == "both":
            os.makedirs(opt.save_dir, exist_ok=True)
            os.makedirs(opt.save_dir + "/Loss", exist_ok=True)
            os.makedirs(opt.save_dir + "/Model", exist_ok=True)

            with open(self.config_path, 'w') as f:
                json.dump(self.opt.__dict__, f)
        
        if opt.mode == "showAttn":
            os.makedirs(opt.save_dir + "/VisualizeAttn", exist_ok=True)
        

        return opt
