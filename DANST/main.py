import os
import sys
import argparse
import options
from model import *
from model.get_location import *
from model.neighbor_graph import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='murine_cellline', help='The name of benchmarking datasets')
args = parser.parse_args()

def main():
    random_seed(42)
    dataset = args.dataset
    ### Start Running DANST ###
    print("------Start Running DANST------")
    opt = options.get_option_list(dataset = dataset)
    ### Run Stage 1 ###
    print("------Start Running Stage 1 : Mixup reference------")
    model_mx = ReferMixup(opt)
    source_data, target_data = model_mx.mixup()
    print("The dim of source data is :")
    print(source_data.shape)
    print("The dim of target data is :")
    print(target_data.shape)
    print("Stage 1 : Mixup finished!")
    
    ### Run Stage 1-1 ###
    print("------Start Running Stage 1-1 : get location of source data------")
    source_data = location(opt, source_data,target_data,tool='leiden')
    print("Stage 1-1 : get location finished!")
    
    print("------Start Running Stage 1-2 : get adj for each dataset ------")
    adj = calculate_all_adj(opt,datatype='10x', source_data=source_data, target_data=target_data)
    print("Stage 1-2 : get adj finished!")


    ### Run Stage 2 ###
    print("------Start Running Stage 2 : Training VAEoptimize model------")
    model_im = VAEoptimize(opt) 
    source_recon_data,target_recon_data,trained_predictor = model_im.train(source_data, target_data,adj) 
    print("Stage 2 : VAEoptimize model training finished!")

    ### Run Stage 3 ###
    print("------Start Running Stage 3 : Training DANST model------")
    model_da = DANST(opt) 
    model_da.train(source_recon_data, target_recon_data, trained_predictor)  
    print("Stage 3 : DANST model training finished!")

    ### Run Stage 4 ###
    print("------Start Running Stage 4 : Inference for target data------")
    if opt['target_type'] == "simulated":
        final_preds_target, ground_truth_target = model_da.prediction()
        SavePredPlot(opt['SaveResultsDir'], final_preds_target, ground_truth_target)
        final_preds_target.to_csv(os.path.join(opt['SaveResultsDir'], "target_predicted_fractions.csv"))

    elif opt['target_type'] == "real":
        final_preds_target, _ = model_da.prediction() 
        final_preds_target.to_csv(os.path.join(opt['SaveResultsDir'], "target_predicted_fraction.csv"))
    print("Stage 4 : Inference for target data finished!")

if __name__ == "__main__":
    main()
