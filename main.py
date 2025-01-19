# coding: utf-8

import os 
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='CleanMR', help='name of models') #MM_adapter_v2(best) MM_ab_wo_mm
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')

    config_dict = {
        'learning_rate': [1e-3],  
        'reg_weight': [0.01], 
        'n_layers': [2],
        'gpu_id': 2,
    }

    args, _ = parser.parse_known_args()

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)


