import argparse
import json
import torch
import sys, os
import yaml
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_utils import TrainSplitDataset
from predictor import NoteLevelAST
from pathlib import Path
import pickle
import time

import warnings
warnings.filterwarnings('ignore')


def main(args):
    # Create predictor
    # print (time.time())
    device= 'cpu'
    if torch.cuda.is_available():
        device = args.device
    print ("use", device)
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    with open(args.yaml_path, 'r') as stream:
        select_param_args = yaml.load(stream, Loader=yaml.FullLoader)

    predictor = NoteLevelAST(network_file=select_param_args["network_file"], network_class_name=select_param_args["network_class_name"], device=device)

    dataset_to_predict = TrainSplitDataset()
    dataset_to_predict.load_dataset_from_h5(dataset_dir=select_param_args["dataset_dir"], h5py_feature_file_name=select_param_args["h5py_feature_file_name"])

    # Feed dataset to the model
    print (time.time())

    print ("Start evaluating...", time.time())

    start_epoch = 1
    end_epoch = select_param_args["epoch"] + 1

    for i in range(start_epoch, end_epoch):
        model_path = args.model_path_prefix + "_" + str(i)
        predictor.load_model(model_path)

        predict_file_path = args.prediction_output_prefix + "_" + str(i) + ".pkl"

        # onset thres and offset thres do not matter. What we want is only the raw output
        _, raw_output = predictor.predict(dataset_to_predict, results_dict={}, show_tqdm=True, onset_thres=0.98, offset_thres=0.98)

        with open(predict_file_path, 'wb') as f:
            pickle.dump(raw_output, f, protocol=4)
    
    print (time.time())
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('yaml_path')
    parser.add_argument('device')
    parser.add_argument('prediction_output_prefix')
    parser.add_argument('model_path_prefix')

    args = parser.parse_args()

    main(args)