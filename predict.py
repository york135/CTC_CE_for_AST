import argparse
import json
import torch
import sys, os, pickle, time
import yaml
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_utils import TrainSplitDataset
from predictor import NoteLevelAST

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
        inference_kwargs = yaml.load(stream, Loader=yaml.FullLoader)

    predictor = NoteLevelAST(network_file=inference_kwargs["network_file"], network_class_name=inference_kwargs["network_class_name"], device=device)

    dataset_to_predict = TrainSplitDataset()
    dataset_to_predict.load_dataset_from_h5(dataset_dir=args.dataset_dir, h5py_feature_file_name=args.h5py_feature_file_name)

    print ("Start inferencing dataset...", time.time())

    model_path = args.model_path
    predictor.load_model(model_path)

    results, raw_output = predictor.predict(dataset_to_predict, results_dict={}, show_tqdm=True
        , onset_thres=inference_kwargs["onset_thres"], offset_thres=inference_kwargs["offset_thres"])

    with open(args.predict_json_file, 'w') as f:
        output_string = json.dumps(results)
        f.write(output_string)

    print (time.time())
    print('Prediction done. JSON results written to: {}'.format(args.predict_json_file))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('yaml_path')
    parser.add_argument('dataset_dir')
    parser.add_argument('h5py_feature_file_name')
    parser.add_argument('predict_json_file')
    parser.add_argument('model_path')
    parser.add_argument('device')

    args = parser.parse_args()

    main(args)