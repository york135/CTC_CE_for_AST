import torch
import torch.nn as nn
import argparse
from predictor import NoteLevelAST
import os, yaml
from data_utils import TrainSplitDataset

def main(args, training_args):

    device= 'cpu'
    if torch.cuda.is_available():
        device = args.device
    print ("use", device)

    torch.backends.cudnn.deterministic = True
    torch.multiprocessing.set_sharing_strategy('file_system')

    predictor = NoteLevelAST(network_file=training_args["network_file"], network_class_name=training_args["network_class_name"]
        , device=device, model_path=args.pretrained_path)

    if training_args["use_strongly_dataset"] == True:
        predictor.training_dataset = TrainSplitDataset()
        predictor.training_dataset.load_dataset_from_h5(dataset_dir=training_args["training_dataset_dir"], h5py_feature_file_name=training_args["h5py_feature_file_name"])
        predictor.training_dataset.load_gt_pkl_label(gt_dataset_path=training_args["training_gt_pkl_label"])

    if training_args["use_weakly_dataset"] == True:
        predictor.training_weakly_dataset = TrainSplitDataset()
        predictor.training_weakly_dataset.load_dataset_from_h5(dataset_dir=training_args["training_weakly_dataset_dir"], h5py_feature_file_name=training_args["h5py_feature_file_name"])
        predictor.training_weakly_dataset.load_gt_pkl_label(gt_dataset_path=training_args["training_weakly_gt_pkl_label"])

    predictor.validation_dataset = TrainSplitDataset()
    predictor.validation_dataset.load_dataset_from_h5(dataset_dir=training_args["val_dataset_dir"], h5py_feature_file_name=training_args["h5py_feature_file_name"])
    predictor.validation_dataset.load_gt_pkl_label(gt_dataset_path=training_args["val_gt_pkl_label"])


    predictor.fit(
        use_weakly_dataset=training_args["use_weakly_dataset"],
        use_strongly_dataset=training_args["use_strongly_dataset"],
        model_save_dir=training_args["model_save_dir"],
        batch_size=training_args["batch_size"],
        epoch=training_args["epoch"],
        warmup_epoch=training_args["warmup_epoch"],
        lr=training_args["lr"],
        save_every_epoch=1,
        model_save_prefix=training_args["model_save_prefix"],
        log_path=training_args["log_path"]
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('yaml_path')
    parser.add_argument('device')
    parser.add_argument('--pretrained_path', default=None)
    args = parser.parse_args()

    with open(args.yaml_path, 'r') as stream:
        training_args = yaml.load(stream, Loader=yaml.FullLoader)

    main(args, training_args)
