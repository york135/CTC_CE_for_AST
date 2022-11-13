# import argparse
import pickle
import json
from pathlib import Path
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_utils import TrainSplitDataset

import yaml

if __name__ == "__main__":
    with open(sys.argv[1], 'r') as stream:
        kwargs = yaml.load(stream, Loader=yaml.FullLoader)

    dataset = TrainSplitDataset()
    dataset.init_feature_extractor(feature_extractor_file=kwargs["feature_extractor_file"], feature_extractor_class_name=kwargs["feature_extractor_class_name"])
    dataset.create_dataset(audio_dir=kwargs["audio_dir"]
        , vocal_name=kwargs["vocal_name"], inst_name=kwargs["inst_name"]
        , feature_output_name=kwargs["feature_output_name"])