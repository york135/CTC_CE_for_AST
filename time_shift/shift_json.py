import json
# import pandas as pd
import numpy as np
import time, sys, os



if __name__ == '__main__':
    gt_path = sys.argv[1]
    apply_time_shift_ms = float(sys.argv[2])
    apply_time_shift_s = apply_time_shift_ms / 1000.0

    with open(gt_path) as json_data:
        gt = json.load(json_data)

    for key in gt.keys():
        for i in range(len(gt[key])):
            gt[key][i][0] = gt[key][i][0] + apply_time_shift_s
            gt[key][i][1] = gt[key][i][1] + apply_time_shift_s

    output_path = sys.argv[3]
    with open(output_path, 'w') as f:
        output_string = json.dumps(gt)
        f.write(output_string)