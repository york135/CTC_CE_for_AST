import matplotlib.pyplot as plt
import pickle
import os
import statistics
import sys
import json
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from evaluate import MirEval
import numpy as np

# PG learning curve
if __name__ == '__main__':

    
    # eval_class = MirEval()
    # eval_class.add_gt("../json/ismir_gt.json")

    # json_path1 = "../json_result/noshift#1_ismir2014_92_80_36.json"

    # with open(json_path1) as json_data:
    #     result = json.load(json_data)

    # onset_result = []
    # for i in range(201):
    #     # print (time.time())
    #     best_con = 0.0
    #     best_result = None
    #     best_result10 = None
 
    #     eval_class.add_tr_tuple_and_prepare(result)
    #     # from -50ms to 50ms
    #     eval_result = eval_class.accuracy(0.05, method="traditional", print_result=False, shifting=((i-100)/2000.0))
    #     print (((i-100)/2000.0), eval_result[8])
    #     onset_result.append([i, eval_result[8]])

    # with open("noshift#1_ismir2014_92_80_36_shift_plot.pkl", 'wb') as f:
    #     pickle.dump(onset_result, f)
    
    
    # json_path2 = "../json_result/ce#2_ismir2014_100_78_40.json"

    # with open(json_path2) as json_data:
    #     result = json.load(json_data)

    # onset_result2 = []
    # for i in range(201):
    #     # print (time.time())
    #     best_con = 0.0
    #     best_result = None
    #     best_result10 = None
 
    #     eval_class.add_tr_tuple_and_prepare(result)
    #     # from -50ms to 50ms
    #     eval_result = eval_class.accuracy(0.05, method="traditional", print_result=False, shifting=((i-100)/2000.0))
    #     print (((i-100)/2000.0), eval_result[8])
    #     onset_result2.append([i, eval_result[8]])

    # with open("ce#2_ismir2014_100_78_40_shift_plot.pkl", 'wb') as f:
    #     pickle.dump(onset_result2, f)
    
    onset_result = pickle.load(open("noshift#1_ismir2014_92_80_36_shift_plot.pkl", 'rb'))
    onset_result2 = pickle.load(open("ce#2_ismir2014_100_78_40_shift_plot.pkl", 'rb'))

    shiftings = [(onset_result[i][0] - 100) / 2.0 for i in range(len(onset_result))]
    con_1 = [onset_result[i][1] for i in range(len(onset_result))]
    con_2 = [onset_result2[i][1] for i in range(len(onset_result2))]

    bests = [max(con_1), max(con_2)]
    best_idx = [np.argmax(con_1), np.argmax(con_2)]
    best_idx_annotation = ["(" + "{:.2f}".format((best_idx[0]-100)/2.0) + "ms, " + "{:.4f}".format(bests[0]) + ")"
    , "(" + "{:.2f}".format((best_idx[1]-100)/2.0) + "ms, " + "{:.4f}".format(bests[1]) + ")"]

    fig, ax = plt.subplots()
    # ax.scatter([((best_idx[0]-100)/2.0), ((best_idx[1]-100)/2.0)], bests, c="r")

    ax.scatter([((best_idx[0]-100)/2.0),], bests[0:1], c="b")
    ax.scatter([((best_idx[1]-100)/2.0),], bests[1:2], c="orange")

    ax.annotate(best_idx_annotation[0], (((best_idx[0]-100)/2.0), bests[0]), xycoords='data'
            , xytext=(19.5, 0.9388), size=8)
    ax.annotate(best_idx_annotation[1], (((best_idx[1]-100)/2.0), bests[1]), xycoords='data'
            , xytext=(-13, 0.9384), size=8)

    ax.axvline(x=0, color='r', linewidth=1.0)
    # ax.annotate(best_idx_annotation[0], (((best_idx[0]-100)/2.0), bests[0]), size=8)
    # ax.annotate(best_idx_annotation[1], (((best_idx[1]-100)/2.0), bests[1]), size=8)

    plt.plot(shiftings, con_1, label='CE (no shift)')
    plt.plot(shiftings, con_2, linestyle='--', label='CE (gt shifted)')

    ax.axis([-50, 50, 0.05, 0.97])

    plt.title('COn F1-score v.s. Time shift on ISMIR2014 dataset')
    plt.xlabel('Time shift (ms)')
    plt.ylabel('F1-score')
    plt.legend()
    plt.savefig('figure3.png', dpi=500)
    # plt.savefig('f1_score_to_time_shift_ismir2014_preliminary.eps', format='eps')
    # plt.savefig('f1_score_to_time_shift_ce_smooth_ctcce.png', format='png')
    plt.cla()
