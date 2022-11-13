import matplotlib.pyplot as plt
import pickle
import statistics
import os, sys, json, time

if __name__ == '__main__':
    metrics = pickle.load(open(sys.argv[1], 'rb'))

    with open(sys.argv[2]) as json_data:
        threshold_data = json.load(json_data)

    # print (metrics)
    print ("Training set metrics path:", sys.argv[1])
    print ("Training set threshold data path:", sys.argv[2])
    best_conpoff = 0.0
    best_epoch = 0

    for i in range(1, 101):
        # print (i, metrics[i-1][1][2], best_conpoff)
        if metrics[i-1][1][2] >= best_conpoff:
            best_conpoff = metrics[i-1][1][2]
            best_epoch = i


    print ("Best epoch", best_epoch, "onset threshold =", threshold_data[best_epoch-1][0], "offset threshold =", threshold_data[best_epoch-1][1])
    print("         Precision Recall F1-score")
    print("COnPOff  %f %f %f" % (metrics[best_epoch-1][1][0], metrics[best_epoch-1][1][1], metrics[best_epoch-1][1][2]))
    print("COnP     %f %f %f" % (metrics[best_epoch-1][1][3], metrics[best_epoch-1][1][4], metrics[best_epoch-1][1][5]))
    print("COn      %f %f %f" % (metrics[best_epoch-1][1][6], metrics[best_epoch-1][1][7], metrics[best_epoch-1][1][8]))
    print ("gt note num:", metrics[best_epoch-1][1][9], "tr note num:", metrics[best_epoch-1][1][10])

        