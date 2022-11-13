import sys, os, json
import numpy as np

from scipy import stats

# MIR-ST500 test set
ce10_ctc400 = {"COnPOff": [0.480872, 0.485512, 0.491522], "COnP": [0.663562, 0.674154, 0.675988], "COn": [0.746241, 0.75253, 0.755068]}
ce10 = {"COnPOff": [0.266501, 0.259994, 0.281845], "COnP": [0.436996, 0.429742, 0.465972], "COn": [0.560644, 0.562186, 0.593934]}

ce100_ctc400 = {"COnPOff": [0.534549, 0.543044, 0.538713], "COnP": [0.708169, 0.71351, 0.715514], "COn": [0.77283, 0.779388, 0.77727]}
ce100 = {"COnPOff": [0.418651, 0.416647, 0.412212], "COnP": [0.6251, 0.621265, 0.620061], "COn": [0.706657, 0.699049, 0.701494]}

print ("=== MIR-ST500 test set ===")
for metric in ["COn", "COnP", "COnPOff"]:
	ce10_ctc400_interval = stats.t.interval(0.95, len(ce10_ctc400[metric])-1, loc=np.mean(ce10_ctc400[metric]), scale=stats.sem(ce10_ctc400[metric]))
	ce10_interval = stats.t.interval(0.95, len(ce10[metric])-1, loc=np.mean(ce10[metric]), scale=stats.sem(ce10[metric]))
	ce100_ctc400_interval = stats.t.interval(0.95, len(ce100_ctc400[metric])-1, loc=np.mean(ce100_ctc400[metric]), scale=stats.sem(ce100_ctc400[metric]))
	ce100_interval = stats.t.interval(0.95, len(ce100[metric])-1, loc=np.mean(ce100[metric]), scale=stats.sem(ce100[metric]))

	print ("CE10+CTC400 {:s}: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(metric, ce10_ctc400_interval[0], ce10_ctc400_interval[1]
		, np.mean(ce10_ctc400[metric]), np.mean(ce10_ctc400[metric]) - ce10_ctc400_interval[0]))
	print ("CE10 {:s}: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(metric, ce10_interval[0], ce10_interval[1]
		, np.mean(ce10[metric]), np.mean(ce10[metric]) - ce10_interval[0]))
	print ("CE100+CTC400 {:s}: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(metric, ce100_ctc400_interval[0], ce100_ctc400_interval[1]
		, np.mean(ce100_ctc400[metric]), np.mean(ce100_ctc400[metric]) - ce100_ctc400_interval[0]))
	print ("CE100 {:s}: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(metric, ce100_interval[0], ce100_interval[1]
		, np.mean(ce100[metric]), np.mean(ce100[metric]) - ce100_interval[0]))

	result = stats.ttest_ind(ce100_ctc400[metric], ce10_ctc400[metric], equal_var=False, nan_policy='propagate', alternative='greater')
	print ("Ind t-test for CE100+CTC400 vs. CE10+CTC400", metric, "result:", result)

	result = stats.ttest_ind(ce10_ctc400[metric], ce10[metric], equal_var=False, nan_policy='propagate', alternative='greater')
	print ("Ind t-test for CE10+CTC400 vs. CE10", metric, "result:", result)

	result = stats.ttest_ind(ce10_ctc400[metric], ce100[metric], equal_var=False, nan_policy='propagate', alternative='greater')
	print ("Ind t-test for CE10+CTC400 vs. CE100", metric, "result:", result)

	result = stats.ttest_ind(ce100_ctc400[metric], ce10[metric], equal_var=False, nan_policy='propagate', alternative='greater')
	print ("Ind t-test for CE100+CTC400 vs. CE10", metric, "result:", result)

	result = stats.ttest_ind(ce100_ctc400[metric], ce100[metric], equal_var=False, nan_policy='propagate', alternative='greater')
	print ("Ind t-test for CE100+CTC400 vs. CE100", metric, "result:", result, "\n")

# ISMIR2014 dataset
ce10_ctc400 = {"COnPOff": [0.522737, 0.52967, 0.532645], "COnP": [0.720394, 0.696481, 0.717462], "COn": [0.872713, 0.89302, 0.87894]}
ce10 = {"COnPOff": [0.334255, 0.310404, 0.322256], "COnP": [0.505518, 0.520462, 0.499476], "COn": [0.681214, 0.718851, 0.691107]}

ce100_ctc400 = {"COnPOff": [0.595889, 0.635490, 0.606599], "COnP": [0.732098, 0.762282, 0.748445], "COn": [0.905322, 0.922659, 0.922530]}
ce100 = {"COnPOff": [0.475326, 0.446810, 0.434009], "COnP": [0.657045, 0.646034, 0.658242], "COn": [0.824671, 0.820597, 0.823828]}

print ("=== ISMIR2014 dataset ===")
for metric in ["COn", "COnP", "COnPOff"]:
	ce10_ctc400_interval = stats.t.interval(0.95, len(ce10_ctc400[metric])-1, loc=np.mean(ce10_ctc400[metric]), scale=stats.sem(ce10_ctc400[metric]))
	ce10_interval = stats.t.interval(0.95, len(ce10[metric])-1, loc=np.mean(ce10[metric]), scale=stats.sem(ce10[metric]))
	ce100_ctc400_interval = stats.t.interval(0.95, len(ce100_ctc400[metric])-1, loc=np.mean(ce100_ctc400[metric]), scale=stats.sem(ce100_ctc400[metric]))
	ce100_interval = stats.t.interval(0.95, len(ce100[metric])-1, loc=np.mean(ce100[metric]), scale=stats.sem(ce100[metric]))

	print ("CE10+CTC400 {:s}: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(metric, ce10_ctc400_interval[0], ce10_ctc400_interval[1]
		, np.mean(ce10_ctc400[metric]), np.mean(ce10_ctc400[metric]) - ce10_ctc400_interval[0]))
	print ("CE10 {:s}: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(metric, ce10_interval[0], ce10_interval[1]
		, np.mean(ce10[metric]), np.mean(ce10[metric]) - ce10_interval[0]))
	print ("CE100+CTC400 {:s}: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(metric, ce100_ctc400_interval[0], ce100_ctc400_interval[1]
		, np.mean(ce100_ctc400[metric]), np.mean(ce100_ctc400[metric]) - ce100_ctc400_interval[0]))
	print ("CE100 {:s}: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(metric, ce100_interval[0], ce100_interval[1]
		, np.mean(ce100[metric]), np.mean(ce100[metric]) - ce100_interval[0]))
	    
	result = stats.ttest_ind(ce100_ctc400[metric], ce10_ctc400[metric], equal_var=False, nan_policy='propagate', alternative='greater')
	print ("Ind t-test for CE100+CTC400 vs. CE10+CTC400", metric, "result:", result)

	result = stats.ttest_ind(ce10_ctc400[metric], ce10[metric], equal_var=False, nan_policy='propagate', alternative='greater')
	print ("Ind t-test for CE10+CTC400 vs. CE10", metric, "result:", result)

	result = stats.ttest_ind(ce10_ctc400[metric], ce100[metric], equal_var=False, nan_policy='propagate', alternative='greater')
	print ("Ind t-test for CE10+CTC400 vs. CE100", metric, "result:", result)

	result = stats.ttest_ind(ce100_ctc400[metric], ce10[metric], equal_var=False, nan_policy='propagate', alternative='greater')
	print ("Ind t-test for CE100+CTC400 vs. CE10", metric, "result:", result)

	result = stats.ttest_ind(ce100_ctc400[metric], ce100[metric], equal_var=False, nan_policy='propagate', alternative='greater')
	print ("Ind t-test for CE100+CTC400 vs. CE100", metric, "result:", result, "\n")

ce400 = {"COnPOff": [0.564670, 0.574562, 0.577373, 0.569821, 0.559417]}
result = stats.ttest_ind(ce100_ctc400["COnPOff"], ce400["COnPOff"], equal_var=False, nan_policy='propagate', alternative='greater')
print ("Ind t-test for CE100+CTC400 vs. CE400 COnPOff on ISMIR2014 dataset result:", result, "\n")