import sys, os, json
import numpy as np

from scipy import stats

# MIR-ST500 test set
ce_ctc = {"COnPOff": [0.569402, 0.574951, 0.57911, 0.57204, 0.574232], "COnP": [0.74068, 0.744627, 0.74832, 0.74299, 0.741911], "COn": [0.79385, 0.797042, 0.799063, 0.797303, 0.793511]}
ce_ctc_mix = {"COnPOff": [0.548389, 0.54626, 0.5406], "COnP": [0.724734, 0.727024, 0.721102], "COn": [0.778521, 0.779623, 0.774639]}
ce_ctc_voc = {"COnPOff": [0.564155, 0.556007, 0.559029], "COnP": [0.732583, 0.727226, 0.7304], "COn": [0.78543, 0.780253, 0.78421]}

print ("=== MIR-ST500 test set ===")
for metric in ["COn", "COnP", "COnPOff"]:
	ce_ctc_interval = stats.t.interval(0.95, len(ce_ctc[metric])-1, loc=np.mean(ce_ctc[metric]), scale=stats.sem(ce_ctc[metric]))
	ce_ctc_mix_interval = stats.t.interval(0.95, len(ce_ctc_mix[metric])-1, loc=np.mean(ce_ctc_mix[metric]), scale=stats.sem(ce_ctc_mix[metric]))
	ce_ctc_voc_interval = stats.t.interval(0.95, len(ce_ctc_voc[metric])-1, loc=np.mean(ce_ctc_voc[metric]), scale=stats.sem(ce_ctc_voc[metric]))
	
	print ("CE+CTC {:s}: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(metric, ce_ctc_interval[0], ce_ctc_interval[1]
		, np.mean(ce_ctc[metric]), np.mean(ce_ctc[metric]) - ce_ctc_interval[0]))
	print ("CE+CTC (mix) {:s}: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(metric, ce_ctc_mix_interval[0], ce_ctc_mix_interval[1]
		, np.mean(ce_ctc_mix[metric]), np.mean(ce_ctc_mix[metric]) - ce_ctc_mix_interval[0]))
	print ("CE+CTC (voc) {:s}: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(metric, ce_ctc_voc_interval[0], ce_ctc_voc_interval[1]
		, np.mean(ce_ctc_voc[metric]), np.mean(ce_ctc_voc[metric]) - ce_ctc_voc_interval[0]))

	result = stats.ttest_ind(ce_ctc[metric], ce_ctc_mix[metric], equal_var=False, nan_policy='propagate', alternative='greater')
	print ("Ind t-test for CE+CTC vs. CE+CTC (mix)", metric, "result:", result)

	result = stats.ttest_ind(ce_ctc[metric], ce_ctc_voc[metric], equal_var=False, nan_policy='propagate', alternative='greater')
	print ("Ind t-test for CE+CTC vs. CE+CTC (voc)", metric, "result:", result)

	result = stats.ttest_ind(ce_ctc_mix[metric], ce_ctc_voc[metric], equal_var=False, nan_policy='propagate', alternative='greater')
	print ("Ind t-test for CE+CTC (mix) vs. CE+CTC (voc)", metric, "result:", result, "\n")

# ISMIR2014 dataset
ce_ctc = {"COnPOff": [0.635565, 0.623312, 0.627134, 0.638191, 0.655592], "COnP": [0.770662, 0.756210, 0.766052, 0.757362, 0.786386], "COn": [0.934606, 0.925807, 0.928142, 0.928806, 0.933298]}
ce_ctc_mix = {"COnPOff": [0.637325, 0.613912, 0.623117], "COnP": [0.781532, 0.770798, 0.778471], "COn": [0.928212, 0.916482, 0.922307]}
ce_ctc_voc = {"COnPOff": [0.641755, 0.629986, 0.642349], "COnP": [0.772922, 0.773596, 0.770493], "COn": [0.928781, 0.930632, 0.932787]}


print ("=== ISMIR2014 dataset ===")
for metric in ["COn", "COnP", "COnPOff"]:
	ce_ctc_interval = stats.t.interval(0.95, len(ce_ctc[metric])-1, loc=np.mean(ce_ctc[metric]), scale=stats.sem(ce_ctc[metric]))
	ce_ctc_mix_interval = stats.t.interval(0.95, len(ce_ctc_mix[metric])-1, loc=np.mean(ce_ctc_mix[metric]), scale=stats.sem(ce_ctc_mix[metric]))
	ce_ctc_voc_interval = stats.t.interval(0.95, len(ce_ctc_voc[metric])-1, loc=np.mean(ce_ctc_voc[metric]), scale=stats.sem(ce_ctc_voc[metric]))
	
	print ("CE+CTC {:s}: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(metric, ce_ctc_interval[0], ce_ctc_interval[1]
		, np.mean(ce_ctc[metric]), np.mean(ce_ctc[metric]) - ce_ctc_interval[0]))
	print ("CE+CTC (mix) {:s}: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(metric, ce_ctc_mix_interval[0], ce_ctc_mix_interval[1]
		, np.mean(ce_ctc_mix[metric]), np.mean(ce_ctc_mix[metric]) - ce_ctc_mix_interval[0]))
	print ("CE+CTC (voc) {:s}: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(metric, ce_ctc_voc_interval[0], ce_ctc_voc_interval[1]
		, np.mean(ce_ctc_voc[metric]), np.mean(ce_ctc_voc[metric]) - ce_ctc_voc_interval[0]))

	result = stats.ttest_ind(ce_ctc[metric], ce_ctc_mix[metric], equal_var=False, nan_policy='propagate', alternative='greater')
	print ("Ind t-test for CE+CTC vs. CE+CTC (mix)", metric, "result:", result)

	result = stats.ttest_ind(ce_ctc[metric], ce_ctc_voc[metric], equal_var=False, nan_policy='propagate', alternative='greater')
	print ("Ind t-test for CE+CTC vs. CE+CTC (voc)", metric, "result:", result)

	result = stats.ttest_ind(ce_ctc_mix[metric], ce_ctc_voc[metric], equal_var=False, nan_policy='propagate', alternative='greater')
	print ("Ind t-test for CE+CTC (mix) vs. CE+CTC (voc)", metric, "result:", result, "\n")