import sys, os, json
import numpy as np

from scipy import stats

ce_con = [0.774577, 0.779325, 0.773991, 0.772098, 0.773266]
ctc_ce_con = [0.79385, 0.797042, 0.799063, 0.797303, 0.793511]
ce_smooth_con = [0.771266, 0.770666, 0.767657, 0.76902, 0.771294]

# ce_con = [0.706657, 0.699049, 0.701494]
# ctc_ce_con = [0.77283, 0.779388, 0.77727]

# ce_con = [0.569257, 0.561076, 0.593934]
# ctc_ce_con = [0.744366, 0.751188, 0.752358]


ce_con_interval = stats.t.interval(0.95, len(ce_con)-1, loc=np.mean(ce_con), scale=stats.sem(ce_con))
ctc_ce_con_interval = stats.t.interval(0.95, len(ctc_ce_con)-1, loc=np.mean(ctc_ce_con), scale=stats.sem(ctc_ce_con))
ce_smooth_con_interval = stats.t.interval(0.95, len(ce_smooth_con)-1, loc=np.mean(ce_smooth_con), scale=stats.sem(ce_smooth_con))

print ("CE COn: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(ce_con_interval[0], ce_con_interval[1], np.mean(ce_con), np.mean(ce_con) - ce_con_interval[0]))
print ("CTC+CE COn: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(ctc_ce_con_interval[0], ctc_ce_con_interval[1], np.mean(ctc_ce_con), np.mean(ctc_ce_con) - ctc_ce_con_interval[0]))
print ("CE smooth COn: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(ce_smooth_con_interval[0], ce_smooth_con_interval[1]
	, np.mean(ce_smooth_con), np.mean(ce_smooth_con) - ce_smooth_con_interval[0]))
    
result = stats.ttest_ind(ctc_ce_con, ce_con, equal_var=False, nan_policy='propagate', alternative='greater')
print ("Ind t-test for CTC+CE vs. CE COn result:", result)

result = stats.ttest_ind(ctc_ce_con, ce_smooth_con, equal_var=False, nan_policy='propagate', alternative='greater')
print ("Ind t-test for CTC+CE vs. CE smooth COn result:", result)

ce_conp = [0.718314, 0.721409, 0.720304, 0.718975, 0.718843]
ctc_ce_conp = [0.74068, 0.744627, 0.74832, 0.74299, 0.741911]
ce_smooth_conp = [0.715798, 0.717215, 0.710096, 0.715867, 0.714231]

ce_conp_interval = stats.t.interval(0.95, len(ce_conp)-1, loc=np.mean(ce_conp), scale=stats.sem(ce_conp))
ctc_ce_conp_interval = stats.t.interval(0.95, len(ctc_ce_conp)-1, loc=np.mean(ctc_ce_conp), scale=stats.sem(ctc_ce_conp))
ce_smooth_conp_interval = stats.t.interval(0.95, len(ce_smooth_conp)-1, loc=np.mean(ce_smooth_conp), scale=stats.sem(ce_smooth_conp))

print ("CE COnP: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(ce_conp_interval[0], ce_conp_interval[1], np.mean(ce_conp), np.mean(ce_conp) - ce_conp_interval[0]))
print ("CTC+CE COnP: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(ctc_ce_conp_interval[0], ctc_ce_conp_interval[1], np.mean(ctc_ce_conp), np.mean(ctc_ce_conp) - ctc_ce_conp_interval[0]))
print ("CE smooth COnP: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(ce_smooth_conp_interval[0], ce_smooth_conp_interval[1]
	, np.mean(ce_smooth_conp), np.mean(ce_smooth_conp) - ce_smooth_conp_interval[0]))
    
result = stats.ttest_ind(ctc_ce_conp, ce_conp, equal_var=False, nan_policy='propagate', alternative='greater')
print ("Ind t-test for CTC+CE vs. CE COnP result:", result)

result = stats.ttest_ind(ctc_ce_conp, ce_smooth_conp, equal_var=False, nan_policy='propagate', alternative='greater')
print ("Ind t-test for CTC+CE vs. CE smooth COnP result:", result)


ce_conpoff = [0.528745, 0.535136, 0.52884, 0.52961, 0.529773]
ctc_ce_conpoff = [0.569402, 0.574951, 0.57911, 0.57204, 0.574232]
ce_smooth_conpoff = [0.536219, 0.535585, 0.531738, 0.526681, 0.537717]

ce_conpoff_interval = stats.t.interval(0.95, len(ce_conpoff)-1, loc=np.mean(ce_conpoff), scale=stats.sem(ce_conpoff))
ctc_ce_conpoff_interval = stats.t.interval(0.95, len(ctc_ce_conpoff)-1, loc=np.mean(ctc_ce_conpoff), scale=stats.sem(ctc_ce_conpoff))
ce_smooth_conpoff_interval = stats.t.interval(0.95, len(ce_smooth_conpoff)-1, loc=np.mean(ce_smooth_conpoff), scale=stats.sem(ce_smooth_conpoff))

print ("CE COnPOff: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(ce_conpoff_interval[0], ce_conpoff_interval[1]
	, np.mean(ce_conpoff), np.mean(ce_conpoff) - ce_conpoff_interval[0]))
print ("CTC+CE COnPOff: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(ctc_ce_conpoff_interval[0], ctc_ce_conpoff_interval[1]
	, np.mean(ctc_ce_conpoff), np.mean(ctc_ce_conpoff) - ctc_ce_conpoff_interval[0]))
print ("CE smooth COnPOff: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(ce_smooth_conpoff_interval[0], ce_smooth_conpoff_interval[1]
	, np.mean(ce_smooth_conpoff), np.mean(ce_smooth_conpoff) - ce_smooth_conpoff_interval[0]))

result = stats.ttest_ind(ctc_ce_conpoff, ce_conpoff, equal_var=False, nan_policy='propagate', alternative='greater')
print ("Ind t-test for CTC+CE vs. CE COnPOff result:", result)

result = stats.ttest_ind(ctc_ce_conpoff, ce_smooth_conpoff, equal_var=False, nan_policy='propagate', alternative='greater')
print ("Ind t-test for CTC+CE vs. CE smooth COnPOff result:", result)


# ISMIR2014 dataset
print ("=== ISMIR2014 dataset ===")
ce_con = [0.903277, 0.913317, 0.909919, 0.912553, 0.911607]
ctc_ce_con = [0.934606, 0.925807, 0.928142, 0.928806, 0.933298]
ce_smooth_con = [0.922150, 0.906221, 0.909006, 0.907576, 0.910529]


ce_con_interval = stats.t.interval(0.95, len(ce_con)-1, loc=np.mean(ce_con), scale=stats.sem(ce_con))
ctc_ce_con_interval = stats.t.interval(0.95, len(ctc_ce_con)-1, loc=np.mean(ctc_ce_con), scale=stats.sem(ctc_ce_con))
ce_smooth_con_interval = stats.t.interval(0.95, len(ce_smooth_con)-1, loc=np.mean(ce_smooth_con), scale=stats.sem(ce_smooth_con))

print ("CE COn: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(ce_con_interval[0], ce_con_interval[1], np.mean(ce_con), np.mean(ce_con) - ce_con_interval[0]))
print ("CTC+CE COn: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(ctc_ce_con_interval[0], ctc_ce_con_interval[1], np.mean(ctc_ce_con), np.mean(ctc_ce_con) - ctc_ce_con_interval[0]))
print ("CE smooth COn: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(ce_smooth_con_interval[0], ce_smooth_con_interval[1]
	, np.mean(ce_smooth_con), np.mean(ce_smooth_con) - ce_smooth_con_interval[0]))
    
result = stats.ttest_ind(ctc_ce_con, ce_con, equal_var=False, nan_policy='propagate', alternative='greater')
print ("Pairwise t-test for CTC+CE vs. CE COn result:", result, "\n")

result = stats.ttest_ind(ctc_ce_con, ce_smooth_con, equal_var=False, nan_policy='propagate', alternative='greater')
print ("Pairwise t-test for CTC+CE vs. CE smooth COn result:", result, "\n")


ce_conp = [0.752285, 0.748746, 0.766066, 0.758542, 0.745667]
ctc_ce_conp = [0.770662, 0.756210, 0.766052, 0.757362, 0.786386]
ce_smooth_conp = [0.781133, 0.764492, 0.763635, 0.742278, 0.755442]


ce_conp_interval = stats.t.interval(0.95, len(ce_conp)-1, loc=np.mean(ce_conp), scale=stats.sem(ce_conp))
ctc_ce_conp_interval = stats.t.interval(0.95, len(ctc_ce_conp)-1, loc=np.mean(ctc_ce_conp), scale=stats.sem(ctc_ce_conp))
ce_smooth_conp_interval = stats.t.interval(0.95, len(ce_smooth_conp)-1, loc=np.mean(ce_smooth_conp), scale=stats.sem(ce_smooth_conp))

print ("CE COnP: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(ce_conp_interval[0], ce_conp_interval[1], np.mean(ce_conp), np.mean(ce_conp) - ce_conp_interval[0]))
print ("CTC+CE COnP: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(ctc_ce_conp_interval[0], ctc_ce_conp_interval[1], np.mean(ctc_ce_conp), np.mean(ctc_ce_conp) - ctc_ce_conp_interval[0]))
print ("CE smooth COnP: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(ce_smooth_conp_interval[0], ce_smooth_conp_interval[1]
	, np.mean(ce_smooth_conp), np.mean(ce_smooth_conp) - ce_smooth_conp_interval[0]))
    
result = stats.ttest_ind(ctc_ce_conp, ce_conp, equal_var=False, nan_policy='propagate', alternative='greater')
print ("Pairwise t-test for CTC+CE vs. CE COnP result:", result, "\n")

result = stats.ttest_ind(ctc_ce_conp, ce_smooth_conp, equal_var=False, nan_policy='propagate', alternative='greater')
print ("Pairwise t-test for CTC+CE vs. CE smooth COnP result:", result, "\n")

ce_conpoff = [0.564670, 0.574562, 0.577373, 0.569821, 0.559417]
ctc_ce_conpoff = [0.635565, 0.623312, 0.627134, 0.638191, 0.655592]
ce_smooth_conpoff = [0.622222, 0.599402, 0.611700, 0.592744, 0.587081]


ce_conpoff_interval = stats.t.interval(0.95, len(ce_conpoff)-1, loc=np.mean(ce_conpoff), scale=stats.sem(ce_conpoff))
ctc_ce_conpoff_interval = stats.t.interval(0.95, len(ctc_ce_conpoff)-1, loc=np.mean(ctc_ce_conpoff), scale=stats.sem(ctc_ce_conpoff))
ce_smooth_conpoff_interval = stats.t.interval(0.95, len(ce_smooth_conpoff)-1, loc=np.mean(ce_smooth_conpoff), scale=stats.sem(ce_smooth_conpoff))

print ("CE COnPOff: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(ce_conpoff_interval[0], ce_conpoff_interval[1]
	, np.mean(ce_conpoff), np.mean(ce_conpoff) - ce_conpoff_interval[0]))
print ("CTC+CE COnPOff: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(ctc_ce_conpoff_interval[0], ctc_ce_conpoff_interval[1]
	, np.mean(ctc_ce_conpoff), np.mean(ctc_ce_conpoff) - ctc_ce_conpoff_interval[0]))
print ("CE smooth COnPOff: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(ce_smooth_conpoff_interval[0], ce_smooth_conpoff_interval[1]
	, np.mean(ce_smooth_conpoff), np.mean(ce_smooth_conpoff) - ce_smooth_conpoff_interval[0]))
    
result = stats.ttest_ind(ctc_ce_conpoff, ce_conpoff, equal_var=False, nan_policy='propagate', alternative='greater')
print ("Pairwise t-test for CTC+CE vs. CE COnPOff result:", result, "\n")

result = stats.ttest_ind(ctc_ce_conpoff, ce_smooth_conpoff, equal_var=False, nan_policy='propagate', alternative='greater')
print ("Pairwise t-test for CTC+CE vs. CE smooth COnPOff result:", result, "\n")



# CTC only. Clearly CTC performs the worst, so I only compute 95% CI here.
ctc_mirst500 = {"COnPOff": [0.01461, 0.335269, 0.047765, 0.105125, 0.032604], "COnP": [0.034516, 0.582417, 0.201013, 0.290841, 0.168769]
					, "COn": [0.051512, 0.67014, 0.254696, 0.3323, 0.218305]}
ctc_ismir2014 = {"COnPOff": [0.00576, 0.377921, 0.081319, 0.132347, 0.047687], "COnP": [0.01452, 0.580303, 0.25791, 0.336861, 0.17226]
					, "COn": [0.022636, 0.767311, 0.329653, 0.445876, 0.215603]}

print ("=== CTC: MIR-ST500 test set ===")
for metric in ["COn", "COnP", "COnPOff"]:
	ctc_interval = stats.t.interval(0.95, len(ctc_mirst500[metric])-1, loc=np.mean(ctc_mirst500[metric]), scale=stats.sem(ctc_mirst500[metric]))
	
	print ("CTC {:s}: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(metric, ctc_interval[0], ctc_interval[1]
		, np.mean(ctc_mirst500[metric]), np.mean(ctc_mirst500[metric]) - ctc_interval[0]))

print ("=== CTC: ISMIR2014 dataset ===")
for metric in ["COn", "COnP", "COnPOff"]:
	ctc_interval = stats.t.interval(0.95, len(ctc_ismir2014[metric])-1, loc=np.mean(ctc_ismir2014[metric]), scale=stats.sem(ctc_ismir2014[metric]))
	
	print ("CTC {:s}: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(metric, ctc_interval[0], ctc_interval[1]
		, np.mean(ctc_ismir2014[metric]), np.mean(ctc_ismir2014[metric]) - ctc_interval[0]))