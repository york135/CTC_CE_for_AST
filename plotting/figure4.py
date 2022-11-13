import matplotlib.pyplot as plt
import numpy as np

con = [np.mean([0.560644, 0.562186, 0.593934]), 0.656786, 0.717723, 0.731381, np.mean([0.746241, 0.75253, 0.755068])]
conp = [np.mean([0.436996, 0.429742, 0.465972]), 0.505454, 0.625984, 0.640526, np.mean([0.663562, 0.674154, 0.675988])]
conpoff = [np.mean([0.266501, 0.259994, 0.281845]), 0.323391, 0.434695, 0.455058, np.mean([0.480872, 0.485512, 0.491522])]

ctc_number = [0, 10, 50, 100, 400]

print (con)
print (conp)
print (conpoff)

plt.plot(ctc_number, con, ".-", label="COn F1-score")
plt.plot(ctc_number, conp, ".-", label="COnP F1-score")
plt.plot(ctc_number, conpoff, ".-", label="COnPOff F1-score")
plt.xlabel("Num. of weakly labeled data")
plt.ylabel("F1-score")
# plt.xscale('symlog')
plt.legend()
plt.title('F1-scores vs. number of weakly labeled data (CE10)')
plt.savefig("figure4.png", dpi=500)