"""
Assignment:

  Read through the least squares approximation demonstration on Pages 22 through 37.
  You can test your understanding of the reading by completing the following exercise:

  1. Fit a cubic polynomial (degree 3) to the website traffic data.
     Using the code on Page 37 as a guide, when would a cubic model predict website traffic to reach 100,000 hits per hour?
"""


from matplotlib.lines import lineStyles
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

data = np.genfromtxt("web_traffic.tsv", delimiter="\t")

# print(data) Data loaded
# print(data.shape) (743,2)

hour = data[:,0]
web_hits = data[:,1]

hour = hour[~np.isnan(web_hits)]
web_hits = web_hits[~np.isnan(web_hits)]

inflection = int(3.5 * 7 * 24)

# Before inflection
hour_before_inflection = hour[:inflection]
web_hits_before_inflection = web_hits[:inflection]

# After inflection
hour_after_inflection = hour[inflection:]
web_hits_after_inflection = web_hits[inflection:]

cubic_polynomial = np.poly1d(np.polyfit(hour_after_inflection, web_hits_after_inflection, 2))

print("cubic_polynomial(x) = n%s" % cubic_polynomial)
max_hits = fsolve(cubic_polynomial - 100000, x0 = 800)
reached_max = max_hits[0] / (7 * 24)

print("100,000 hits/hour expected at week %f" % reached_max)



def plot_web_traffic(hour, web_hits, models=None):

  plt.figure(figsize=(12,6))
  plt.scatter(hour, web_hits, s=10)
  plt.title("Web Traffic Over the Last Month")

  plt.xlabel("Hours")
  plt.ylabel("Web Hits/Hour")
  plt.xticks([w*7*24 for w in range(5)],
            ['Week %i' %(w + 1) for w in range(5)])

  if models:
    colors = ['g', 'k', 'b', 'm', 'r']
    linestyles = ['-', '-.', '--', ':', '-']

    mx = np.linspace(0, hour[-1], 1000)
    for model, style, color in zip(models, linestyles, colors):
      plt.plot(mx, model(mx), linestyle=style, linewidth=2, c=color)

    plt.legend(["d=%i" % m.order for m in models], loc="upper left")
  plt.autoscale(tight=True)
  plt.grid()
  plt.show()

plot_web_traffic(hour, web_hits)
