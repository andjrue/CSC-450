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

x = data[:,0]
y = data[:,1]

x = x[~np.isnan(y)]
y = y[~np.isnan(y)]

inflection = int(3.5 * 7 * 24)

xa = x[:inflection]
ya = y[:inflection]

xb = x[inflection:]
yb = y[inflection:]

cubic_polynomial = np.poly1d(np.polyfit(xb, yb, 3))

def error(f, x, y):
  return np.sum((f(x) - y) ** 2)

# fp1 = np.polyfit(x, y, 1) Following book for reference, working
# f1 = np.poly1d(fp1)

# print("Model paramteres: %s" % fp1) # This actually gets a different answer than what is in the book, but it matches what was in the video.
                                    # Going to roll with it.






def plot_web_traffic(x, y, models=None, mx=None, ymax=None):

  plt.figure(figsize=(12,6))
  plt.scatter(x, y, s=10)
  plt.title("Web Traffic Over the Last Month")
  plt.xlabel("Weeks")
  plt.ylabel("Web Hits/Hour")

  if models:
    colors = ['g', 'k', 'b', 'm', 'r']
    linestyles = ['-', '-.', '--', ':', '-']

    for model, style, color in zip(models, linestyles, colors):
      plt.plot(mx / (7 * 24), model(mx), linestyle=style, linewidth=2, c=color)

    plt.legend(["d=%i" % m.order for m in models], loc="upper left")
  plt.autoscale(tight=True)
  plt.grid()
  plt.show()

plot_web_traffic(x, y, [cubic_polynomial], mx=np.linspace(0, 6 * 7 * 24, 100), ymax=1000)
