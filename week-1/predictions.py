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

x = data[:,0]
y = data[:,1]

x = x[~np.isnan(y)]
y = y[~np.isnan(y)]

inflection = int(3.5 * 7 * 24)

xa = x[:inflection]
ya = y[:inflection]

xb = x[inflection:]
yb = y[inflection:]

fa = np.poly1d(np.polyfit(xa, ya, 1))
fb = np.poly1d(np.polyfit(xb, yb, 1))

# print(xb) - Understanding what's going on now.

"""
For additional explanation, I decided to use the entire xb slice to train the cubic polynomial model.
I thought this would be the best way to go, and it took a while to get here. You could also train using the entirety of the data,
but given the recent uptick in users (after inflection), I thought that wouldn't be the optimal solution.

Now! You could also randomly choose a data set from the slice. This is great until its not. I found that the randomness
also led to chaos. There would be instances I ran it and we would hit 100k web hits in -1 one week, other instances it would
take 5.5 weeks, etc. etc.

Although it's not a completely fair test, it at least produces consistent results. It was cool to see how choosing different data sets
would reflect in the model.

With the current setup, it's expected we hit 100k web hits in ~8.6 weeks. I think that's a fair estimate.

You could also use random sampling like they do in the example, but then average it over 1000 runs. Pretty cool.
"""

start_idx = 0
end_idx = int(len(xb))

train = np.arange(start_idx, end_idx)
# print(train)

ans = np.poly1d(np.polyfit(xb[train], yb[train], 3))

# print(ans)

initial_max = fsolve(ans - 100000, x0=800) # I ran into some weird tupling problem here and wasn't able to do it exactly as the book has it.
                                   # From what I can tell, the math seems ok. The answer is reasonable, anyway.

                                   # Update! I took the same sample data from the books repo and used that. Although I couldn't get the same answer.
                                   # my results did seem like they were within reason. This is because the book is taking a random sample of data.

reached_max = initial_max[0] / (7 * 24)


print(f"Web hits will hit 100,000 per day in {reached_max} weeks")



def get_error(f, x, y):
  return np.sum((f(x) - y) ** 2)

fa_err = get_error(fa, xa, ya)
fb_err = get_error(fb, xb, yb)

print("Error inflection = %f" %  (fa_err + fb_err))

fp3 = np.polyfit(x, y, 3)
print("Model Params: \n %s" % fp3) # Matches video but is different from what is in the book

f3 = np.poly1d(fp3)

print(get_error(f3, x, y))

def plot_web_traffic(x, y, models=None, mx=None, ymax=None):

  plt.figure(figsize=(12,6))
  plt.scatter(x, y, s=10)
  plt.title("Web traffic over last month")

  plt.xlabel("Time")
  plt.ylabel("Hits/Hour")
  plt.xticks([w * 7 * 24 for w in range(6)], ["Week %i" % (w + 1) for w in range(6)])
  if models:
    colors = ["g", "k", "b", "m", "r"]
    linestyles = ["-", "-.", "--", ":", "-"]

    if mx is None:
      mx = np.linspace(0, x[-1], 1000)

    for model, style, color in zip(models, linestyles, colors):
      plt.plot(mx, model(mx), linestyle=style, linewidth=2, c=color)

    plt.legend(["d=%i" % m.order for m in models], loc="upper left")

  plt.autoscale(tight=True)
  if ymax:
    plt.ylim(ymax=ymax)
  plt.grid()
  plt.ylim(ymin=0)
  plt.show()

plot_web_traffic(x, y, [ans], mx=np.linspace(0, 6 * 7 * 24, 100), ymax=10000)

"""
Bib:
  This github repo helped me a ton here. I ran into a bunch of smaller problems, particularly with the graph.
  Without settings mins and maxes, the graph on page 31 of the book came out horribly. Was fixed
  after reviewing this ->
  https://github.com/PacktPublishing/Building-Machine-Learning-Systems-with-Python-Third-edition/blob/master/Chapter01/chapter_01.ipynb
"""
