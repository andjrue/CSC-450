import numpy as np
from matplotlib.lines import lineStyles
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

data = np.genfromtxt("test_traffic.tsv", delimiter="\t")

x, y = data[:,0], data[:,1]

x, y = x[~np.isnan(y)], y[~np.isnan(y)]

inflection = int(3.5 * 7 * 24) # We're going to use the inflection point again. Otherwise, the models might not get to 100k.

xb, yb = x[inflection:], y[inflection:]

models = []
mod_num = 0

for model in range(1, 41):
  mod_num += 1
  model = np.poly1d(np.polyfit(xb, yb, mod_num))
  models.append(model)

# print(models) - This seems ok

def get_error(f, x, y):
  return np.sum((f(x) - y) ** 2)

model_error = []
# count = 0

for poly_model in models:
  # count += 1
  model_error.append(get_error(poly_model, xb, yb))
  # Seems to be working so far

# print(model_error)
# print(count) - There are 40

start_idx = 0
end_idx = int(len(xb))
train = np.arange(start_idx, end_idx)
# print(train)

def hit_100k(models, training_slice, x, y):
  model_number = 0
  time_to_100k = []

  for i in models:
    model_number += 1

    ans = np.poly1d(np.polyfit(x[train], y[train], model_number))

    max = fsolve(ans - 100000, x0=800)
    reached_max = max[0] / (7 * 24)

    time_to_100k.append(reached_max)


  print(time_to_100k)
  return time_to_100k

def plot100k(times):
  plt.figure(figsize=(12,6))
  plt.scatter(range(1, 41), times)

  plt.title("Time to Hit 100k Web Hits")
  plt.xlabel("Polynomial Degree")
  plt.ylabel("Predicted Time")
  plt.xticks(range(1, 41))

  plt.show()

def plot_error(arr):

  plt.figure(figsize=(12,6))
  plt.scatter(range(1, 41), arr)

  plt.title("Error for Each Model")
  plt.xlabel("Polynomial Degree")
  plt.ylabel("Total Error")
  plt.xticks(range(1, 41))

  plt.show()

time_to_100k = hit_100k(models, train, xb, yb)
plot_error(model_error)
plot100k(time_to_100k)

"""
This seems to be working. I ran the first 3 models through predictions.py to get a baseline and the math matches.
"""
