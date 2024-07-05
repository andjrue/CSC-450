"""
XX Part 1. If you have not yet done so, work through the first example in the book.
        We'll be modifying this code, so it will help to start with a working version.

        DONE

XX Part 2. Modify the code so that 40 models are built: one for each degree polynomial.
        This means you should have a degree one model, degree two model, degree three model, and so on.
        Tip: avoid creating 40 different variables.
        On Page 33, the author created 5 separate variables for each model: fb1, fb2, fb3, fb10, and fb100.
        Instead, create a list of polynomial models. This will make the next two parts a bit easier.

        I think this is done?

Part 3. Create a list of total errors, one for each polynomial model.
        Use this list to create a graph (using matplotlib) with polynomial degree on the x axis and total error on the y axis.

Part 4. Create a list of predicted lengths of time to reach 100,000 hits/hour, as predicted using each model.
        Use this list to create a graph (using matplotlib) with polynomial degree on the x axis, and predicted time on the y axis.

Part 5. Use your graphs to answer the following questions. As a rough guide of how in-depth your answers should be, all three should total about half a page to a page.
        If you find yourself writing more than a page, you are likely going into more detail than is required. If you write significantly less, you should probably expand on your points.

1. With regards to the total error graph, which model is "best"? Justify your answer.

2. With regards to the predicted graph, which model is "best"? Justify your answer.

3. Are any of the predictions unreasonable? If so, which--and why?

Turn in your responses, graphs, and Python code to the Homework 1 submission folder.

"""

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

for model in range(40):
  mod_num += 1
  model = np.poly1d(np.polyfit(xb, yb, mod_num))
  models.append(model)

# print(models) - This seems ok?





def get_error(f, x, y):
  return np.sum((f(x) - y) ** 2)
