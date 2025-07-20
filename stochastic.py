import pandas as pd
import matplotlib.pyplot as plt
import random

# Read the CSV
data = pd.read_csv('data.csv')

def stochastic_gradient_descent(m_now, b_now, points, L):
    N = len(points)
    i = random.randint(0, N - 1)
    x = points.iloc[i].usertime
    y = points.iloc[i].score

    m_gradient = -(2) * x * (y - (m_now * x + b_now))
    b_gradient = -(2) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b

m = 0
b = 0
L = 0.0001
iterations = 1000

for i in range(iterations):
    m, b = stochastic_gradient_descent(m, b, data, L)

print(m, b)

plt.scatter(data.usertime, data.score, color="black")
plt.plot(list(range(0, 100)), [m * x + b for x in range(0, 100)], color="red")
plt.show()
