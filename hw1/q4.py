import numpy as np

def monte_carlo_pi(N):
    # generating N random pts...
    x = np.random.uniform(-1, 1, size=N)
    y = np.random.uniform(-1, 1, size=N)


    # checking which pts lie inside the circle
    within_circle = (x**2 + y**2) <= 1


    # approximating pi...
    num_pts_inside = np.sum(within_circle)
    pi = 4 * (num_pts_inside / N)

    return pi


# test cases... 
for n in range(1,7):
  N = 10**n

  print(f'N = {N}, pi approx = {monte_carlo_pi(N)}')