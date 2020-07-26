import pandas as pd
import numpy as np
import math
import statistics

############################ SECTION 3 ############################

# Consider a chess knight moving on the first quadrant of the plane. It starts at (0,0),
# and at each step will move two units in one direction and one unit in the other, such
# that x≥0 and y≥0. At each step the knight randomly selects a valid move, with uniform
# probability. For example, from (0,1), the knight will move to (1,3), (2,2), or (2,0),
# each with probability one-third.

# %%
# Q1:
# After 10 moves, what is the expected Euclidean distance of the knight from the origin?
# (If the knight is at (2,1), its distance is sqrt(2**2 + 1**2) = 2.24

x, y = [], []

# Initializing pseudo-random number generator
np.random.seed(123)

# This is the constant distance between consecutive knight moves in any possible direction
dist = math.sqrt(2**2 + 1**2)

# Initializing starting position of X and Y
x.append(0)
y.append(0)

moves = 0
while moves < 10:
    # Computing possible locations where the knight can move from current location
    Pos_X = []
    Pos_Y = []
    for i in range(4):  # Within (0,0) to (3,3) quadrant
        for j in range(4):  # Within (0,0) to (3,3) quadrant
            EuclDist = math.sqrt((i - x[-1])**2 + (j - y[-1])**2)
            if EuclDist == dist:
                Pos_X.append(i)
                Pos_Y.append(j)

    # Picking a possible random move based on equal probability
    RandMove = np.random.randint(0, len(Pos_X))
    x.append(Pos_X[RandMove])
    y.append(Pos_Y[RandMove])
    moves += 1

KnightXYlocs10 = list(zip(x, y))
FinalEuclDist10 = math.sqrt(x[-1]**2 + y[-1]**2)

# %%
# Q2:
# What is the expected standard deviation in this distance?

EuclDists = []
for i in range(10):
    EuclDists.append(math.sqrt(x[i+1]**2 + y[i+1]**2))

SD10 = statistics.stdev(EuclDists)

# %%
# Q3:
# If the knight made it a distance of at least 10 from the origin some time during
# those 10 moves, what is its expected Euclidean distance at the end of the 10 moves?

# Computing cumulative distance traveled by the Knight after 10 moves
CumEuclDist = 0
for i in range(10):
    CumEuclDist = CumEuclDist + \
        math.sqrt((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2)

# %%
# Q4:
# What is the expected standard deviation in this distance?

# Answer: The total cumulative distance after 10 moves will be the same every time
# (i.e. 22.36) since the Knight travels a distance of 2.236 each move regardless
# of the direction. Hence, the standard deviation will be 0.

# %%
# Q5:
# After 100 moves, what is the expected Euclidean distance of the knight from the origin?

x, y = [], []

# This is the constant distance between consecutive knight moves in any possible direction
dist = math.sqrt(2**2 + 1**2)

# Initializing starting position of X and Y
x.append(0)
y.append(0)

moves = 0
while moves < 100:
    # Computing possible locations where the knight can move from current location
    Pos_X = []
    Pos_Y = []
    for i in range(4):  # Within (0,0) to (3,3) quadrant
        for j in range(4):  # Within (0,0) to (3,3) quadrant
            EuclDist = math.sqrt((i - x[-1])**2 + (j - y[-1])**2)
            if EuclDist == dist:
                Pos_X.append(i)
                Pos_Y.append(j)

    # Picking a possible random move based on equal probability
    RandMove = np.random.randint(0, len(Pos_X))
    x.append(Pos_X[RandMove])
    y.append(Pos_Y[RandMove])
    moves += 1

KnightXYlocs100 = list(zip(x, y))
FinalEuclDist100 = math.sqrt(x[-1]**2 + y[-1]**2)

# %%
# Q6:
# What is the expected standard deviation in this distance?

EuclDists = []
for i in range(100):
    EuclDists.append(math.sqrt(x[i+1]**2 + y[i+1]**2))

SD100 = statistics.stdev(EuclDists)
