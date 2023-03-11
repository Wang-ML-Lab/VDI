import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
a = pd.read_csv("USAGraphPos.csv")

pos = dict()

for tup in zip(a['state'], a['pos_x'], a['pos_y']):
    tup_tmp = [float(tup[1]), float(tup[2])]
    pos[tup[0]] = np.array(tup_tmp, dtype=np.float)

g = nx.read_adjlist('usa_map.txt')
print(pos)
nx.draw(g, pos, with_labels=True)
plt.show()