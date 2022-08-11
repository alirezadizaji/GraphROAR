import os

import numpy as np

for i in range(1000):
    mask1 = np.load(f'../subgraphx_10%/{i}.npy')
    mask2 = np.load(f'../subgraphx_30%/{i}.npy')
    mask3 = np.load(f'../subgraphx_50%/{i}.npy')
    mask4 = np.load(f'../subgraphx_70%/{i}.npy')
    mask5 = np.load(f'../subgraphx_90%/{i}.npy')
    mask = mask1 + mask2 + mask3 + mask4 + mask5
    mask = mask.astype(np.float)
    mask = mask / 5
    np.save(f'{i}.npy', mask)