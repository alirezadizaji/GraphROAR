import matplotlib.pyplot as plt

if __name__ == "__main__":
    x = [0, 10, 30, 50, 70, 90, 100]

    # # BA3Motifs GCN3l ROAR-ABSOLUTE
    # gnnexplainer = [0.0,31.1715,34.0556,33.0337,32.9873,32.9795,32.9794]
    # gradcam = [0.0,52.2067,33.0351,32.9794,32.9793,32.9802,32.9794]
    # random = [0.0,29.9732,33.6065,33.035,32.9894,32.9796,32.9794]
    # pgexplainer = [0.0,26.2512,33.2365,33.0124,32.9814,32.9796,32.9794]
    # subgraphx = [0.0, 41.3574, 33.1734, 33.0124, 32.988, 32.9798,32.9794]

    # # BA3Motifs GCN3l ROAR-NORMALIZED
    # gnnexplainer = [0.0,31.2704,34.1408,33.1062,33.0587,33.0507,33.0506]
    # gradcam = [0.0,52.4588,33.1076,33.0506,33.0505,33.0514,33.0506]
    # random = [0.0,30.0832,33.6912,33.1075,33.0609,33.0508,33.0506]
    # pgexplainer = [0.0,26.3339,33.3142,33.0843,33.0527,33.0508,33.0506]
    # subgraphx = [0.0,41.5974,33.2488,33.0841, 33.0594, 33.051,33.0506]

    # # BA3Motifs GCN3l KAR-ABSOLUTE
    # gnnexplainer = [32.9794,32.9796,32.9837,33.1271,33.4082,30.7064,0.0]
    # gradcam = [32.9794,32.9795,33.0921,34.5272,31.9951,3.9718,-0.0]
    # random = [32.9794,32.9796,32.9848,33.1734,33.0227,33.5859,0.0]
    # pgexplainer = [32.9794,32.9796,32.9897,33.1152,33.4178,32.5034,0.0]
    # subgraphx = [32.9794,32.98, 32.9859, 33.0451, 33.5149,17.7891,-0.0]

    # BA3Motifs GCN3l KAR-NORMALIZED
    # gnnexplainer = [33.0506,33.0508,33.0549,33.2024,33.4938,30.8326,0.0]
    # gradcam = [33.0506,33.0507,33.1658,34.6331,32.1031,4.0133,0.0]
    # random = [33.0506,33.0508,33.0561,33.2489,33.1067,33.7203,-0.0]
    # pgexplainer = [33.0506,33.0508,33.0611,33.1894,33.5002,32.5605,-0.0]
    # subgraphx = [33.0506,33.0512,33.0573,33.1178,33.5994,17.9051,0.0]
    
    # BA2Motifs GCN3l ROAR-ABSOLUTE
    # gnnexplainer = [0.0,36.3793,39.9078,39.9177,39.9177,39.9177,39.9177]
    # gradcam = [-0.0,50.2563,39.9046,39.9177,39.9177,39.9177,39.9177]
    # random = [0.0,36.8855,39.9247,39.9177,39.9177,39.9177,39.9177]
    # pgexplainer = [0.0,38.5746,39.9159,39.9177,39.9177,39.9177,39.9177]
    # subgraphx = [0.0,51.411, 39.7548,39.9177,39.9177,39.9177,39.9177]

    # # BA2Motifs GCN3l ROAR-NORMALIZED
    # gnnexplainer = [0.0,41.5225,45.8262,45.8381,45.8381,45.8381,45.8381]
    # gradcam = [-0.0,56.8505,45.8233,45.8381,45.8381,45.8381,45.8381]
    # random = [0.0,42.6039,45.8685,45.8381,45.8381,45.8381,45.8381]
    # pgexplainer = [0.0,43.7409,45.8361,45.8381,45.8381,45.8381,45.8381]
    # subgraphx = [0.0,57.8846,45.6509,45.8381,45.8381,45.8381,45.8381]

    # # BA2Motifs GCN3l KAR-ABSOLUTE
    # gnnexplainer = [39.9177,39.9177,39.9177,39.9177,39.9374,41.6562,0.0]
    # gradcam = [39.9177,39.9177,39.9177,39.9177,39.8973,10.192,-0.0]
    # random = [39.9177,39.9177,39.9177,39.9177,40.0664,42.2022,0.0]
    # pgexplainer = [39.9177,39.9177,39.9177,39.9177,39.9166,36.6707,0.0]
    # subgraphx = [39.9177,39.9177,39.9177,39.9177,39.9426,22.6811,-0.0]

    # # BA2Motifs GCN3l KAR-NORMALIZE
    # gnnexplainer = [45.8381,45.8381,45.8381,45.8381,45.8604,47.3901,0.0]
    # gradcam = [45.8381,45.8381,45.8381,45.8381,45.8144,11.7479,0.0]
    # random = [45.8381,45.8381,45.8381,45.8381,46.0008,47.7883,0.0]
    # pgexplainer = [45.8381,45.8381,45.8381,45.8381,45.8369,42.1983,0.0]
    # subgraphx = [45.8381,45.8381,45.8381,45.8381,45.8649,26.3297,0.0]

    # MUTAG GIN3l ROAR-ABSOLUTE
    # gnnexplainer = [0.0,14.2174,30.2198,33.2427,33.5177,33.3536,33.391]
    # gradcam = [0.0,32.2135,36.8479,36.0133,35.2221,33.7596,33.391]
    # random = [0.0,19.5724,31.993,33.55,33.3197,33.3793,33.391]
    # pgexplainer = [-0.0,27.3179,33.2487,32.716,33.1603,33.374,33.391]
    # subgraphx = [0.0,15.5889,33.3494,34.2826,34.0169,33.4021,33.391]

    # MUTAG GIN3l ROAR-NORMALIZED
    gnnexplainer = [0.0,12.8445,28.8138,32.2943,32.5764,32.3194,32.3804]
    gradcam = [0.0,34.8017,39.3221,36.8981,35.0521,32.9351,32.3804]
    random = [0.0,19.329,31.5603,32.8178,32.3023,32.3553,32.3804]
    pgexplainer = [0.0,29.3566,33.2762,31.9909,32.2244,32.3541,32.3804]
    subgraphx = [0.0,17.1384,34.5091,34.4757,33.6249,32.4888,32.3804]

    # MUTAG GIN3l KAR-ABSOLUTE
    gnnexplainer = [33.391,33.3956,33.5193,33.4887,32.1618,23.2305,0.0]
    gradcam = [33.391,33.2993,32.9228,30.8584,23.7789,4.5915,0.0]
    random = [33.391,33.362,33.3686,32.6469,32.8561,17.3159,-0.0]
    pgexplainer = [33.391,33.263,33.3682,31.4079,27.7312,6.0519,0.0]
    subgraphx = [33.391,33.2592,33.0914,32.706,25.1696,0.3207,0.0]

    # MUTAG GIN3l KAR-NORMALIZED
    gnnexplainer = [32.3804,32.3839,32.5978,32.7728,31.7531,24.4873,0.0]
    gradcam = [32.3804,32.2496,31.7627,29.1507,20.5849,1.4622,-0.0]
    random = [32.3804,32.3411,32.3378,31.5824,32.7278,16.5684,0.0]
    pgexplainer = [32.3804,32.1739,32.5489,30.1528,26.0968,4.7827,0.0]
    subgraphx = [32.3804,32.1906,31.9766,31.7744,23.478,-3.1137,-0.0]

    plt.xlabel("Sparsity (KAR%)")
    plt.ylabel("Fidelity (Normalize %)")
    plt.plot(x, gnnexplainer)
    plt.scatter(x, gnnexplainer)
    
    plt.plot(x, gradcam)
    plt.scatter(x, gradcam)
    
    plt.plot(x, random)
    plt.scatter(x, random)

    plt.plot(x, subgraphx, color='#D62728')
    plt.scatter(x, subgraphx, color='#D62728')
    
    plt.plot(x, pgexplainer, color='#9467BD')
    plt.scatter(x, pgexplainer, color='#9467BD')

    plt.legend(['GNNExplainer', 'GradCAM', 'Random', 'SubgraphX', 'PGExplainer'])
    plt.show()