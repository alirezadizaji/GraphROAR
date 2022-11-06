import matplotlib.pyplot as plt
import numpy as np
if __name__ == "__main__":
    x = np.linspace(0, 100, 100)

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

    ################# WITH NODE (ZERO FEATURE) ##############################
    # # BA2Motifs GCN3l ROAR-NORMALIZED
    # gnnexplainer = [-0.0,-0.0,15.2323,15.2086,25.3339,25.3295,34.0384,34.036,37.8379,37.8368,41.5225,41.5221,42.804,42.8038,43.7264,43.7263,45.3162,45.3161,45.7496,45.7496,45.7771,46.1191,45.5648,45.706,45.5652,45.8257,45.8176,45.8338,45.8256,45.8476,45.8262,45.8314,45.8295,45.8417,45.8402,45.8411,45.8411,45.8427,45.8369,45.8382,45.8382,45.8382,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381]
    # gradcam = [0.0,0.0,42.0595,42.0595,54.411,54.9334,42.8258,43.1792,54.337,54.3339,56.8505,56.8439,46.7691,46.4314,45.8329,45.2839,45.4178,45.4103,45.6194,45.6191,45.7181,45.795,45.7578,45.8153,45.8059,45.8264,45.8233,45.8244,45.8223,45.825,45.8232,45.8388,45.8374,45.839,45.8386,45.839,45.8383,45.8384,45.8382,45.8382,45.8382,45.8382,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381]
    # random = [-0.0,-0.0,15.933,15.9143,25.7843,25.7761,35.308,35.308,39.3245,39.3214,42.6039,42.6025,43.8391,43.8385,45.8746,45.8744,46.2687,46.2686,46.126,46.126,46.0244,46.1683,46.3093,46.3413,46.4207,46.4301,45.9971,46.0006,45.9512,45.9653,45.8685,45.8714,45.8589,45.8604,45.8849,45.8861,45.844,45.8445,45.8398,45.8399,45.8385,45.8385,45.8382,45.8382,45.8382,45.8382,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381]
    # pgexplainer = [-0.0,-0.0,35.9396,35.9396,45.3699,45.8923,25.8051,26.1585,41.9663,41.9631,43.7409,43.7343,30.8804,30.5427,39.7946,39.2456,42.3281,42.3206,44.7708,44.7706,45.4461,45.7971,45.7525,45.8204,45.8111,45.8361,45.8312,45.8389,45.8369,45.8386,45.8361,45.8375,45.8368,45.8378,45.8377,45.838,45.838,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381]
    # subgraphx = [0.0,0.0,36.6254,36.6254,55.0411,55.0411,59.6099,66.8183,60.138,57.8846,57.0674,57.0672,49.711,49.711,47.5915,47.5915,46.4741,48.434,46.4997,46.6642,46.0157,45.9763,45.5288,45.5152,45.4424,45.7484,45.7291,45.7211,45.6551,45.6509,45.8368,45.8418,45.8392,45.8399,45.8383,45.8389,45.8386,45.8387,45.8383,45.8383,45.8382,45.8382,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381]
    
    # BA2Motifs GCN3l KAR-NORMALIZED
    gnnexplainer = [45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8379,45.8379,45.8381,45.8381,45.8367,45.8364,45.8371,45.8352,45.8359,45.8378,45.8451,45.8314,45.8606,45.8535,45.8604,45.8333,45.8476,45.8539,45.8592,45.7313,45.7307,45.6349,46.0503,45.2385,45.2248,45.2335,44.6907,44.6907,44.8806,44.9791,42.3032,42.2313,43.1633,43.3725,39.4154,39.4844,39.0771,39.5682,31.2453,31.2654,28.3845,28.1128,7.063,-0.0]
    gradcam = [45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8382,45.8382,45.8383,45.8384,45.8383,45.8383,45.8385,45.838,45.8387,45.8384,45.841,45.8403,45.8577,45.8533,45.8484,45.8099,45.8143,45.8004,45.816,45.795,45.8348,45.8236,45.8697,45.86,45.8842,45.6803,44.7855,44.8222,36.9465,36.9369,32.4367,32.4297,13.4771,13.4771,11.2091,11.2086,9.2516,9.2519,5.6278,5.6343,1.9694,1.9688,28.0034,28.0029,0.9454,-0.0]
    random = [45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8382,45.8382,45.8383,45.8383,45.8382,45.8382,45.8398,45.8389,45.841,45.8411,45.8652,45.859,45.8566,45.8566,45.9976,45.9934,46.1734,46.1716,45.6643,45.5706,45.7381,45.7417,46.2891,46.2414,46.2377,46.2485,44.5296,44.5298,43.7858,43.7787,45.3285,45.3685,44.0436,44.3151,42.2026,42.1979,40.0864,40.5995,30.5158,30.5164,26.0365,25.7649,3.453,0.0]
    pgexplainer = [45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8369,45.8377,45.8377,45.8377,45.8377,45.837,45.8371,45.8364,45.8364,45.8367,45.8369,45.8361,45.837,45.834,45.8395,45.8351,45.8366,45.6892,45.6927,45.4584,44.8215,44.8215,44.6333,44.6334,43.5119,43.5124,42.8451,42.8476,41.8484,41.8621,39.0842,39.1022,33.505,33.5013,21.1961,21.1972,14.4365,14.4365,0.8006,0.0]
    subgraphx = [45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8382,45.8382,45.8383,45.8383,45.8392,45.8393,45.8388,45.8388,45.8394,45.8394,45.8403,45.8402,45.8449,45.8449,45.8485,45.8649,45.8289,45.8289,45.8352,45.8339,45.8626,45.8617,45.8691,45.8679,45.8998,45.7588,45.8222,44.4306,44.664,35.1273,35.3396,29.7443,26.4283,26.4283,24.9006,24.9006,18.3086,18.3086,15.7617,15.7617,-1.6741,-1.6741,1.4539,1.4539,1.9778,0.0]
    
     ################# WITH NODE (FEATURE) ##############################
    # BA2Motifs GCN3l ROAR-NORMALIZED
    # gnnexplainer = [0.0,0.0,15.2323,15.2086,15.8006,15.8037,27.6099,27.6031,28.3483,28.3488,33.9158,33.9127,34.5726,34.5729,38.6426,38.6412,40.6662,40.6663,43.903,43.9023,43.7706,44.916,42.7482,42.8388,42.8048,45.4819,44.9919,44.58,44.811,45.7941,45.2133,45.1853,45.428,46.0765,45.8863,45.6698,45.7169,45.9758,45.5674,45.7106,45.8654,45.8653,45.7977,45.7977,45.8486,45.8486,45.833,45.8331,45.8493,45.8493,45.8378,45.838,45.8453,45.8453,45.8397,45.8398,45.8419,45.8419,45.8374,45.8375,45.8388,45.839,45.8381,45.8387,45.8388,45.8387,45.838,45.8383,45.8384,45.8384,45.8381,45.8382,45.8382,45.8382,45.8381,45.8382,45.8382,45.8382,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381]
    # gradcam = [0.0,-0.0,42.0595,42.0594,51.3275,51.8499,40.5987,40.9521,64.9478,64.9447,68.236,68.2295,61.1721,61.082,49.7887,49.0127,46.6014,46.6014,46.939,46.9371,48.1468,48.5875,46.5576,46.5106,46.8665,46.7154,46.2872,46.3106,46.1711,46.2082,45.859,46.2176,46.012,46.2251,46.0971,46.1227,46.021,46.0262,45.9117,45.9017,45.9105,45.9105,45.8788,45.8789,45.8662,45.8662,45.8534,45.8534,45.8527,45.8527,45.847,45.847,45.8437,45.8437,45.8409,45.8409,45.8402,45.8402,45.839,45.839,45.8386,45.8386,45.8384,45.8384,45.8383,45.8383,45.8382,45.8382,45.8382,45.8382,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381]
    # random = [-0.0,0.0,15.933,15.9143,16.3665,16.3679,30.393,30.393,30.7049,30.6962,37.2613,37.262,38.2933,38.2891,43.8548,43.8552,44.6099,44.6074,45.6664,45.667,45.7015,47.2204,47.118,47.1408,47.3839,47.554,45.9377,45.9397,46.1537,46.5758,45.9196,45.89,45.9759,46.1326,46.1176,46.1845,45.9071,46.0084,45.8775,45.854,45.9073,45.9074,45.8368,45.8383,45.8684,45.8681,45.8369,45.8361,45.8478,45.8492,45.8382,45.8385,45.8438,45.8444,45.8393,45.8386,45.8407,45.8405,45.8381,45.8382,45.8393,45.8394,45.8384,45.8389,45.839,45.839,45.8381,45.8383,45.8383,45.8383,45.8382,45.8382,45.8382,45.8382,45.8381,45.8382,45.8382,45.8382,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381]
    # pgexplainer = [0.0,-0.0,35.9396,35.9396,44.6611,45.1835,25.2659,25.6192,43.7793,43.7762,47.871,47.8644,34.4835,34.3933,38.1762,37.4003,31.73,31.73,37.082,37.0801,40.4008,46.8654,45.1919,44.7242,45.0104,46.4964,45.7109,45.6492,45.6416,46.0017,45.5998,45.6157,45.5409,45.7881,45.7051,45.7569,45.7574,45.8448,45.8193,45.8288,45.857,45.8554,45.8347,45.8338,45.8401,45.84,45.8406,45.8408,45.8486,45.8486,45.8395,45.8396,45.8436,45.8436,45.8409,45.8409,45.844,45.8441,45.8414,45.8414,45.8417,45.8418,45.8394,45.8396,45.8392,45.8393,45.8384,45.8385,45.8385,45.8385,45.8382,45.8383,45.8382,45.8383,45.8382,45.8382,45.8382,45.8382,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381]
    # subgraphx = [-0.0,-0.0,36.6254,36.6254,53.8861,53.8861,55.2,57.0013,50.307,51.4451,64.4213,64.4211,50.2739,50.274,50.192,50.1921,47.9102,49.8811,49.2364,49.9389,47.8951,47.1876,45.8878,45.3481,43.8888,46.0599,45.5614,45.8484,44.9686,44.7535,46.4801,46.485,46.2165,46.235,46.0811,46.0853,46.0488,46.036,45.9693,45.969,45.925,45.925,45.8786,45.8786,45.8371,45.8372,45.8377,45.8377,45.8378,45.8381,45.8462,45.8462,45.8463,45.8463,45.8435,45.8435,45.8416,45.8416,45.8398,45.8398,45.8386,45.8385,45.8384,45.8383,45.8382,45.8381,45.838,45.8382,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381]
    # # BA2Motifs GCN3l KAR-NORMALIZED
    # gnnexplainer = [45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8382,45.8382,45.8382,45.8382,45.8383,45.8383,45.8384,45.8383,45.8385,45.8385,45.8388,45.8385,45.8394,45.839,45.8388,45.8388,45.8383,45.8384,45.8392,45.8391,45.8411,45.8412,45.8349,45.835,45.8371,45.8371,45.8064,45.8064,45.8306,45.8306,45.712,45.712,45.8186,45.8183,45.6021,45.5622,45.6516,45.3644,45.4432,45.805,46.1687,45.6944,46.0676,45.653,45.819,45.1687,45.3809,45.7354,46.1795,44.4801,44.4353,43.7618,45.83,43.9971,44.0636,44.1131,42.088,42.0882,42.47,42.9341,37.868,37.583,38.3857,38.634,34.0273,34.1141,33.3873,33.5953,23.1243,23.1443,19.72,19.4483,7.063,-0.0]    
    # gradcam = [45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.838,45.838,45.838,45.838,45.8376,45.8376,45.8373,45.8373,45.8362,45.8362,45.8371,45.8371,45.8275,45.8275,45.834,45.834,45.8345,45.8342,45.8341,45.8337,45.8337,45.833,45.8328,45.8324,45.8325,45.8324,45.8324,45.8275,45.8274,45.8222,45.8218,45.81,45.811,45.8053,45.8059,45.7875,45.7843,45.7814,45.6594,45.6723,45.5578,45.5818,45.4841,45.5382,45.0532,45.056,44.484,44.5105,43.7582,43.829,43.6606,43.759,43.0808,43.2156,42.5968,42.6918,38.4283,30.4919,30.5159,19.8137,19.8041,13.2033,13.1963,9.1756,9.1757,6.0122,6.0117,7.2518,7.252,4.9027,4.9093,2.0392,2.0386,28.0741,28.0736,0.9454,-0.0]
    # random = [45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8382,45.8382,45.8382,45.8382,45.8383,45.8382,45.8383,45.8382,45.8383,45.8382,45.8387,45.8385,45.8394,45.8391,45.8385,45.8385,45.8407,45.8407,45.84,45.84,45.8416,45.8416,45.8395,45.8395,45.8426,45.8425,45.8375,45.8375,45.866,45.8659,45.8537,45.8523,45.8794,45.8797,45.7948,45.8319,45.9823,45.8044,45.9465,45.9557,46.5054,46.2427,46.2559,46.1724,47.2147,46.953,47.0976,47.0177,47.543,46.581,46.8287,46.9256,47.6471,46.669,46.0795,46.1374,42.4559,42.4568,41.1423,41.0863,40.5442,40.8133,38.412,38.6994,35.4898,35.4844,30.8089,31.0391,21.1904,21.191,15.434,15.1623,3.453,0.0]
    # pgexplainer = [45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.838,45.838,45.838,45.838,45.8379,45.8375,45.8375,45.8373,45.8373,45.8368,45.8368,45.8361,45.8361,45.8342,45.8342,45.8324,45.8324,45.8281,45.8281,45.8251,45.8251,45.8234,45.8234,45.8197,45.8196,45.6131,45.7443,45.7458,45.7222,45.723,45.6262,45.6334,45.4429,45.4492,45.5201,45.531,45.5025,45.5223,45.4532,45.4944,45.3168,45.3468,44.2821,44.3153,43.7033,43.0805,43.0811,42.0742,42.0753,39.6501,39.6522,37.0574,37.0616,32.9375,32.9456,28.733,28.7511,22.4381,22.4343,12.3037,12.3048,8.3185,8.3185,0.8006,0.0]
    # subgraphx = [45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.838,45.8381,45.8377,45.8376,45.8376,45.8374,45.8375,45.8374,45.8376,45.8373,45.8373,45.8371,45.8372,45.8375,45.837,45.8375,45.8378,45.8385,45.838,45.8373,45.8376,45.8388,45.794,45.7949,45.7958,45.7998,45.8062,45.8019,45.8012,45.8001,45.8089,45.814,45.7866,45.7317,45.7456,45.6891,45.7066,45.5962,45.63,45.5369,45.5473,45.5583,45.5583,45.5245,45.2752,44.2343,44.3694,42.6373,42.7782,39.4886,39.6153,35.0686,35.2903,27.4678,27.8279,18.7052,18.909,13.7218,8.1636,8.1636,6.3896,6.3896,1.425,1.425,2.4948,2.4948,0.6899,0.6899,9.5634,9.5634,1.5633,0.0]
    
    #  ################# NODE Elimination ##############################
    # # BA2Motifs GCN3l ROAR-NORMALIZED
    # gnnexplainer = [0.0,0.0,15.2323,15.2086,10.6688,10.6818,21.541,21.5282,18.4213,18.4309,24.5661,24.555,24.9582,24.9665,32.2266,32.2169,32.3457,32.353,38.4324,38.4236,35.3417,41.8094,34.0811,28.8779,36.7518,46.3422,40.8209,33.0618,40.8406,51.2414,43.6247,36.9631,43.867,51.5497,42.2468,34.5901,39.6717,47.3999,39.5067,33.0259,46.8591,46.6309,34.6495,34.508,46.8264,46.8414,35.4037,35.4303,48.0123,46.891,39.9575,40.4824,53.3578,52.8217,40.9132,42.2245,55.4999,55.4616,42.1027,42.3038,53.3116,47.8984,41.5011,47.7487,54.2322,43.9479,36.1601,42.6246,49.5164,45.4628,37.361,44.6865,47.8939,39.428,35.5015,42.6938,49.5705,42.0708,38.0218,45.6153,43.5522,43.5522,44.9385,44.9385,48.3329,48.3329,45.8506,45.8506,48.9784,48.9784,47.9339,47.9339,48.0387,48.0387,45.8387,45.8387,45.8381,45.8381,45.8381,45.8381]
    # gradcam = [0.0,0.0,42.0595,42.0595,49.4166,49.939,37.7413,38.0947,68.5041,68.5009,73.3159,73.3094,77.5682,77.5656,64.6455,63.8136,48.8898,48.9787,60.0156,60.0032,80.3221,84.1127,68.2476,66.3565,83.0533,84.8823,75.0733,75.0496,78.3875,80.4091,70.1858,66.0574,63.8011,66.9744,60.6822,57.5177,60.4038,61.1275,54.0657,50.5282,62.3893,62.3723,51.2197,51.2347,58.5788,58.6101,51.2045,51.2057,57.0936,57.093,49.5686,49.5607,54.7284,54.7378,48.3562,48.357,54.6117,54.6118,44.4542,44.454,48.5538,47.001,44.3897,47.9889,49.6675,45.4795,42.4057,46.4272,49.0966,49.4732,46.227,46.2381,49.4409,47.458,44.2133,46.034,45.915,45.92,45.6829,45.7966,46.5145,46.5145,6.7142,6.7142,7.5891,7.5891,43.8381,43.8381,29.3386,29.3386,33.9912,33.9912,42.6366,42.6366,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381]
    # random = [0.0,-0.0,15.933,15.9143,11.6748,11.6891,24.9831,24.9831,21.2448,21.2283,28.394,28.4068,28.075,28.0597,37.3139,37.3252,34.3463,34.3295,39.3792,39.3898,39.3394,48.3179,41.6484,36.9238,42.4368,47.566,41.6435,33.8984,39.3066,44.3094,37.1297,29.165,33.7044,43.9087,41.0556,34.1842,37.0078,44.8704,41.269,33.5399,46.4986,45.6906,35.0175,37.0752,47.8446,46.8144,36.4737,36.3476,45.7821,47.1957,37.7129,38.2233,48.2501,47.2803,38.1802,38.5587,45.1829,43.8798,36.2728,37.7879,49.5769,41.9971,36.0546,44.9429,51.1327,41.1268,34.1847,43.5789,47.5267,42.0393,38.1724,45.4546,49.4846,44.7217,40.7958,46.2,51.0352,44.1517,39.0964,45.6011,42.5414,42.5414,45.6857,45.6857,40.0027,40.0027,45.8157,45.8173,49.3311,49.3294,44.7109,44.7109,39.6456,39.6456,45.8386,45.8386,45.8381,45.8381,45.8381,45.8381]
    # pgexplainer = [-0.0,0.0,35.9396,35.9396,44.395,44.9174,24.8369,25.1903,45.7199,45.7168,50.095,50.0884,45.1043,45.1018,42.6454,41.8135,18.5028,18.5917,33.7537,33.7413,42.6259,59.5015,47.4806,31.6912,47.6772,61.9226,46.3753,33.923,38.7402,50.7999,38.3042,30.3338,30.825,40.338,32.982,31.5785,36.3516,45.6637,42.1065,40.11,50.822,50.3705,40.2787,39.2962,42.904,42.4669,35.0286,35.7242,48.3801,47.2812,33.1969,34.3769,49.2723,48.096,39.9425,39.921,57.1733,56.6099,38.9991,39.9604,59.5903,48.5313,35.4792,45.9405,58.1478,47.5555,34.3208,47.2296,49.5399,40.4931,37.1298,46.4847,51.8858,43.7953,35.9114,44.5005,46.9539,36.714,32.0327,46.3924,38.7873,37.9947,43.4461,43.428,39.6679,39.6677,46.7447,46.7447,36.8869,36.8869,42.5077,42.5077,39.1147,39.1147,45.8389,45.8389,45.8381,45.8381,45.8381,45.8381]
    # subgraphx = [-0.0,0.0,36.6254,36.6254,53.2615,53.2615,49.2675,44.6615,37.295,41.6612,67.1391,67.1389,48.4047,48.4048,52.5966,52.5968,45.9158,48.5224,55.021,50.7904,62.3529,62.698,62.689,62.6672,58.5362,60.5153,56.2579,55.5028,55.8087,53.3856,62.4725,62.2267,57.1269,58.2628,58.8254,58.3632,60.0505,56.299,56.1607,55.9721,56.5168,56.4778,54.4939,54.518,48.057,48.0565,44.6397,44.646,48.2967,45.7441,59.449,59.449,63.3413,63.3413,62.2563,62.2563,57.3693,57.3693,55.7639,55.7639,51.1467,48.2957,46.4347,44.2584,46.5596,40.0942,38.6424,48.388,52.5058,50.1951,59.24,59.2399,51.4699,51.464,48.14,47.5791,52.042,52.3521,52.1963,39.2911,44.9599,45.1388,37.1245,12.2427,12.3622,10.0174,38.0562,38.0562,36.3988,36.3988,34.5831,34.5831,-6.3413,-6.3413,45.8379,45.8379,45.8381,45.8381,45.8381,45.8381]

    # # BA2Motifs GCN3l KAR-NORMALIZED
    # gnnexplainer = [45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8382,45.8382,43.2689,43.2689,42.2849,42.2849,46.2151,46.2167,47.5886,47.5937,44.9055,44.9008,48.8426,48.8459,46.6049,48.6338,45.7724,40.5359,42.4137,45.9502,42.826,41.9925,46.3609,50.3353,51.1894,48.2327,51.7207,50.8586,48.7264,45.6294,50.2417,52.1861,51.0825,48.0533,53.3244,53.3529,46.959,47.7157,52.4295,51.6766,49.008,48.2512,50.2696,50.8198,45.9331,45.8613,49.2857,49.1716,48.294,48.2446,49.8773,50.0861,49.0105,48.8122,53.2863,52.1315,49.9616,49.2158,49.9629,50.6224,49.736,48.6433,47.8046,44.6281,40.8382,39.5382,41.1188,39.6822,41.7672,40.5356,40.5649,39.1105,39.6445,39.2892,37.9467,37.9685,32.6893,32.6884,33.2945,33.7566,31.5092,31.4018,31.4652,31.4487,27.498,27.5501,25.8443,25.8223,19.1729,19.1929,15.0761,14.8044,7.063,-0.0]
    # gradcam = [45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8408,45.8408,87.208,87.208,84.0858,84.0858,85.7655,86.7315,36.9159,36.9127,37.5503,37.5666,31.5968,31.6036,24.7646,20.9509,20.5847,15.695,15.788,19.0453,19.1528,14.2018,12.9011,14.0709,14.0755,9.6559,9.6704,14.7846,14.8164,8.4066,8.4171,12.8813,12.9184,10.3827,10.8503,10.8616,7.6234,7.6216,10.8201,9.3639,9.0134,9.0837,12.4221,12.4812,8.9097,8.8745,6.6683,6.6936,5.1359,5.1935,7.6908,7.7787,6.3576,6.1781,8.8075,3.9036,3.8223,3.155,3.3063,3.3888,3.5255,0.6134,0.6123,1.0793,1.2093,-0.5442,-0.3568,0.219,0.3093,-0.5266,0.162,-1.0551,0.177,0.3214,1.4969,1.5035,2.3912,2.3816,-0.3063,-0.3133,3.3991,3.3992,3.268,3.2676,4.6391,4.6393,4.9922,4.9987,2.1375,2.1369,28.1746,28.1741,0.9454,-0.0]
    # random = [45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,43.6953,43.6953,45.5372,45.5828,45.2915,45.246,45.5782,45.5782,44.4265,44.4263,44.6597,44.6597,46.6549,47.8033,47.8094,46.6498,46.7863,48.8194,49.2933,45.0049,46.7354,47.3971,45.9546,44.2361,45.6846,46.1519,43.6792,42.5028,44.3268,45.1755,44.6737,42.3972,44.8263,44.8579,44.6245,44.5834,47.3799,47.4416,44.2756,44.3427,45.1662,44.5287,41.5163,41.4965,43.4644,43.4278,45.8112,45.8222,47.5475,46.8447,44.4853,45.1448,44.8696,43.3385,45.8947,44.9811,46.9244,46.9248,48.5308,48.0177,49.0954,47.9046,45.7297,45.8555,47.1822,44.899,45.4648,47.5844,47.5385,46.6274,45.8436,44.4007,41.1394,41.1575,36.2082,36.2116,35.5148,35.2409,33.3407,33.937,29.0207,29.021,29.0897,29.0866,21.8352,21.8359,16.7559,16.7565,9.9156,9.6439,3.453,-0.0]
    # pgexplainer = [45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8407,45.8407,44.9367,44.9367,64.8085,64.8085,49.2862,50.2523,33.0972,33.094,25.4791,25.4955,30.0774,30.0842,33.8338,35.5908,35.5909,34.0215,34.0218,40.1372,40.124,15.9132,15.9275,31.2963,31.2992,29.7037,29.7229,29.7213,29.7373,21.6788,21.7,25.5863,25.621,23.7427,20.5219,19.3048,19.0804,20.0521,18.4796,18.7951,18.0266,18.2648,25.1323,25.1262,20.1569,20.4269,22.9314,22.2291,20.2686,20.407,21.5697,21.6704,18.5634,18.4943,21.6532,20.7679,20.8666,22.6095,22.6123,20.2819,20.3713,20.8898,21.038,21.4854,21.6296,21.9149,22.1533,22.0412,21.969,23.5532,23.7563,22.7569,22.1698,19.7371,17.2536,17.2547,15.7857,15.7867,12.7277,12.7287,11.6689,11.6699,9.8672,9.8682,10.7837,10.8018,7.6081,7.6044,6.2837,6.2848,4.4603,4.4603,0.8006,-0.0]
    # subgraphx = [45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,45.8381,1.6535,2.6921,35.4962,75.96,75.96,62.1781,62.1781,51.7741,52.7399,48.974,49.3344,11.302,12.9162,13.084,18.4852,15.7352,17.5135,12.4854,16.5955,15.1412,15.7731,15.0937,18.0563,13.4808,14.257,14.6761,15.5633,13.7905,14.5846,13.214,12.2441,14.9852,13.711,11.117,11.4778,8.8401,9.3145,8.4954,8.5114,6.4032,6.2889,6.2226,6.0311,14.6088,14.5436,12.7065,12.7721,11.6325,11.5813,7.7969,7.7576,5.7976,5.7435,5.5759,5.6963,5.5212,9.4639,6.6376,6.5711,5.5325,0.9358,-3.2047,-3.304,7.2799,7.2574,7.6755,7.5592,7.0605,6.5383,2.2756,2.5526,2.114,3.9559,-0.1115,1.4067,1.5219,7.581,7.466,8.0405,0.66,0.66,-0.092,-0.092,-1.2602,-1.2602,-2.0206,-2.0206,-1.1292,-1.1292,1.598,1.598,1.9779,-0.0]

    plt.figure(figsize=(5.6, 3.5))
    # plt.xlabel("RSparsity (%)",fontsize=11)
    plt.xlabel("KSparsity (%)",fontsize=11)
    plt.ylabel("Fidelity (%)",fontsize=11)
    plt.plot(x, gnnexplainer)
    # plt.scatter(x, gnnexplainer)
    
    plt.plot(x, gradcam)
    # plt.scatter(x, gradcam)
    
    plt.plot(x, random)
    # plt.scatter(x, random)

    plt.plot(x, subgraphx, color='#D62728')
    # plt.scatter(x, subgraphx, color='#D62728')
    
    plt.plot(x, pgexplainer, color='#9467BD')
    # plt.scatter(x, pgexplainer, color='#9467BD')

    plt.rcParams.update({'font.size': 11})
    plt.xticks(fontsize=11) 
    plt.yticks(fontsize=11) 
    plt.legend(['GNNExplainer', 'GradCAM', 'Random', 'SubgraphX', 'PGExplainer'])
    plt.savefig('/home/alireza/Desktop/s-f_kar_norm.png', bbox_inches='tight', dpi=300)