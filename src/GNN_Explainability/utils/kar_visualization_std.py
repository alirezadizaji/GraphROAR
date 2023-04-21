import os

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    x = [0, 10, 30, 50, 70, 90, 100]

    # BA-2Motifs GIN3l (both)
    # gradcam = [[50, 50.0, 50.0, 89.0, 94.0, 100.0, 100], [50, 50.0, 54.0, 82.0, 100.0, 100.0, 100], [50, 50.0, 56.0, 93.0, 100.0, 100.0, 100]]
    # gnnexplainer = [[50, 57.0, 50.0, 51.0, 100.0, 100.0, 100], [50, 61.0, 51.0, 51.0, 100.0, 100.0, 100], [50, 53, 50, 58, 100.0, 100.0, 100]]
    # pgexplainer = [[50, 53.0, 91.0, 94.0, 96.0, 96.0, 100], [50, 52.0, 89.0, 96.0, 97.0, 100.0, 100], [50, 98.0, 100.0, 100.0, 100.0, 100.0, 100]]
    # subgraphx = [[50, 50, 51.0, 100.0, 100.0, 100.0, 100], [50, 73.0, 51.0, 97.0, 100.0, 100.0, 100], [50, 50.0, 100.0, 99.0, 100.0, 100.0, 100]]
    # random = [[50, 50.0, 54.0, 61.0, 81.0, 100.0, 100], [50, 54.0, 50.0, 52.0, 58.0, 50.0, 100], [50, 50.0, 50.0, 52.0, 56.0, 100.0, 100]]

    # BA-2Motifs GCN3l (Node elimination only)
    gradcam = [[50.0, 96.0, 99.0, 100.0, 99.0, 100.0, 100.0], [50.0, 95.0, 99.0, 100.0, 100.0, 100.0, 100.0], [50.0, 95.0, 99.0, 100.0, 99.0, 100.0, 100.0]]
    gnnexplainer = [[50.0, 51.0, 63.0, 64.0, 64.0, 71.0, 100.0], [50.0, 51.0, 61.0, 68.0, 66.0, 73.0, 100.0], [50.0, 51.0, 65.0, 69.0, 68.0, 79.0, 100.0]]
    pgexplainer = [[50.0, 67.0, 100.0, 89.0, 98.0, 100.0, 100.0], [50.0, 68.0, 99.0, 92.0, 100.0, 100.0, 100.0], [50.0, 66.0, 100.0, 96.0, 98.0, 100.0, 100.0]]
    subgraphx = [[50.0, 86.0, 94.0, 98.0, 99.0, 99.0, 100.0], [50.0, 87.0, 90.0, 97.0, 99.0, 99.0, 100.0], [50.0, 85.0, 90.0, 99.0, 99.0, 99.0, 100.0]]
    random = [[50.0, 61.0, 57.0, 60.0, 59.0, 85.0, 100.0], [50.0, 54.0, 57.0, 59.0, 63.0, 84.0, 100.0], [50.0, 54.0, 66.0, 62.0, 69.0, 82.0, 100.0]]

    # BA-2Motifs GCN3l (both)
    # gradcam = [[50, 50.0, 66.0, 98.0, 100.0, 100.0, 100], [50, 50.0, 71.0, 98.0, 100.0, 100.0, 100], [50, 51.0, 74.0, 94.0, 100.0, 100.0, 100]]
    # gnnexplainer = [[50, 62.0, 53.0, 59.0, 69.0, 93.0, 100], [50, 56, 63, 94, 98, 100, 100], [50, 56.0, 51.0, 63.0, 94.0, 98.0, 100]]
    # pgexplainer = [[50, 50.0, 57.0, 100.0, 100.0, 100.0, 100], [50, 50.0, 60.0, 98.0, 100, 100, 100], [50, 50.0, 83.0, 97.0, 100.0, 100.0, 100]]
    # subgraphx = [[50, 51.0, 58.0, 100.0, 100.0, 100.0, 100], [50, 55.0, 63.0, 99.0, 100, 100.0, 100], [50, 64.0, 71.0, 88.0, 100.0, 100.0, 100]]
    # random = [[50, 52.0, 57.0, 63.0, 77.0, 99.0, 100], [50, 50.0, 54.0, 64.0, 85.0, 100.0, 100], [50, 50.0, 66.0, 70.0, 65.0, 100.0, 100]]

    # MUTAG GCN3l (Skip eval only)
    # gradcam = [[85.0, 85.0, 80.0, 80.0, 85.0, 85.0, 90.0], [85.0, 80.0, 80.0, 80.0, 80.0, 80.0, 90.0], [85.0, 95.0, 95.0, 95.0, 90.0, 90.0, 90.0]]
    # gnnexplainer = [[85.0, 80.0, 80.0, 85.0, 85.0, 85.0, 90.0], [85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 90.0], [85.0, 85.0, 85.0, 95.0, 95.0, 100.0, 90.0]]
    # pgexplainer = [[85.0, 80.0, 80.0, 80.0, 80.0, 85.0, 90.0], [85.0, 75.0, 75.0, 85.0, 85.0, 90.0, 90.0], [85.0, 70.0, 75.0, 75.0, 90.0, 90.0, 90.0]]
    # subgraphx = [[85.0, 70.0, 75.0, 80.0, 75.0, 80.0, 90.0], [85.0, 65.0, 70.0, 80.0, 75.0, 90.0, 90.0], [85.0, 95.0, 75.0, 90.0, 95.0, 85.0, 90.0]]
    # random = [[85.0, 75.0, 85.0, 85.0, 85.0, 85.0, 90.0], [85.0, 75.0, 80.0, 90.0, 80.0, 90.0, 90.0], [85.0, 75.0, 80.0, 85.0, 85.0, 90.0, 90.0]]
 
    # MUTAG GCN3l (both)
    # gradcam = [[65, 65.0, 80.0, 80.0, 80.0, 80.0, 90], [65, 75.0, 90.0, 90.0, 95.0, 95.0, 90], [65, 80.0, 85.0, 80.0, 90.0, 90.0, 90]]
    # gnnexplainer = [[65, 75.0, 90.0, 90.0, 90.0, 90.0, 90], [65, 85.0, 90.0, 95.0, 95.0, 100.0, 90], [65, 80.0, 85.0, 85.0, 90.0, 90.0, 90]]
    # pgexplainer = [[65, 65.0, 85.0, 80.0, 85.0, 90.0, 90], [65, 65.0, 75.0, 75.0, 90.0, 90.0, 90], [65, 65.0, 75.0, 80.0, 85.0, 90.0, 90]]
    # subgraphx = [[65, 70.0, 75.0, 75.0, 75.0, 75.0, 90], [65, 90.0, 75.0, 85.0, 95.0, 90.0, 90], [65, 65.0, 80.0, 80.0, 75.0, 80.0, 90]]
    # random = [[65, 75.0, 85.0, 85.0, 85.0, 85.0, 90], [65, 75.0, 80.0, 90.0, 80.0, 90.0, 90], [65, 75.0, 80.0, 85.0, 85.0, 90.0, 90]]

    # MUTAG GIN3l (both)
    # gradcam = [[65, 75.0, 85.0, 85.0, 75.0, 85.0, 100], [65, 70.0, 85.0, 95.0, 100.0, 100.0, 100], [65, 75.0, 85.0, 85.0, 80.0, 80.0, 100]]
    # gnnexplainer = [[65, 65.0, 75.0, 80.0, 90.0, 90.0, 100], [65, 75.0, 80.0, 80.0, 85.0, 85.0, 100], [65, 65.0, 75.0, 80.0, 80.0, 80.0, 100]]
    # pgexplainer = [[65, 65.0, 80.0, 95.0, 90.0, 90.0, 100], [65, 65.0, 80.0, 90.0, 100.0, 100.0, 100], [65, 70.0, 80.0, 90.0, 90.0, 95.0, 100]]
    # subgraphx = [[65, 65.0, 75.0, 75.0, 90.0, 90.0, 100], [65, 65.0, 75.0, 85.0, 100.0, 100.0, 100], [65, 65.0, 75.0, 75.0, 70.0, 80.0, 100]]
    # random = [[65, 65.0, 75.0, 80.0, 80.0, 80.0, 100], [65, 70.0, 75.0, 80.0, 80.0, 80.0, 100], [65, 65.0, 65.0, 85.0, 85.0, 80.0, 100]]

    # REDDIT-BINARY GCN3l (both)
    # gradcam = [[50, 79.0, 80.5, 87.5, 94.5, 94.5, 95], [50, 63.5, 70.0, 87.0, 94.5, 94.5, 95], [50, 75.0, 79.5, 89, 94.5, 94.5, 95]]
    # gnnexplainer = [[50, 74.5, 68.5, 69.5, 90.5, 91.0, 95], [50, 78.5, 64.0, 87.5, 91.5, 96.0, 95], [50, 71.0, 70.5, 83.5, 94.0, 95.0, 95]]
    # pgexplainer = [[50, 75.0, 86.0, 91.5, 91.5, 92.0, 95], [50, 77.5, 86.5, 93.5, 95.0, 95.5, 95], [50, 67.5, 83.0, 93.0, 95.0, 94.5, 95]]
    # random = [[50, 72.0, 70.0, 83.0, 90.5, 94.0, 95], [50, 70.0, 73.0, 81.0, 88.5, 92.0, 95], [50, 74.0, 72.0, 81.0, 89.5, 94.0, 95]]
    
    # REDDIT-BINARY (Skip eval only)
    # gradcam = [[50.0, 50.0, 57.0, 77.0, 79.0, 86.0, 90.5], [50.0, 50.0, 60.5, 69.0, 83.5, 91.0, 90.5], [50.0, 60.0, 63.0, 75.5, 83.0, 89.0, 90.5]]
    # gnnexplainer = [[50.0, 50.0, 63.5, 68.5, 73.0, 72.5, 90.5], [50.0, 50.0, 57.0, 73.0, 77.0, 86.5, 90.5], [50.0, 50.0, 56.0, 63.0, 76.5, 86.0, 90.5]]
    # pgexplainer = [[50.0, 55.0, 78.5, 71.0, 82.0, 88.5, 90.5], [50.0, 50.5, 59.5, 81.5, 86.0, 95.0, 90.5], [50.0, 53.5, 64.0, 78.5, 84.5, 91.5, 90.5]]
    # random = [[50.0, 50.0, 54.5, 66.5, 75.5, 85.5, 90.5], [50.0, 50.0, 53.0, 66.0, 77.0, 94.0, 90.5], [50.0, 50.0, 57.0, 71.0, 88.5, 94.0, 90.5]]

    # REDDIT-BINARY GIN3l (both)
    # gradcam = [[50.0, 60.0, 63.0, 75.5, 83.0, 89.0, 90.5], [50.0, 59.5, 76.0, 80.0, 80.5, 88.0, 90.5], [50.0, 50.0, 62.0, 70.5, 82.0, 90.0, 90.5]]
    # gnnexplainer = [[50.0, 54.5, 75.0, 61.5, 83.5, 88.5, 90.5], [50.0, 50.0, 56.0, 63.5, 85.5, 91.5, 90.5], [50.0, 50.0, 54.0, 61.5, 84.5, 89, 90.5]]
    # pgexplainer = [[50.0, 55.0, 69.5, 73.5, 81.0, 85, 90.5], [50.0, 55.0, 72.0, 87.0, 93.5, 94.0, 90.5], [50.0, 52.0, 66.5, 82.5, 89.0, 93.5, 90.5]]
    # random = [[50.0, 50.0, 55.0, 68.5, 82.0, 90.5, 90.5], [50.0, 51.0, 59.5, 68.0, 89.0, 89, 90.5], [50.0, 50.0, 53.0, 70.5, 75.0, 86.5, 90.5]]

    # BA3Motifs GCN3l (Skip eval only)
    # gradcam = [[33.0, 38.67, 50.67, 41.33, 87.33, 95.33, 98.6], [33.0, 41.33, 56.67, 64.0, 93.33, 96.67, 98.6], [33.0, 42.0, 59.33, 63.33, 86.0, 98.0, 98.6]]
    # gnnexplainer = [[33.0, 41.33, 33.33, 45.33, 48.0, 88.67, 98.6], [33.0, 39.33, 33.33, 37.33, 62.0, 97.33, 98.6], [33.0, 34.0, 33.33, 38.67, 46.0, 94.67, 98.6]]
    # pgexplainer = [[33.0, 38.0, 33.33, 41.33, 63.33, 86.0, 98.6], [33.0, 42.0, 33.33, 46.0, 44.67, 81.33, 98.6], [33.0, 39.33, 35.33, 34.0, 41.33, 78.67, 98.6]]
    # subgraphx = [[33.0, 48.0, 57.33, 58.67, 67.33, 73.33, 98.6], [33.0, 44.67, 49.33, 62.67, 72.0, 77.33, 98.6], [33.0, 38.0, 50.67, 65.33, 68.67, 76.0, 98.6]]
    # random = [[33.0, 43.33, 44.67, 51.33, 62.67, 94.67, 98.6], [33.0, 36.0, 33.33, 45.33, 59.33, 92.0, 98.6], [33.0, 36.0, 35.33, 42.33, 56.33, 89.0, 98.6]]

    # BA3Motifs GCN3l (both)
    # gradcam = [[33.0, 37.33, 39.33, 87.33, 94.67, 98.67, 98.6], [33.0, 40.67, 44.0, 70.0, 90.0, 97.33, 98.6], [33.0, 46.0, 49.33, 33.33, 81.33, 94.67, 98.6]]
    # gnnexplainer = [[33.0, 40.0, 34.0, 42.0, 58.67, 92.0, 98.6], [33.0, 39.33, 38.67, 50.67, 66.0, 96.0, 98.6], [33.0, 33.33, 33.33, 36.0, 45.33, 82.67, 98.6]]
    # pgexplainer = [[33.0, 42.0, 36.0, 52.0, 80.67, 90.67, 98.6], [33.0, 42.67, 36.0, 38.67, 55.33, 78.67, 98.6], [33.0, 43.33, 40.0, 42.0, 56.0, 80.0, 98.6]]
    # subgraphx = [[33.0, 33.33, 62.67, 75.33, 78.0, 81, 98.6], [33.0, 37.33, 52.0, 78.67, 76.67, 80.67, 98.6], [33.0, 34.0, 53.33, 72.67, 71.33, 78.0, 98.6]]
    # random = [[33.0, 43.33, 44.67, 51.33, 62.67, 94.67, 98.6], [33.0, 36.0, 33.33, 45.33, 59.33, 92.0, 98.6], [33.0, 34.0, 35.33, 43.33, 61.33, 88.0, 98.6]]
    
    # BA3Motifs GIN3l (both)  
    # gradcam = [[33.0, 33.33, 34.0, 56.0, 64.0, 99.33, 95.33], [33.0, 33.33, 34.67, 58.67, 64.67, 98.67, 95.33], [33.0, 33.33, 35.33, 44.0, 61.33, 96.0, 95.33]]
    # gnnexplainer = [[33.0, 34.67, 33.33, 33.33, 40.67, 99.33, 95.33], [33.0, 33.33, 33.33, 34.67, 44.0, 68.67, 95.33], [33.0, 33.33, 35.33, 48.0, 54.0, 96.67, 95.33]]
    # pgexplainer = [[33.0, 33.33, 33.33, 33.33, 35.33, 37.33, 95.33], [33.0, 33.33, 33.33, 33.33, 33.33, 43.33, 95.33], [33.0, 33.33, 33.33, 40.0, 34.0, 44.0, 95.33]]
    # subgraphx = [[33.0, 38.67, 37.33, 39.33, 51.33, 66.67, 95.33], [33.0, 38.0, 34.67, 37.33, 48.0, 66.0, 95.33], [33.0, 33.33, 34.67, 34.0, 42.67, 65.33, 95.33]]
    # random = [[33.0, 33.33, 34.0, 33.33, 43.33, 48.0, 95.33], [33.0, 33.33, 33.33, 39.33, 36.67, 40.67, 95.33], [33.0, 33.33, 33.33, 36.33, 39.67, 42.67, 95.33]]

    # Enzyme GCN3l (both)
    # gradcam = [[16.67, 48.33, 58.33, 63.33, 65.0, 65.0, 75.0], [16.67, 46.67, 51.67, 56.67, 53.33, 58.33, 75.0], [16.67, 55.0, 66.67, 65.0, 63.33, 66.67, 75.0]]
    # gnnexplainer = [[16.67, 48.33, 53.33, 68.33, 66.67, 65.0, 75.0], [16.67, 46.67, 58.33, 56.67, 66.67, 68.33, 75.0], [16.67, 48.33, 56.67, 63.33, 65.0, 66.67, 75.0]]
    # pgexplainer = [[16.67, 50.0, 56.67, 63.33, 66.67, 66.67, 75.0], [16.67, 46.67, 53.33, 58.33, 61.67, 70.0, 75.0], [16.67, 48.33, 56.67, 65.0, 68.33, 68.33, 75.0]]
    # subgraphx = [[16.67, 50.0, 60.0, 56.67, 58.33, 65.0, 75.0], [16.67, 43.33, 48.33, 55.0, 55.0, 61.67, 75.0], [16.67, 48.33, 51.67, 61.67, 66.67, 70, 75.0]]
    # random = [[16.67, 51.67, 53, 59, 63, 66, 75.0], [16.67, 46.67, 55.0, 61.67, 71.67, 70.0, 75.0], [16.67, 46.67, 63.33, 61.67, 64.0, 65.67, 75.0]]

    # Enzyme GIN3l (Skip eval only)
    # gradcam = [[61.67, 38.33, 45.0, 46.67, 50.0, 58.33, 75.0], [61.67, 28.33, 38.33, 53.33, 55.0, 65.0, 75.0], [61.67, 35.0, 48.33, 51.67, 63.33, 68.33, 75.0]]
    # gnnexplainer = [[61.67, 51.67, 50.0, 53.33, 60.0, 61.67, 75.0], [61.67, 35.0, 40.0, 43.33, 51.67, 60.0, 75.0], [61.67, 40.0, 43.33, 50.0, 63.33, 70.0, 75.0]]
    # pgexplainer = [[61.67, 33.33, 36.67, 38.33, 41.67, 48.33, 75.0], [61.67, 28.33, 30.0, 36.67, 41.67, 58.33, 75.0], [61.67, 36.67, 40.0, 43.33, 51.67, 65.0, 75.0]]
    # subgraphx = [[61.67, 41.67, 43.33, 53.33, 61.67, 58.33, 75.0], [61.67, 35.0, 40.0, 50.0, 46.67, 53.33, 75.0], [61.67, 43.33, 55.0, 58.33, 60.0, 66.67, 75.0]]
    # random = [[61.67, 33.33, 46.67, 60.0, 58.33, 68.33, 75.0], [61.67, 31.33, 44.67, 59.0, 57.33, 66.33, 75.0], [61.67, 35.33, 48.67, 58.0, 57.33, 67.33, 75.0]]

    # Enzyme GIN3l (both)
    # gradcam = [[16.67, 43.33, 51.67, 55.0, 55.0, 56.67, 75.0], [16.67, 33.33, 43.33, 51.67, 55.0, 63.33, 75.0], [16.67, 41.67, 50.0, 56.67, 63.33, 68.33, 75.0]]
    # gnnexplainer = [[16.67, 35.0, 40.0, 50.0, 51.67, 56.67, 75.0], [16.67, 36.67, 46.67, 53.33, 56.67, 61.67, 75.0], [16.67, 31.67, 40.0, 55.0, 65.0, 68.33, 75.0]]
    # pgexplainer = [[16.67, 30.0, 38.33, 36.67, 48.33, 60.0, 75.0], [16.67, 25.0, 41.67, 48.33, 55.0, 60.0, 75.0], [16.67, 33.33, 38.33, 51.67, 55.0, 70.0, 75.0]]
    # subgraphx = [[16.67, 36.67, 46.67, 45.0, 58.33, 65.0, 75.0], [16.67, 30.0, 41.67, 63.33, 55.0, 63.33, 75.0], [16.67, 36.67, 51.67, 60.0, 56.67, 70.0, 75.0]]
    # random = [[16.67, 33.33, 44.67, 57.0, 55.33, 65.33, 75.0], [16.67, 31.33, 45.67, 54.0, 57.33, 63.33, 75.0], [16.67, 31.33, 40.67, 50.0, 60.33, 60.33, 75.0]]

    # IMDB-BINARY GCN3l (both)
    # gradcam = [[50.0, 63.0, 53.0, 65.0, 65.0, 69, 75.0], [50.0, 56.0, 59.0, 63.0, 69.0, 71, 75.0], [50.0, 54.0, 57.0, 59.0, 65, 70.5, 75.0]]
    # gnnexplainer = [[50.0, 56.0, 61.0, 62.0, 66.0, 68, 75.0], [50.0, 60.0, 61.0, 62.0, 70.0, 71, 75.0], [50.0, 66.0, 61.0, 63, 67, 68, 75.0]]
    # pgexplainer = [[50.0, 66.0, 63.0, 63.0, 68.0, 70, 75.0], [50.0, 68.0, 67.0, 67.0, 66.0, 68, 75.0], [50.0, 70.0, 65.0, 68.0, 68.5, 70, 75.0]]
    # subgraphx = [[50.0, 59.0, 67.0, 70.0, 69.0, 70.0, 75.0], [50.0, 66.0, 72.0, 67.0, 73.0, 71.0, 75.0], [50.0, 66.0, 64.0, 65.0, 63.0, 69.0, 75.0]]
    # random = [[50.0, 64.0, 67.0, 65.0, 66.0, 67.0, 75.0], [50.0, 65.0, 67.0, 65.0, 66.0, 68.0, 75.0], [50.0, 60.0, 53.0, 68.0, 64.0, 71.0, 75.0]]

    # IMDB-BINARY GIN3l (both)
    # gradcam = [[50.0, 50.0, 51.0, 57.0, 63.0, 81.0, 80.0], [50.0, 50.0, 57.0, 61.0, 62.0, 76.0, 80.0], [50.0, 50.0, 44.0, 50.0, 57.0, 61.0, 80.0]]
    # gnnexplainer = [[50.0, 61.0, 51.0, 45.0, 57.0, 65.0, 80.0], [50.0, 50.0, 69.0, 50.0, 64.0, 76.0, 80.0], [50.0, 50.0, 50.0, 51.0, 62.0, 75.0, 80.0]]
    # pgexplainer = [[50.0, 50.0, 50.0, 63.0, 76.0, 80.0, 80.0], [50.0, 73.0, 45.0, 53.0, 60.0, 73.0, 80.0], [50.0, 50.0, 50.0, 49.0, 60.0, 61.0, 80.0]]
    # subgraphx = [[50.0, 57.0, 46.0, 49.0, 57.0, 63.0, 80.0], [50.0, 57.0, 50.0, 57.0, 74.0, 78.0, 80.0], [50.0, 57.0, 50.0, 59.0, 66.0, 79.0, 80.0]]
    # random = [[50.0, 50.0, 50.0, 50.0, 60.0, 78.0, 80.0], [50.0, 50.0, 50.0, 50.0, 61.0, 77.0, 80.0], [50.0, 50.0, 50.0, 55.0, 60.0, 79.0, 80.0]]

    gnnexplainer = np.array(gnnexplainer).T
    gradcam = np.array(gradcam).T
    random = np.array(random).T
    subgraphx = np.array(subgraphx).T
    pgexplainer = np.array(pgexplainer).T


    # plt.title('KAR performance on BA2Motifs-GCN3l (both)')
    plt.figure(figsize=(5.6, 4.2))
    plt.xlabel("Edge Keep (KAR%)",fontsize=11)
    plt.ylabel("Acc (%)", fontsize=11)

    # eb1 = plt.errorbar(x, np.mean(gnnexplainer, axis=1), np.std(gnnexplainer, axis=1), label=None)
    # eb1[-1][0].set_linestyle('--')
    
    # eb2 = plt.errorbar(x, np.mean(gradcam, axis=1), np.std(gradcam, axis=1), label=None)
    # eb2[-1][0].set_linestyle('--')
    
    # eb3 = plt.errorbar(x, np.mean(random, axis=1), np.std(gradcam, axis=1), label=None)
    # eb3[-1][0].set_linestyle('--')

    # eb4 = plt.errorbar(x, np.mean(subgraphx, axis=1), np.std(subgraphx, axis=1), label=None)
    # eb4[-1][0].set_linestyle('--')
    
    # eb5 = plt.errorbar(x, np.mean(pgexplainer, axis=1), np.std(pgexplainer, axis=1), label=None)
    # eb5[-1][0].set_linestyle('--')

    plt.plot(x, np.mean(gnnexplainer, axis=1))
    plt.fill_between(x, np.mean(gnnexplainer, axis=1) - np.std(gnnexplainer, axis=1),  np.mean(gnnexplainer, axis=1) + np.std(gnnexplainer, axis=1),
                 color='#1F77B4', alpha=0.2)

    plt.plot(x, np.mean(gradcam, axis=1))
    plt.fill_between(x, np.mean(gradcam, axis=1) - np.std(gradcam, axis=1),  np.mean(gradcam, axis=1) + np.std(gradcam, axis=1),
                 color='#FF7F0E', alpha=0.2)
    
    plt.plot(x, np.mean(random, axis=1))
    plt.fill_between(x, np.mean(random, axis=1) - np.std(random, axis=1),  np.mean(random, axis=1) + np.std(random, axis=1),
                 color='#2CA02C', alpha=0.2)
    
    plt.plot(x, np.mean(subgraphx, axis=1))
    plt.fill_between(x, np.mean(subgraphx, axis=1) - np.std(subgraphx, axis=1),  np.mean(subgraphx, axis=1) + np.std(subgraphx, axis=1),
                 color='#D62728', alpha=0.2)

    plt.plot(x, np.mean(pgexplainer, axis=1), color='#9467BD')
    plt.fill_between(x, np.mean(pgexplainer, axis=1) - np.std(pgexplainer, axis=1),  np.mean(pgexplainer, axis=1) + np.std(pgexplainer, axis=1),
                 color='#9467BD', alpha=0.2)
    # plt.plot(x, gradcam_gcn3l, color='darkblue')
    # plt.scatter(x, gradcam_gcn3l, color='darkblue')
    
    # plt.legend(['GNNExplainer', 'GradCAM', 'Random', 'SubgraphX', 'PGExplainer', 'GradCAM (GCN3l)'])
    plt.legend(['GNNExplainer', 'GradCAM', 'Random', 'SubgraphX', 'PGExplainer'])
    plt.rcParams.update({'font.size': 11})
    plt.xticks(fontsize=11) 
    plt.yticks(fontsize=11)
    
    pth = '/home/alireza/Desktop/cloudy/ba2motifs/gcn3l'
    os.makedirs(pth, exist_ok=True)
    plt.savefig(os.path.join(pth, 'kar_node_eli.png'), bbox_inches='tight', dpi=300)