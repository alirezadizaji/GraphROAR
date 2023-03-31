import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    x = [0, 10, 30, 50, 70, 90, 100]
    ## BA-2Motifs GIN3l
    # gnnexplainer = [50, 50, 50, 58, 98, 100, 100]
    # gradcam = [50, 50, 50, 100, 100, 100, 100]
    # random = [50, 53, 51, 52, 58, 67, 100]
    # subgraphx = [50, 50, 50, 96, 100, 100, 100]
    # pgexplainer = [50, 65, 93, 100, 98, 100, 100]
    
    ## BA-2Motifs GIN3l (node elimination)
    # gnnexplainer = [50, 60, 61, 62, 76, 92, 100]
    # gradcam = [50, 97, 100, 100, 100, 100, 100]
    # random = [50, 50, 50, 50, 53, 65, 100]
    # subgraphx = [50, 76, 100, 99, 100, 100, 100]
    # pgexplainer = [50, 91, 99, 100, 100, 100, 100]

   # BA-2Motifs GIN3l (both)
    gradcam = [[50, 50.0, 50.0, 89.0, 94.0, 100.0, 100], [50, 50.0, 54.0, 82.0, 100.0, 100.0, 100], [50, 50.0, 56.0, 93.0, 100.0, 100.0, 100]]
    gnnexplainer = [[50, 57.0, 50.0, 51.0, 100.0, 100.0, 100], [50, 61.0, 51.0, 51.0, 100.0, 100.0, 100], [50, 53, 50, 58, 100.0, 100.0, 100]]
    pgexplainer = [[50, 53.0, 91.0, 94.0, 96.0, 96.0, 100], [50, 52.0, 89.0, 96.0, 97.0, 100.0, 100], [50, 98.0, 100.0, 100.0, 100.0, 100.0, 100]]
    subgraphx = [[50, 50, 51.0, 100.0, 100.0, 100.0, 100], [50, 73.0, 51.0, 97.0, 100.0, 100.0, 100], [50, 50.0, 100.0, 99.0, 100.0, 100.0, 100]]
    random = [[50, 50.0, 54.0, 61.0, 81.0, 100.0, 100], [50, 54.0, 50.0, 52.0, 58.0, 50.0, 100], [50, 50.0, 50.0, 52.0, 56.0, 100.0, 100]]
    # gradcam_gcn3l = [50, 51, 74, 94, 100, 100, 100]

    ## BA-2Motifs GCN3l
    # gnnexplainer = [50, 50, 61, 68, 67, 81, 100]
    # gradcam = [50, 95, 99, 100, 100, 100, 100]
    # random = [50, 55, 55, 57, 70, 90, 100]
    # subgraphx = [50, 86, 95, 94, 100, 99, 100]
    # pgexplainer = [50, 67, 97, 100, 100, 100, 100]

    ## BA-2Motifs GCN3l (node elimination)
    # gnnexplainer = [50, 51, 63, 71, 65, 70, 100]
    # gradcam = [50, 95, 99, 100, 100, 100, 100]
    # random = [50, 52, 59, 62, 61, 67, 100]
    # subgraphx = [50, 86, 94, 98, 100, 99, 100]
    # pgexplainer = [50, 66, 100, 95, 100, 100, 100]

    # BA-2Motifs GCN3l (both)
    gradcam = [[50, 50.0, 66.0, 98.0, 100.0, 100.0, 100], [50, 50.0, 71.0, 98.0, 100.0, 100.0, 100], [50, 51.0, 74.0, 94.0, 100.0, 100.0, 100]]
    gnnexplainer = [[50, 62.0, 53.0, 59.0, 69.0, 93.0, 100], [50, 56, 51, 63, 94, 98, 100, 100], [50, 56.0, 51.0, 63.0, 94.0, 98.0, 100]]
    pgexplainer = [[50, 50.0, 57.0, 100.0, 100.0, 100.0, 100], [50, 50.0, 60.0, 98.0, 100, 100, 100], [50, 50.0, 83.0, 97.0, 100.0, 100.0, 100]]
    subgraphx = [[50, 51.0, 58.0, 100.0, 100.0, 100.0, 100], [50, 55.0, 63.0, 99.0, 100, 100.0, 100], [50, 64.0, 71.0, 88.0, 100.0, 100.0, 100]]
    random = [[50, 52.0, 57.0, 63.0, 77.0, 99.0, 100], [50, 50.0, 54.0, 64.0, 85.0, 100.0, 100], [50, 50.0, 66.0, 70.0, 65.0, 100.0, 100]]

    ## MUTAG GCN3l
    # gnnexplainer = [85, 85, 90, 90, 90, 85, 90]
    # gradcam = [85, 85, 85, 90, 90, 90, 90]
    # random = [85, 85, 85, 83.6, 83.6, 83.6, 90]
    # subgraphx = [85, 90, 90, 85, 90, 95, 90]
    # pgexplainer = [85, 85, 85, 80, 85, 90, 90]

    # ## MUTAG GCN3l (node elimination)
    # gnnexplainer = [65, 85, 80, 90, 85, 80, 90]
    # gradcam = [65, 85, 90, 90, 90, 90, 90]
    # random = [65, 73.3, 76.67, 84.6, 81.67, 85, 90]
    # subgraphx = [65, 65, 85, 85, 90, 90, 90]
    # pgexplainer = [65, 85, 80, 80, 85, 85, 90]

    ## MUTAG GCN3l (skip eval)
    # gnnexplainer = [85, 85, 80, 80, 80, 80, 90]
    # gradcam = [85, 80, 80, 80, 80, 80, 90]
    # random = [85, 78.2, 82.5, 85, 85, 90, 90]
    # subgraphx = [85, 65, 80, 80, 80, 90, 90]
    # pgexplainer = [85, 75, 70, 85, 85, 90, 90]
    
    # MUTAG GCN3l (both)
    gradcam = [[65, 65.0, 80.0, 80.0, 80.0, 80.0, 90], [65, 75.0, 90.0, 90.0, 95.0, 95.0, 90], [65, 80.0, 85.0, 80.0, 90.0, 90.0, 90]]
    gnnexplainer = [[65, 75.0, 90.0, 90.0, 90.0, 90.0, 90], [65, 85.0, 90.0, 95.0, 95.0, 100.0, 90], [65, 80.0, 85.0, 85.0, 90.0, 90.0, 90]]
    pgexplainer = [[65, 65.0, 85.0, 80.0, 85.0, 90.0, 90], [65, 65.0, 75.0, 75.0, 90.0, 90.0, 90], [65, 65.0, 75.0, 80.0, 85.0, 90.0, 90]]
    subgraphx = [[65, 70.0, 75.0, 75.0, 75.0, 75.0, 90], [65, 90.0, 75.0, 85.0, 95.0, 90.0, 90], [65, 65.0, 80.0, 80.0, 75.0, 80.0, 90]]
    random = [[65, 75.0, 85.0, 85.0, 85.0, 85.0, 90], [65, 75.0, 80.0, 90.0, 80.0, 90.0, 90], [65, 75.0, 80.0, 85.0, 85.0, 90.0, 90]]

    ## MUTAG GIN3l
    # gnnexplainer = [65, 75, 80, 80, 85, 95, 100]
    # gradcam = [65, 90, 100, 100, 95, 100, 100]
    # random = [65, 85, 85, 88.5, 83.6, 88.5, 100]
    # subgraphx = [65, 90, 90, 85, 90, 95, 100]
    # pgexplainer = [65, 95, 85, 80, 75, 100, 100]

    ## MUTAG GIN3l (node elimination)
    # gnnexplainer = [65, 70, 70, 80, 90, 80, 100]
    # gradcam = [65, 85, 95, 100, 95, 95, 100]
    # random = [65, 75, 80, 80, 85, 85, 100]
    # subgraphx = [65, 75, 95, 85, 90, 90, 100]
    # pgexplainer = [65, 85, 85, 90, 85, 95, 100]

    # MUTAG GIN3l (skip during evaluation)
    # gnnexplainer = [65, 65, 65, 80, 80, 80, 100]
    # gradcam = [65, 65, 75, 85, 85, 90, 100]
    # random = [65, 67.5, 70, 82.5, 82.5, 80, 100]
    # subgraphx = [65, 65, 70, 75, 75, 80, 100]
    # pgexplainer = [65, 70, 80, 80, 80, 80, 100]

    # MUTAG GIN3l (both)
    gradcam = [[65, 75.0, 85.0, 85.0, 75.0, 85.0, 100], [65, 70.0, 85.0, 95.0, 100.0, 100.0, 100], [65, 75.0, 85.0, 85.0, 80.0, 80.0, 100]]
    gnnexplainer = [[65, 65.0, 75.0, 80.0, 90.0, 90.0, 100], [65, 75.0, 80.0, 80.0, 85.0, 85.0, 100], [65, 65.0, 75.0, 80.0, 80.0, 80.0, 100]]
    pgexplainer = [[65, 65.0, 80.0, 95.0, 90.0, 90.0, 100], [65, 65.0, 80.0, 90.0, 100.0, 100.0, 100], [65, 70.0, 80.0, 90.0, 90.0, 95.0, 100]]
    subgraphx = [[65, 65.0, 75.0, 75.0, 90.0, 90.0, 100], [65, 65.0, 75.0, 85.0, 100.0, 100.0, 100], [65, 65.0, 75.0, 75.0, 70.0, 80.0, 100]]
    random = [[65, 65.0, 75.0, 80.0, 80.0, 80.0, 100], [65, 70.0, 75.0, 80.0, 80.0, 80.0, 100], [65, 65.0, 65.0, 85.0, 85.0, 80.0, 100]]

    # REDDIT-BINARY GCN3l
    # gnnexplainer = [69, 82, 84, 90.5, 90, 93, 95]
    # gradcam = [69, 88, 90, 95, 94.5, 94, 95]
    # random = [69, 70.5, 80.25, 83.5, 90.25, 93.75, 95]
    # # subgraphx = [69, 65, 70, 75, 75, 80, 95]
    # pgexplainer = [69, 74.5, 72, 83, 85, 92, 95]

    # REDDIT-BINARY GCN3l (skipped during evaluation)
    # gnnexplainer = [69, 68.5, 58, 65, 85.5, 93, 95]
    # gradcam = [69, 71, 70, 71.5, 83.5, 91, 95]
    # random = [69, 69.75, 64.5, 76.5, 83.25, 92.5, 95]
    # # subgraphx = [69, 65, 70, 75, 75, 80, 95]
    # pgexplainer = [69, 71.5, 69, 72.5, 70, 92.5, 95]

    # REDDIT-BINARY GCN3l (both)
    gradcam = [[50, 79.0, 80.5, 87.5, 94.5, 94.5, 95], [50, 63.5, 70.0, 87.0, 94.5, 94.5, 95], [50, 75.0, 79.5, 89, 94.5, 94.5, 95]]
    gnnexplainer = [[50, 74.5, 68.5, 69.5, 90.5, 91.0, 95], [50, 78.5, 64.0, 87.5, 91.5, 96.0, 95], [50, 71.0, 70.5, 83.5, 94.0, 95.0, 95]]
    pgexplainer = [[50, 75.0, 86.0, 91.5, 91.5, 92.0, 95], [50, 77.5, 86.5, 93.5, 95.0, 95.5, 95], [50, 67.5, 83.0, 93.0, 95.0, 94.5, 95]]
    random = [[50, 72.0, 70.0, 83.0, 90.5, 94.0, 95], [50, 70.0, 73.0, 81.0, 88.5, 92.0, 95], [50, 74.0, 72.0, 81.0, 89.5, 94.0, 95]]
    
    # REDDIT-BINARY GIN3l
    # gnnexplainer = [50, 88, 88.5, 86, 84, 93, 90.5]
    # gradcam = [50, 89, 91, 91.5, 91.5, 92.5, 90.5]
    # random = [50, 70.5, 77.75, 91.5, 93.25, 94.25, 90.5]
    # # subgraphx = [69, 65, 70, 75, 75, 80, 90.5]
    # pgexplainer = [50, 63.5, 80, 82, 93, 94, 90.5]

    # REDDIT-BINARY GIN3l (skipped during evaluation)
    # gnnexplainer = [50, 50, 56, 69, 73.5, 86, 90.5]
    # gradcam = [50, 50, 54.5, 71, 79, 85, 90.5]
    # random = [50, 50, 54.25, 65.75, 71.5, 81.5, 90.5]
    # # subgraphx = [69, 65, 70, 75, 75, 80, 90.5]
    # pgexplainer = [50, 50, 67.5, 70, 72, 75.5, 90.5]

    # REDDIT-BINARY GIN3l (both)
    gradcam = [[50.0, 60.0, 63.0, 75.5, 83.0, 89.0, 90.5], [50.0, 59.5, 76.0, 80.0, 80.5, 88.0, 90.5], [50.0, 50.0, 62.0, 70.5, 82.0, 90.0, 90.5]]
    gnnexplainer = [[50.0, 54.5, 75.0, 61.5, 83.5, 88.5, 90.5], [50.0, 50.0, 56.0, 63.5, 85.5, 91.5, 90.5], [50.0, 50.0, 54.0, 61.5, 84.5, 89, 90.5]]
    pgexplainer = [[50.0, 55.0, 69.5, 73.5, 81.0, 90.5], [50.0, 55.0, 72.0, 87.0, 93.5, 94.0, 90.5], [50.0, 52.0, 66.5, 82.5, 89.0, 93.5, 90.5]]
    random = [[50.0, 50.0, 55.0, 68.5, 82.0, 90.5, 90.5], [50.0, 51.0, 59.5, 68.0, 89.0, 90.5], 50.0, 50.0, 53.0, 70.5, 75.0, 86.5, 90.5]

    # BA3Motifs GCN3l
    # gnnexplainer = [33, 42, 47.33, 48, 50.67, 75.33, 98.6]
    # gradcam = [33, 70, 93.3, 94, 95.33, 98, 98.6]
    # random = [33, 37.33, 38, 44, 46.67, 66, 98.6]
    # subgraphx = [33, 61.33, 75.33, 88.66, 86.66, 72.66, 98.6]
    # pgexplainer = [33, 64, 63.33, 47.33, 62.66, 80.66, 98.6]

    # BA3Motifs GCN3l (node elimination)
    # gnnexplainer = [33, 42, 47.33, 48, 50.67, 75.33, 98.6]
    # gradcam = [33, 74.67, 92.67, 90.67, 94.67, 96.67, 98.6]
    # random = [33, 48, 52, 43.33, 48.67, 78.67, 98.6]
    # subgraphx = [33, 83.33, 71.33, 82.67, 87.33, 82.67, 98.6]
    # pgexplainer = [33, 59.33, 54.67, 44.67, 68.67, 78.67, 98.6]

    # # BA3Motifs GCN3l (skipped during evaluation)
    # gnnexplainer = [33, 33.33, 33.33, 36, 45.33, 82.66, 98.6]
    # gradcam = [33, 46, 49.33, 33.33, 81.33, 94.66, 98.6]
    # random = [33, 36, 33.33, 45.33, 59.33, 92, 98.6]
    # subgraphx = [33, 42.66, 59.33, 66.66, 69.33, 78, 98.6]
    # pgexplainer = [33, 43.33, 40, 42, 56, 80, 98.6]

    # BA3Motifs GCN3l (both)
    gradcam = [[33.0, 37.33, 39.33, 87.33, 94.67, 98.67, 98.6], [33.0, 40.67, 44.0, 70.0, 90.0, 97.33, 98.6], [33.0, 46.0, 49.33, 33.33, 81.33, 94.67, 98.6]]
    gnnexplainer = [[33.0, 40.0, 34.0, 42.0, 58.67, 92.0, 98.6], [33.0, 39.33, 38.67, 50.67, 66.0, 96.0, 98.6], [33.0, 33.33, 33.33, 36.0, 45.33, 82.67, 98.6]]
    pgexplainer = [[33.0, 42.0, 36.0, 52.0, 80.67, 90.67, 98.6], [33.0, 42.67, 36.0, 38.67, 55.33, 78.67, 98.6], [33.0, 43.33, 40.0, 42.0, 56.0, 80.0, 98.6]]
    subgraphx = [[33.0, 33.33, 62.67, 75.33, 78.0, 81, 98.6], [33.0, 37.33, 52.0, 78.67, 76.67, 80.67, 98.6], [33.0, 34.0, 53.33, 72.67, 71.33, 78.0, 98.6]]
    random = [[33.0, 43.33, 44.67, 51.33, 62.67, 94.67, 98.6], [33.0, 36.0, 33.33, 45.33, 59.33, 92.0, 98.6], [33.0, 34.0, 35.33, 43.33, 61.33, 88.0, 98.6]]
    
    # BA3Motifs GIN3l
    # gnnexplainer = [33, 76, 72, 44.66, 51.33, 50, 95.33]
    # gradcam = [33, 93.33, 81.33, 91.33, 85.33, 93.33, 95.33]
    # random = [33, 67.33, 69.33, 38.66, 43.33, 47.33, 95.33]
    # subgraphx = [33, 76.67, 80.67, 85.33, 84.67, 65.33, 95.33]
    # pgexplainer = [33, 67.33, 66, 33.33, 86, 33.33, 95.33]

    # BA3Motifs GIN3l (node elimination)
    # gnnexplainer = [33, 54.67, 59.33, 44.67, 48, 70.67, 95.33]
    # gradcam = [33, 94, 84, 93.33, 91.33, 96, 95.33]
    # random = [33, 52, 41.33, 38, 40.67, 34, 95.33]
    # subgraphx = [33, 66, 64.67, 86, 71.33, 88.67, 95.33]
    # pgexplainer = [33, 44, 48.67, 54.67, 88.67, 80, 95.33]

    # BA3Motifs GIN3l (skip during evaluation)
    # gnnexplainer = [33, 33.33, 35.33, 48, 54, 95.3, 95.33]
    # gradcam = [33, 33.33, 35.33, 44, 61.33, 95, 95.33]
    # random = [33, 33.33, 33.33, 39.33, 36.66, 40.66, 95.33]
    # subgraphx = [33, 33.33, 34, 33.33, 42.67, 38, 95.33]
    # pgexplainer = [33, 33.33, 33.33, 40, 34, 44, 95.33]

    # BA3Motifs GIN3l (both)  
    gradcam = [[33.0, 33.33, 34.0, 56.0, 64.0, 99.33, 95.33], [33.0, 33.33, 34.67, 58.67, 64.67, 98.67, 95.33], [33.0, 33.33, 35.33, 44.0, 61.33, 96.0, 95.33]]
    gnnexplainer = [[33.0, 34.67, 33.33, 33.33, 40.67, 99.33, 95.33], [33.0, 33.33, 33.33, 34.67, 44.0, 68.67, 95.33], [33.0, 33.33, 35.33, 48.0, 54.0, 96.67, 95.33]]
    pgexplainer = [[33.0, 33.33, 33.33, 33.33, 35.33, 37.33, 95.33], [33.0, 33.33, 33.33, 33.33, 33.33, 43.33, 95.33], [33.0, 33.33, 33.33, 40.0, 34.0, 44.0, 95.33]]
    subgraphx = [[33.0, 38.67, 37.33, 39.33, 51.33, 66.67, 95.33], [33.0, 38.0, 34.67, 37.33, 48.0, 66.0, 95.33], [33.0, 33.33, 34.67, 34.0, 42.67, 65.33, 95.33]]
    random = [[33.0, 33.33, 34.0, 33.33, 43.33, 48.0, 95.33], [33.0, 33.33, 33.33, 39.33, 36.67, 40.67, 95.33], [33.0, 33.33, 33.33, 36.33, 39.67, 42.67, 95.33]]

    # # Enzyme GCN3l
    # gnnexplainer = [75.67, 70, 66.67, 68.33, 70, 70, 75]
    # gradcam = [75.67, 70, 71.66, 70, 66.67, 70, 75]
    # random = [75.67, 58.33, 63.33, 70, 70, 75, 75]
    # subgraphx = [75.67, 71.67, 71.67, 71.67, 70, 68.33, 75]
    # pgexplainer = [75.67, 71.66, 73.33, 70, 75, 73.33, 75]

    # Enzyme GCN3l (node elimination)
    # gnnexplainer = [16.67, 56.67, 53.33, 63.33, 66.67, 71.67, 75]
    # gradcam = [16.67, 66.67, 71.66, 71.66, 70, 70, 75]
    # random = [16.67, 42.22, 53.62, 60.55, 66.11, 69.44, 75]
    # subgraphx = [16.67, 67.64, 58.33, 66.67, 68.33, 70, 75]
    # pgexplainer = [16.67, 63.33, 65, 71.66, 71.66, 70, 75]

    # Enzyme GCN3l (both)
    gradcam = [[16.67, 48.33, 58.33, 63.33, 65.0, 65.0, 75.0], [16.67, 46.67, 51.67, 56.67, 53.33, 58.33, 75.0], [16.67, 55.0, 66.67, 65.0, 63.33, 66.67, 75.0]]
    gnnexplainer = [[16.67, 48.33, 53.33, 68.33, 66.67, 65.0, 75.0], [16.67, 46.67, 58.33, 56.67, 66.67, 68.33, 75.0], [16.67, 48.33, 56.67, 63.33, 65.0, 66.67, 75.0]]
    pgexplainer = [[16.67, 50.0, 56.67, 63.33, 66.67, 66.67, 75.0], [16.67, 46.67, 53.33, 58.33, 61.67, 70.0, 75.0], [16.67, 48.33, 56.67, 65.0, 68.33, 68.33, 75.0]]
    subgraphx = [[16.67, 50.0, 60.0, 56.67, 58.33, 65.0, 75.0], [16.67, 43.33, 48.33, 55.0, 55.0, 61.67, 75.0], [16.67, 48.33, 51.67, 61.67, 66.67, 70, 75.0]]
    random = [[16.67, 51.67, 53, 59, 63, 66, 75.0], [16.67, 46.67, 55.0, 61.67, 71.67, 70.0, 75.0], [16.67, 46.67, 63.33, 61.67, 70.0, 71.67, 75.0]]

    # Enzyme GIN3l
    # gnnexplainer = [61.67, 55, 51.67, 48.33, 63.33, 66.66, 75]
    # gradcam = [61.67, 65, 70, 66.67, 70, 71.67, 75]
    # random = [61.67, 55, 63.33, 56.67, 61.67, 66.67, 75]
    # subgraphx = [61.67, 60, 60, 68.33, 65, 66.68, 75]
    # pgexplainer = [61.67, 66.67, 71.67, 65, 61.67, 58.33, 75]

    # Enzyme GIN3l (node elimination)
    # gnnexplainer = [16.67, 46.67, 51.67, 56.67, 63.33, 65, 75]
    # gradcam = [16.67, 61.67, 61.67, 65, 63.33, 70, 75]
    # random = [16.67, 45.57, 53.33, 60.55, 66.11, 68.33, 75]
    # subgraphx = [16.67, 56.67, 60, 66.67, 60, 66.67, 75]
    # pgexplainer = [16.67, 75, 70, 71.66, 71.66, 63.33, 75]

    # Enzyme GIN3l (both)
    gradcam = [[16.67, 43.33, 51.67, 55.0, 55.0, 56.67, 75.0], [16.67, 33.33, 43.33, 51.67, 55.0, 63.33, 75.0], [16.67, 41.67, 50.0, 56.67, 63.33, 68.33, 75.0]]
    gnnexplainer = [[16.67, 35.0, 40.0, 50.0, 51.67, 56.67, 75.0], [16.67, 36.67, 46.67, 53.33, 56.67, 61.67, 75.0], [16.67, 31.67, 40.0, 55.0, 65.0, 68.33, 75.0]]
    pgexplainer = [[16.67, 30.0, 38.33, 36.67, 48.33, 60.0, 75.0], [16.67, 25.0, 41.67, 48.33, 55.0, 60.0, 75.0], [16.67, 33.33, 38.33, 51.67, 55.0, 70.0, 75.0]]
    subgraphx = [[16.67, 36.67, 46.67, 45.0, 58.33, 65.0, 75.0], [16.67, 30.0, 41.67, 63.33, 55.0, 63.33, 75.0], [16.67, 36.67, 51.67, 60.0, 56.67, 70.0, 75.0]]
    random = [[16.67, 33.33, 46.67, 60.0, 58.33, 68.33, 75.0], [16.67, 75.0], [16.67, 75.0]]
    # pgexplainer = [16.67, 33.33, 38.33, 51.66, 55, 70, 75]

    # IMDB-BINARY GCN3l
    # gnnexplainer = [50, 68, 72, 73, 68, 76, 75]
    # gradcam = [50, 77, 74, 77, 74, 73, 75]
    # random = [50, 69, 68, 67, 66, 72, 75]
    # subgraphx = [50, 76, 75, 73, 71, 71, 75]
    # pgexplainer = [50, 71, 70, 67, 75, 68, 75]

    # # IMDB-BINARY GCN3l (node elimination)
    # gnnexplainer = [50, 64, 70, 69, 74, 75, 75]
    # gradcam = [50, 75, 77, 73, 74, 72, 75]
    # random = [50, 65, 63, 65, 64, 73, 75]
    # subgraphx = [50, 78, 73, 71, 71, 74, 75]
    # pgexplainer = [50, 74, 69, 71, 71, 71, 75]

    # IMDB-BINARY GCN3l (skipped during evaluation)
    # gnnexplainer = [50, 60, 69, 65, 67, 71, 75]
    # gradcam = [50, 59, 59, 57, 61, 72, 75]
    # random = [50, 56, 55, 63, 64, 69, 75]
    # subgraphx = [50, 71, 55, 71, 70, 73, 75]
    # pgexplainer = [50, 66, 60, 59, 58, 73, 75]

    # IMDB-BINARY GCN3l (both)
    gradcam = [[50.0, 63.0, 53.0, 65.0, 65.0, 69, 75.0], [50.0, 56.0, 59.0, 63.0, 69.0, 71, 75.0], [50.0, 54.0, 57.0, 59.0, 65, 70.5, 75.0]]
    gnnexplainer = [[50.0, 56.0, 61.0, 62.0, 66.0, 68, 75.0], [50.0, 60.0, 61.0, 62.0, 70.0, 71, 75.0], [50.0, 66.0, 61.0, 63, 67, 68, 75.0]]
    pgexplainer = [[50.0, 66.0, 63.0, 63.0, 68.0, 70, 75.0], [50.0, 68.0, 67.0, 67.0, 66.0, 68, 75.0], [50.0, 70.0, 65.0, 68.0, 68.5, 70, 75.0]]
    subgraphx = [[50.0, 59.0, 67.0, 70.0, 69.0, 70.0, 75.0], [50.0, 66.0, 72.0, 67.0, 73.0, 71.0, 75.0], [50.0, 66.0, 64.0, 65.0, 63.0, 69.0, 75.0]]
    random = [[50.0, 64.0, 67.0, 65.0, 66.0, 67.0, 75.0], [50.0, 65.0, 67.0, 65.0, 66.0, 68.0, 75.0], [50.0, 60.0, 53.0, 68.0, 64.0, 71.0, 75.0]]

    # IMDB-BINARY GIN3l
    # gnnexplainer = [50, 78, 78, 76, 78, 79, 80]
    # gradcam = [50, 76, 80, 78, 79, 79, 80]
    # random = [50, 79, 80, 82, 82, 79, 80]
    # subgraphx = [50, 79, 82, 81, 84, 81, 80]
    # pgexplainer = [50, 79, 84, 83, 81, 81, 80]

    # IMDB-BINARY GIN3l (node elimination)
    # gnnexplainer = [50, 79, 79, 81, 81, 75, 80]
    # gradcam = [50, 76, 76, 77, 81, 79, 80]
    # random = [50, 64, 76, 79.5, 78, 78, 80]
    # subgraphx = [50, 73, 75, 80, 83, 80, 80]
    # pgexplainer = [50, 68, 69, 78, 76, 79, 80]

    # # IMDB-BINARY GIN3l (skipped during evaluation)
    # gnnexplainer = [50, 50, 50, 53, 71, 77, 80]
    # gradcam = [50, 50, 53, 55, 60, 76, 80]
    # random = [50, 50, 50, 50, 66, 76, 80]
    # subgraphx = [50, 50, 50, 58, 70, 78, 80]
    # pgexplainer = [50, 50, 50, 53, 62, 75, 80]

    # IMDB-BINARY GIN3l (both)
    gradcam = [[50.0, 50.0, 51.0, 57.0, 63.0, 81.0, 80.0], [50.0, 50.0, 57.0, 61.0, 62.0, 76.0, 80.0], [50.0, 50.0, 44.0, 50.0, 57.0, 61.0, 80.0]]
    gnnexplainer = [[50.0, 61.0, 51.0, 45.0, 57.0, 65.0, 80.0], [50.0, 50.0, 69.0, 50.0, 64.0, 76.0, 80.0], [50.0, 50.0, 50.0, 51.0, 62.0, 75.0, 80.0]]
    pgexplainer = [[50.0, 50.0, 50.0, 63.0, 76.0, 80.0, 80.0], [50.0, 73.0, 45.0, 53.0, 60.0, 73.0, 80.0], [50.0, 50.0, 50.0, 49.0, 60.0, 61.0, 80.0]]
    subgraphx = [[50.0, 57.0, 46.0, 49.0, 57.0, 63.0, 80.0], [50.0, 57.0, 50.0, 57.0, 74.0, 78.0, 80.0], [50.0, 57.0, 50.0, 59.0, 66.0, 79.0, 80.0]]
    random = [[50.0, 50.0, 50.0, 50.0, 60.0, 78.0, 80.0], [50.0, 50.0, 50.0, 50.0, 61.0, 77.0, 80.0], [50.0, 50.0, 50.0, 55.0, 60.0, 79.0, 80.0]]

    # # MSRC9 GCN3l (both)
    # gnnexplainer = [16, 80, 96, 96, 100, 100, 92]
    # gradcam = [16, 81, 92, 92, 97, 97, 92]
    # random = [16, 96, 95, 95, 96, 96, 92]
    # subgraphx = [16, 88, 96, 92, 92, 88, 92]
    # pgexplainer = [16, 92, 96, 96, 96, 96, 92]

    # # MSRC9 GIN3l (both)
    # gnnexplainer = [12, 88, 88, 88, 88, 96, 100]
    # gradcam = [12, 96, 92, 96, 96, 96, 100]
    # random = [12, 92, 95, 95, 95, 96, 100]
    # subgraphx = [12, 89, 97, 96, 96, 97, 100]
    # pgexplainer = [12, 88, 95, 96, 96, 96, 100]


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
    plt.show()