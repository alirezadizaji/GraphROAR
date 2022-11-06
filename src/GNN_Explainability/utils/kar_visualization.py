from turtle import color
import matplotlib.pyplot as plt
import matplotlib
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
    # gnnexplainer = [50, 58, 53, 59, 99, 100, 100]
    # gradcam = [50, 50, 52, 100, 100, 100, 100]
    # random = [50, 50, 51.33, 55, 66, 83.33, 100]
    # subgraphx = [50, 51, 59, 99.5, 99.5, 100, 100]
    # pgexplainer = [50, 55, 63, 98, 97.5, 100, 100]
    # gradcam_gcn3l = [50, 51, 74, 94, 100, 100, 100]

    ## BA-2Motifs GCN3l
    # gnnexplainer = [50, 50, 61, 68, 67, 81, 100]
    # gradcam = [50, 95, 99, 100, 100, 100, 100]
    # random = [50, 55, 55, 57, 70, 90, 100]
    # subgraphx = [50, 86, 95, 94, 100, 99, 100]
    # pgexplainer = [50, 67, 97, 100, 100, 100, 100]

    ## BA-2Motifs GCN3l (skip evaluation)
    # gnnexplainer = [50, 50, 50, 50, 65, 97, 100]
    # gradcam = [50, 51, 94, 100, 100, 100, 100]
    # random = [50, 50, 59, 62, 75, 99, 100]
    # subgraphx = [50, 52, 88, 99, 100, 99, 100]
    # pgexplainer = [50, 50, 93, 98, 100, 100, 100]

    ## BA-2Motifs GCN3l (node elimination)
    # gnnexplainer = [50, 51, 63, 71, 65, 70, 100]
    # gradcam = [50, 95, 99, 100, 100, 100, 100]
    # random = [50, 52, 59, 62, 61, 67, 100]
    # subgraphx = [50, 86, 94, 98, 100, 99, 100]
    # pgexplainer = [50, 66, 100, 95, 100, 100, 100]

    # BA-2Motifs GCN3l (both)
    # gnnexplainer = [50, 56, 51, 63, 94, 98, 100]
    # gradcam = [50, 50, 82, 100, 100, 100, 100]
    # random = [50, 52, 57, 63, 77, 99, 100]
    # subgraphx = [50, 55, 71, 88, 100, 100, 100]
    # pgexplainer = [50, 50, 83, 97, 100, 100, 100]

    # MUTAG GCN3l
    # gnnexplainer = [85, 84, 90, 90, 88, 85, 90]
    # gradcam = [85, 86, 87, 90, 90, 90, 90]
    # random = [85, 85, 83, 83.6, 83.6, 83.6, 90]
    # subgraphx = [85, 90, 90, 85, 90, 95, 90]
    # pgexplainer = [85, 85, 85, 80, 85, 90, 90]

    ## MUTAG GCN3l (node elimination)
    # gnnexplainer = [65, 85, 80, 90, 85, 80, 90]
    # gradcam = [65, 87, 90, 90, 90, 90, 90]
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
    # gnnexplainer = [65, 80, 85, 90, 84, 90, 90]
    # gradcam = [65, 80.5, 86, 80, 90, 90, 90]
    # random = [65, 75, 82.5, 82.5, 82.5, 85, 90]
    # subgraphx = [65, 70, 80, 80, 75, 80, 90]
    # pgexplainer = [65, 65, 75, 80, 85, 90, 90]

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
    # gnnexplainer = [65, 66, 74, 79, 80, 80, 100]
    # gradcam = [65, 75, 85, 85, 80, 80, 100]
    # random = [65, 65, 73, 79, 78, 80, 100]
    # subgraphx = [65, 65, 75, 75, 70, 80, 100]
    # pgexplainer = [65, 70, 80, 90, 90, 90, 100]

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
    # gnnexplainer = [50, 71, 70.5, 83.5, 94, 95, 93.5]
    # gradcam = [50, 63.5, 70, 87, 95, 95, 93.5]
    # random = [50, 68, 69, 83, 90.5, 93, 93.5]
    # # subgraphx = [69, 65, 70, 75, 75, 80, 93.5]
    # pgexplainer = [50, 67.5, 83, 93, 95, 94.5, 93.5]
    
    # REDDIT-BINARY GIN3l
    # gnnexplainer = [50, 90.5, 92.5, 93.5, 94, 94, 93.5]
    # gradcam = [50, 90, 93.5, 93, 94.5, 94, 93.5]
    # random = [50, 70.5, 87, 90.5, 91, 93, 93.5]
    # # subgraphx = [69, 65, 70, 75, 75, 80, 93.5]
    # pgexplainer = [50, 78, 80.5, 88.5, 93, 94.5, 93.5]

    # REDDIT-BINARY GIN3l (node elimination)
    # gnnexplainer = [50, 89.5, 92, 93.5, 94, 94, 93.5]
    # gradcam = [50, 90, 93.5, 93.5, 93, 94, 93.5]
    # random = [50, 70.5, 77, 90, 91, 94.5, 93.5]
    # # subgraphx = [69, 65, 70, 75, 75, 80, 93.5]
    # pgexplainer = [50, 80.5, 84.5, 88.5, 91.5, 94.5, 93.5]

    # REDDIT-BINARY GIN3l (skipped during evaluation)
    # gnnexplainer = [50, 50, 56, 63.5, 81, 89, 90.5]
    # gradcam = [50, 52, 62, 70.5, 82, 90, 90.5]
    # random = [50, 50, 55, 67, 81, 92, 90.5]
    # # subgraphx = [69, 65, 70, 75, 75, 80, 90.5]
    # pgexplainer = [50, 51, 66.5, 81, 89, 93, 90.5]

    # REDDIT-BINARY GIN3l (both)
    # gnnexplainer = [50, 55, 68.5, 70.5, 71, 89.5, 93.5]
    # gradcam = [50, 60.5, 70, 73, 82.5, 90, 93.5]
    # random = [50, 51, 59.5, 68, 87, 92, 93.5]
    # # subgraphx = [69, 65, 70, 75, 75, 80, 90.5]
    # pgexplainer = [50, 54.5, 69, 72, 87.5, 93, 93.5]

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
    # gnnexplainer = [33, 39.33, 38.67, 50.67, 66, 96, 98.6]
    # gradcam = [33, 40.67, 44, 70, 90, 97.33, 98.6]
    # random = [33, 43.33, 44.67, 51.33, 62.67, 94.67, 98.6]
    # subgraphx = [33, 34, 53.33, 72.67, 71.33, 80.67, 98.6]
    # pgexplainer = [33, 42.67, 36, 38.67, 55.33, 78.67, 98.6]
    
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
    # gnnexplainer = [33, 33.33, 33.33, 34.33, 44, 68.33, 95.33]
    # gradcam = [33, 33.33, 34.67, 58.67, 64.67, 98.67, 95.33]
    # random = [33, 33.33, 34, 33.33, 43.33, 48, 95.33]
    # subgraphx = [33, 38, 34.67, 37.33, 48, 65.33, 95.33]
    # pgexplainer = [33, 33.33, 33.33, 33.33, 33.33, 43.33, 95.33]

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
    # gnnexplainer = [16.67, 48.33, 56.67, 63.33, 65, 66.67, 75]
    # gradcam = [16.67, 55, 66.67, 65.5, 63.33, 66.67, 75]
    # random = [16.67, 46.67, 59.37, 61.67, 70, 71.67, 75]
    # subgraphx = [16.67, 48.7, 51.67, 62, 66.67, 71.67, 75]
    # pgexplainer = [16.67, 48.33, 56.67, 65, 68.33, 68.33, 75]

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
    # gnnexplainer = [16.67, 31.67, 40, 55, 65, 68.33, 75]
    # gradcam = [16.67, 41.67, 50, 56.67, 63.33, 68.33, 75]
    # random = [16.67, 33, 46, 52, 58, 68, 75]
    # subgraphx = [16.67, 36.67, 51.67, 60, 57, 70, 75]
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
    # gnnexplainer = [50, 66, 61, 68, 68, 68, 72]
    # gradcam = [50, 54, 56, 66, 67, 66, 72]
    # random = [50, 62.5, 60, 64, 65, 69.5, 72]
    # subgraphx = [50, 66, 64, 65, 63, 69, 72]
    # pgexplainer = [50, 67, 67, 66, 65, 71, 72]

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
    gnnexplainer = [50, 50, 50, 51, 62, 75, 80]
    gradcam = [50, 51, 51, 60, 73, 79, 80]
    random = [50, 49, 50, 50, 60, 78, 80]
    subgraphx = [50, 57, 50, 59, 66, 79, 80]
    pgexplainer = [50, 60, 46, 51, 69, 77, 80]

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

    # plt.title('KAR performance on REDDIT_BINARY-GIN3l')
    plt.figure(figsize=(5.6, 3.5))
    plt.xlabel("Edge Keep (KAR%)",fontsize=11)
    plt.ylabel("Acc (%)", fontsize=11)
    plt.plot(x, gnnexplainer)
    plt.scatter(x, gnnexplainer)
    
    plt.plot(x, gradcam)
    plt.scatter(x, gradcam)
    
    plt.plot(x, random)
    plt.scatter(x, random)

    plt.plot(x, subgraphx)
    plt.scatter(x, subgraphx)
    
    plt.plot(x, pgexplainer, color='#9467BD')
    plt.scatter(x, pgexplainer, color='#9467BD')

    # plt.plot(x, gradcam_gcn3l, color='darkblue')
    # plt.scatter(x, gradcam_gcn3l, color='darkblue')
    
    # plt.legend(['GradCAM', 'GradCAM (GCN3l)'])
    plt.rcParams.update({'font.size': 11})
    plt.xticks(fontsize=11) 
    plt.yticks(fontsize=11) 
    # plt.legend(['GNNExplainer', 'GradCAM', 'Random', 'SubgraphX', 'PGExplainer'])
    plt.legend(['GNNExplainer', 'GradCAM', 'Random', 'PGExplainer'])
    # plt.savefig('/home/alireza/Desktop/img.png', dpi=300)
    plt.savefig('/home/alireza/Desktop/kar_both.png', bbox_inches='tight', dpi=300)
