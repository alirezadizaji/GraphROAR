import matplotlib.pyplot as plt

if __name__ == "__main__":
    x = [0, 10, 30, 50, 70, 90, 100]
    ## BA-2Motifs GIN3l
    # gnnexplainer = [50, 50, 50, 58, 98, 100, 100]
    # gradcam = [50, 50, 50, 100, 100, 100, 100]
    # random = [50, 53, 51, 52, 58, 67, 100]
    # subgraphx = [50, 50, 50, 96, 100, 100, 100]
    # pgexplainer = [50, 65, 93, 100, 98, 100, 100]

    ## BA-2Motifs GCN3l
    # gnnexplainer = [50, 50, 61, 68, 67, 81, 100]
    # gradcam = [50, 95, 99, 100, 100, 100, 100]
    # random = [50, 55, 55, 57, 70, 90, 100]
    # subgraphx = [50, 86, 95, 94, 100, 99, 100]
    # pgexplainer = [50, 67, 97, 100, 100, 100, 100]

    ## MUTAG GCN3l
    # gnnexplainer = [85, 85, 90, 90, 90, 85, 90]
    # gradcam = [85, 85, 85, 90, 90, 90, 90]
    # random = [85, 85, 85, 83.6, 83.6, 83.6, 90]
    # subgraphx = [85, 90, 90, 85, 90, 95, 90]
    # pgexplainer = [85, 85, 85, 80, 85, 90, 90]

    ## MUTAG GCN3l (skip eval)
    # gnnexplainer = [85, 85, 80, 80, 80, 80, 90]
    # gradcam = [85, 80, 80, 80, 80, 80, 90]
    # random = [85, 78.2, 82.5, 85, 85, 90, 90]
    # subgraphx = [85, 65, 80, 80, 80, 90, 90]
    # pgexplainer = [85, 75, 70, 85, 85, 90, 90]
    
    ## MUTAG GIN3l
    # gnnexplainer = [65, 75, 80, 80, 85, 95, 100]
    # gradcam = [65, 90, 100, 100, 95, 100, 100]
    # random = [65, 85, 85, 88.5, 83.6, 88.5, 100]
    # subgraphx = [65, 90, 90, 85, 90, 95, 100]
    # pgexplainer = [65, 95, 85, 80, 75, 100, 100]

    # MUTAG GIN3l (skip during evaluation)
    # gnnexplainer = [65, 65, 65, 80, 80, 80, 100]
    # gradcam = [65, 65, 75, 85, 85, 90, 100]
    # random = [65, 67.5, 70, 82.5, 82.5, 80, 100]
    # subgraphx = [65, 65, 70, 75, 75, 80, 100]
    # pgexplainer = [65, 70, 80, 80, 80, 80, 100]

    # REDDIT-BINARY GCN3l
    # gnnexplainer = [69, 82, 84, 90.5, 90, 93, 95]
    # gradcam = [69, 88, 90, 95, 94.5, 94, 95]
    # random = [69, 71, 80.5, 85.5, 90.5, 94.5, 95]
    # # subgraphx = [69, 65, 70, 75, 75, 80, 95]
    # pgexplainer = [69, 74.5, 72, 83, 85, 92, 95]

    # REDDIT-BINARY GCN3l (skipped during evaluation)
    gnnexplainer = [69, 68.5, 58, 65, 85.5, 93, 95]
    gradcam = [69, 71, 70, 71.5, 83.5, 91, 95]
    random = [69, 71, 80.5, 85.5, 90.5, 94.5, 95]
    # subgraphx = [69, 65, 70, 75, 75, 80, 95]
    pgexplainer = [69, 71.5, 69, 72.5, 70, 92.5, 95]
    
    # REDDIT-BINARY GIN3l
    gnnexplainer = [50, 88, 88.5, 86, 84, 93, 90.5]
    gradcam = [50, 89, 91, 91.5, 91.5, 92.5, 90.5]
    random = [50, 71, 83, 91, 93, 93.5, 90.5]
    # subgraphx = [69, 65, 70, 75, 75, 80, 90.5]
    pgexplainer = [50, 63.5, 80, 82, 93, 94, 90.5]

    # REDDIT-BINARY GIN3l (skipped during evaluation)
    gnnexplainer = [50, 88, 88.5, 86, 84, 93, 90.5]
    gradcam = [50, 89, 91, 91.5, 91.5, 92.5, 90.5]
    random = [50, 71, 83, 91, 93, 93.5, 90.5]
    # subgraphx = [69, 65, 70, 75, 75, 80, 90.5]
    pgexplainer = [50, 63.5, 80, 82, 93, 94, 90.5]

    plt.title('KAR performance on REDDIT-GIN3l')
    plt.xlabel("Edge Keep (KAR %)")
    plt.ylabel("Val Acc (%)")
    plt.plot(x, gnnexplainer)
    plt.scatter(x, gnnexplainer)
    
    plt.plot(x, gradcam)
    plt.scatter(x, gradcam)
    
    plt.plot(x, random)
    plt.scatter(x, random)

    # plt.plot(x, subgraphx)
    # plt.scatter(x, subgraphx)
    
    plt.plot(x, pgexplainer)
    plt.scatter(x, pgexplainer)
    
    # plt.legend(['GNNExplainer', 'GradCAM', 'Random', 'SubgraphX', 'PGExplainer'])
    plt.legend(['GNNExplainer', 'GradCAM', 'Random', 'PGExplainer'])
    plt.show()