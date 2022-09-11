import matplotlib.pyplot as plt

if __name__ == "__main__":
    x = [0, 10, 30, 50, 70, 90, 100]

    # BA2Motifs-GIN3l
    # gnnexplainer = [100, 100, 65, 82, 98, 99, 50]
    # gradcam = [100, 57, 57, 50, 50, 53, 50]
    # random = [100, 100, 67, 99, 99, 100, 50]
    # subgraphx = [100, 59, 97, 71, 74, 50, 50]
    # pgexplainer = [100, 99, 96, 96, 95, 50, 50]

    # BA2Motifs-GCN3l
    # gnnexplainer = [100, 90, 70, 61, 57, 52, 50]
    # gradcam = [100, 99, 99, 98, 95, 93, 50]
    # random = [100, 88, 76, 58, 55, 54, 50]
    # subgraphx = [100, 89, 86, 80, 92, 100, 50]
    # pgexplainer = [100, 81, 69, 60, 61, 51, 50]
    
    # MUTAG-GCN3l
    # gnnexplainer = [90, 90, 90, 85, 90, 85, 85]
    # gradcam = [90, 90, 85, 85, 90, 90, 85]
    # random = [90, 85, 85, 80, 90, 90, 85]
    # subgraphx = [90, 90, 95, 90, 90, 90, 85]
    # pgexplainer = [90, 90, 90, 90, 90, 85, 85]

    # MUTAG-GCN3l (skip eval)
    # gnnexplainer = [90, 90, 90, 90, 90, 90, 85]
    # gradcam = [90, 85, 85, 85, 85, 85, 85]
    # random = [90, 90, 90, 85, 80, 70, 85]
    # subgraphx = [90, 90, 85, 85, 70, 70, 85]
    # pgexplainer = [90, 90, 90, 90, 90, 90, 85]

    # MUTAG-GIN3l
    # gnnexplainer = [100, 95, 90, 85, 80, 85, 65]
    # gradcam = [100, 90, 100, 100, 95, 95, 65]
    # random = [100, 85, 85, 80, 90, 90, 65]
    # subgraphx = [100, 80, 90, 85, 85, 80, 65]
    # pgexplainer = [100, 85, 95, 70, 75, 80, 65]

    # MUTAG-GIN3l (skip eval)
    # gnnexplainer = [100, 85, 80, 70, 75, 85, 65]
    # gradcam = [100, 80, 75, 75, 70, 70, 65]
    # random = [100, 87.5, 85, 80, 77.5, 82.5, 65]
    # subgraphx = [100, 80, 80, 80, 75, 65, 65]
    # pgexplainer = [100, 80, 85, 75, 75, 75, 65]

    # REDDIT-BINARY GCN3l
    # gnnexplainer = [95, 90.5, 87, 86.5, 80, 78.5, 69]
    # gradcam = [95, 93, 92.5, 94.5, 92.5, 93.5, 69]
    # random = [95, 93.25, 91.25, 87, 84.5, 73.75, 69]
    # # subgraphx = [95, 65, 70, 75, 75, 80, 69]
    # pgexplainer = [95, 92, 90.5, 88.5, 80.5, 84, 69]

    # REDDIT-BINARY GCN3l (skipped during evaluation)
    # gnnexplainer = [95, 89.5, 86, 66.5, 54.5, 50, 69]
    # gradcam = [95, 91, 81.5, 69.5, 72, 68, 69]
    # random = [95, 93.25, 91.25, 87, 84.5, 73.75, 69]
    # # subgraphx = [95, 65, 70, 75, 75, 80, 69]
    # pgexplainer = [95, 90.5, 83.5, 84.5, 77, 71, 69]
    

    # REDDIT-BINARY GIN3l
    gnnexplainer = [90.5, 89.5, 91.5, 85, 82, 80.5, 50]
    gradcam = [90.5, 88, 92, 91.5, 89, 78, 50]
    random = [90.5, 89.7, 92.5, 89.6, 87, 83.5, 50]
    # subgraphx = [90.5, 65, 70, 75, 75, 80, 50]
    pgexplainer = [90.5, 84.5, 76, 71, 79, 66.5, 50]

    # REDDIT-BINARY GIN3l (skipped during evaluation)
    # gnnexplainer = [90.5, 72, 85.5, 83, 51, 50, 50]
    # gradcam = [90.5, 70.5, 72, 70.5, 67.5, 74, 50]
    # random = [90.5, 83.5, 78.75, 69.5, 61.5, 55.75, 50]
    # # subgraphx = [90.5, 65, 70, 75, 75, 80, 50]
    # pgexplainer = [90.5, 72, 67.5, 60.5, 52.5, 51, 50]

    plt.title('ROAR performance on REDDIT-GIN3l')
    plt.xlabel("Edge Keep (ROAR %)")
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