import matplotlib.pyplot as plt

if __name__ == "__main__":
    x = [0, 10, 30, 50, 70, 90, 100]
    
    # BA2Motifs GCN3l
    # gnnexplainer = [39.91265, 39.91565, 39.91665, 39.907837]
    # gradcam = [39.9191, 39.91880, 39.91765, 39.904606]
    # pgexplainer = [39.91765, 39.91765, 39.91765, 39.915925]
    # subgraphx = [39.927657, 39.929657, 39.92765, 39.754754]

    # # BA3Motifs GCN3l
    # gnnexplainer = [32.38997, 32.395926, 32.435822, 32.817695]
    # gradcam = [33.055884, 33.05507, 33.055866, 33.09207]
    # pgexplainer = [32.366025, 32.368323, 32.387793, 33.08161]
    # subgraphx = [32.489508, 33.04838, 32.474753, 33.160603]

    # BA3Motifs GCN3l ROAR-ABSOLUTE
    gnnexplainer = [0.0,31.1715,34.0556,33.0337,32.9873,32.9795,32.9794]
    gradcam = [0.0,52.2067,33.0351,32.9794,32.9793,32.9802,32.9794]
    random = [0.0,29.9732,33.6065,33.035,32.9894,32.9796,32.9794]
    pgexplainer = [0.0,26.2512,33.2365,33.0124,32.9814,32.9796,32.9794]
    subgraphx = [0.0, 41.3574, 33.1734, 33.0124, 32.988, 32.9798,32.9794]
    
    plt.xlabel("Sparsity (ROAR%)")
    plt.ylabel("Fidelity (Absolute%)")
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