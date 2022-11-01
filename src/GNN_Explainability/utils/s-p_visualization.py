from turtle import color
import matplotlib.pyplot as plt

if __name__ == "__main__":
    x = [10, 30, 50, 70]
    
    # BA2Motifs GCN3l
    gnnexplainer = [39.91265, 39.91565, 39.91665, 39.907837]
    gradcam = [39.9191, 39.91880, 39.91765, 39.904606]
    subgraphx = [39.927657, 39.929657, 39.92765, 39.754754]
    pgexplainer = [39.91765, 39.91765, 39.91765, 39.915925]

    # BA3Motifs GCN3l
    gnnexplainer = [32.38997, 32.395926, 32.435822, 32.817695]
    gradcam = [33.055884, 33.05507, 33.055866, 33.09207]
    subgraphx = [32.489508, 33.04838, 32.474753, 33.160603]
    pgexplainer = [32.366025, 32.368323, 32.387793, 33.08161]

    plt.xlabel("Sparsity (%)")
    plt.ylabel("Fidelity (%)")
    plt.plot(x, gnnexplainer)
    plt.scatter(x, gnnexplainer)
    
    plt.plot(x, gradcam)
    plt.scatter(x, gradcam)
    

    plt.plot(x, subgraphx, color='#D62728')
    plt.scatter(x, subgraphx, color='#D62728')
    
    plt.plot(x, pgexplainer, color='#9467BD')
    plt.scatter(x, pgexplainer, color='#9467BD')

    plt.legend(['GNNExplainer', 'GradCAM', 'SubgraphX', 'PGExplainer'])
    plt.show()