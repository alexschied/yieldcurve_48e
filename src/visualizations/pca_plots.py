import matplotlib.pyplot as plt

def plot_scree(evr, path):
    plt.plot(evr.values, marker="o")
    plt.savefig(path)
    plt.close()
