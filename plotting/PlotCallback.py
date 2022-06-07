import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras


class PlotCallback(keras.callbacks.Callback):

    def __init__(self, filename, loss_metrics, **kwargs):
        super(PlotCallback, self).__init__(**kwargs)
        self.filename = filename
        self.logs = {}
        self.loss_metrics = loss_metrics
        self.epoch = 0
        for metric in loss_metrics:
            self.logs[metric] = []

    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1
        for metric in self.loss_metrics:
            self.logs[metric].append(logs[metric])

    def plot(self):
        N = np.arange(0, self.epoch)
        plt.style.use("seaborn")
        plt.figure()
        for metric in self.loss_metrics:
            plt.plot(N, self.logs[metric], label=metric)
        plt.title('Training Losses')
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend()
        print(f'saving losses plot to {self.filename}')
        plt.savefig(self.filename)