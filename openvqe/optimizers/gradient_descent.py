from openvqe import numpy

# A very simple handwritten GradientDescent optimizer for demonstration purposes
class GradientDescent:

    def __init__(self, stepsize=0.1, save_energies=True, save_gradients=True):
        self.stepsize = stepsize
        self._energies = []
        self._gradients = []
        self.save_energies = save_energies
        self.save_gradients = save_gradients

    def __call__(self, angles, energy, gradient):
        if self.save_energies:
            self._energies.append(energy)
        if self.save_gradients:
            self._gradients.append(gradient)
        return [v - self.stepsize * gradient[i] for i, v in enumerate(angles)]

    def plot(self, plot_energies=True, plot_gradients=False, filename: str = None):
        from matplotlib import pyplot as plt
        if plot_energies:
            plt.plot(self._energies, label="E", color='b', marker='o', linestyle='--')
        if plot_gradients:
            gradients = [numpy.asarray(g).dot(numpy.asarray(g)) for g in self._gradients]
            plt.plot(gradients, label="dE", color='r', marker='o', linestyle='--')
        plt.legend()
        if filename is None:
            plt.show()
        else:
            plt.savefig("filename")