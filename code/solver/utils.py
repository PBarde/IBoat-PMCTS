import sys
sys.path.append("../model/")
import numpy as np
from matplotlib.animation import FuncAnimation
import mpl_toolkits.axes_grid1
import matplotlib.widgets


class Hist:
    """
    Definition of the histogram class.
    """
    #: Lower bound of histogram
    MIN_REWARD = 0

    #: Upper bound of histogram
    MAX_REWARD = 1.5

    #: Number of bins in histogram
    N_BINS = 12

    #: Values of the thresholds between bins (lower and upper included)
    THRESH = np.arange(MIN_REWARD, MAX_REWARD + (MAX_REWARD - MIN_REWARD) / N_BINS, (MAX_REWARD - MIN_REWARD) / N_BINS)

    #: mean value of each bin (mean between upper and lower threshold of each bin)
    MEANS = np.mean(np.stack((THRESH[:-1], THRESH[1:]), axis=0), axis=0)

    def __init__(self, init=[]):
        if len(init) == 0:
            self.h = np.zeros(Hist.N_BINS, dtype=int)
        else:
            self.h = np.array(init, dtype=int)

    def add(self, value):
        """
        Adds the value to the corresponding bin. If value is lower than lowest (resp. highest) threshold
        then the value is added to the first (resp. last) bin.
        :param int value: value to be added to the histogram

        """
        i = 0
        for x in Hist.THRESH[1:-1]:
            if value > x:
                i = i + 1
            else:
                break
        self.h[i] += 1

    def get_mean(self):
        """
        Computes the mean value of the histogram

        :return float: mean value
        """
        summed = sum(self.h)
        if summed == 0:
            return 0
        else:
            return np.dot(self.h, Hist.MEANS) / summed

    def is_empty(self):
        return all(value == 0 for value in self.h)


class Player(FuncAnimation):
    """
    Class implementing interactive plot, inherits from
    `FuncAnimation <https://matplotlib.org/api/_as_gen/matplotlib.animation.FuncAnimation.html>`_ and must be
    used accordingly.
    """

    def __init__(self, fig, func, frames=None, init_func=None, fargs=None,
                 save_count=None, mini=0, maxi=100, pos=(0.125, 0.92), **kwargs):
        self.i = 0
        self.min=mini
        self.max=maxi
        self.runs = True
        self.forwards = True
        self.fig = fig
        self.func = func
        self.setup(pos)
        FuncAnimation.__init__(self,self.fig, self.func, frames=self.play(),
                                           init_func=init_func, fargs=fargs,
                                           save_count=save_count, **kwargs )

    def play(self):
        while self.runs:
            self.i = self.i+self.forwards-(not self.forwards)
            if self.i > self.min and self.i < self.max:
                yield self.i
            else:
                self.stop()
                yield self.i

    def start(self):
        self.runs=True
        self.event_source.start()

    def stop(self, event=None):
        self.runs = False
        self.event_source.stop()

    def forward(self, event=None):
        self.forwards = True
        self.start()
    def backward(self, event=None):
        self.forwards = False
        self.start()
    def oneforward(self, event=None):
        self.forwards = True
        self.onestep()
    def onebackward(self, event=None):
        self.forwards = False
        self.onestep()

    def onestep(self):
        if self.i > self.min and self.i < self.max:
            self.i = self.i+self.forwards-(not self.forwards)
        elif self.i == self.min and self.forwards:
            self.i+=1
        elif self.i == self.max and not self.forwards:
            self.i-=1
        self.func(self.i)
        self.fig.canvas.draw_idle()

    def setup(self, pos):
        playerax = self.fig.add_axes([pos[0],pos[1], 0.22, 0.04])
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(playerax)
        bax = divider.append_axes("right", size="80%", pad=0.05)
        sax = divider.append_axes("right", size="80%", pad=0.05)
        fax = divider.append_axes("right", size="80%", pad=0.05)
        ofax = divider.append_axes("right", size="100%", pad=0.05)
        self.button_oneback = matplotlib.widgets.Button(playerax, label='$\u29CF$')
        self.button_back = matplotlib.widgets.Button(bax, label='$\u25C0$')
        self.button_stop = matplotlib.widgets.Button(sax, label='$\u25A0$')
        self.button_forward = matplotlib.widgets.Button(fax, label='$\u25B6$')
        self.button_oneforward = matplotlib.widgets.Button(ofax, label='$\u29D0$')
        self.button_oneback.on_clicked(self.onebackward)
        self.button_back.on_clicked(self.backward)
        self.button_stop.on_clicked(self.stop)
        self.button_forward.on_clicked(self.forward)
        self.button_oneforward.on_clicked(self.oneforward)
