"""Module for drawing graphs"""

import random

class _PyplotGraph(type):
    """Interface with the pyplot object"""

    def __new__(mcs, *args, **kwargs):
        #import pyplot and register  it
        import matplotlib.pyplot as plt

        mcs.plt = plt
        return type.__new__(mcs, *args, **kwargs)

    def __getattr__(cls, item):
        def decorate(*args, **kwargs):
            o = getattr(cls.plt, item)(*args, **cls.decorate(g_filter = True, **kwargs))
            cls.decorate(**kwargs)
            return o

        return decorate

    def decorate(cls, g_filter = False, g_grid = False, g_xtickslab = None, g_xticks = None,
                 g_xlabel = None, g_ylabel = None, g_title = None,
                 **kwargs):
        if g_filter:
            return kwargs
        ax = cls.plt.gca()
        if g_grid:
            ax.yaxis.grid(True, linestyle = '-', which = 'major', color = 'lightgrey', alpha = 0.5)
        if g_xlabel:
            cls.plt.xlabel(g_xlabel)
        if g_ylabel:
            cls.plt.ylabel(g_ylabel)
        if g_title:
            cls.plt.title(g_title)
        if g_xticks:
            cls.xticks(g_xticks)
        if g_xtickslab:
            ax.set_xticklabels(g_xtickslab)


class Graph(object):
    """Properties for making graphs and interface to graph object"""
    __metaclass__ = _PyplotGraph

    colors = ['b', 'g', 'c', 'm', 'y', 'k', 'aqua', 'blueviolet',
              'chartreuse', 'coral', 'crimson', 'darkblue',
              'darkslateblue', 'firebrick', 'forestgreen',
              'indigo', 'maroon', 'mediumblue', 'navy',
              'orange', 'orangered', 'purple', 'royalblue',
              'seagreen', 'slateblue', 'teal', 'tomato']

    @classmethod
    def getColor(cls, item = None):
        """Get a color for item
        returns random color if item is none
        :param item: hash item to get color
        """
        if item is None:
            return cls.colors[random.randint(0, len(cls.colors) - 1)]
        return cls.colors[hash(item) % len(cls.colors)]