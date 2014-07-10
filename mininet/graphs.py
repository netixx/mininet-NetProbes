"""Module for drawing graphs"""

import random


class _PyplotGraph(type):
    """Interface with the pyplot object"""

    def __new__(mcs, *args, **kwargs):
        # import pyplot and register  it
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
                 g_xlabel = None, g_ylabel = None, g_title = None, g_xgrid = False, g_ygrid = False,
                 **kwargs):
        if g_filter:
            return kwargs
        ax = cls.plt.gca()
        if g_grid:
            g_xgrid = True
            g_ygrid = True
        if g_xgrid:
            ax.xaxis.grid(True, linestyle = '-', which = 'major', color = 'lightgrey', alpha = 0.4)
        if g_ygrid:
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

import collections
class Graph(object):
    """Properties for making graphs and interface to graph object"""
    __metaclass__ = _PyplotGraph

    colors = ['b', 'g', 'c', 'm', 'y', 'k', 'aqua', 'blueviolet', 'brown',
              'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'crimson',
              'darkblue', 'darkcyan', 'darkgrey', 'darkgreen', 'darkslateblue', 'darkgoldenrod', 'darkturquoise',
              'deeppink', 'dodgerblue', 'firebrick', 'forestgreen', 'fuchsia', 'green', 'greenyellow', 'hotpink',
              'indianred', 'indigo', 'lightseagreen', 'lightsalmon', 'limegreen', 'maroon', 'mediumaquamarine', 'mediumblue',
              'mediumvioletred', 'mediumslateblue','navy', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'purple', 'royalblue',
              'seagreen', 'slateblue','sienna', 'steelblue', 'teal', 'tomato']
    random.shuffle(colors)
    markers = ['^', 'd', 'o', 'v', '>', '<', 'p', 's', '*']

    @classmethod
    def getColor(cls, item = None):
        """Get a color for item
        returns random color if item is none
        :param item: hash item to get color
        """
        if item is None or not isinstance(item, collections.Hashable):
            return cls.colors[random.randint(0, len(cls.colors) - 1)]
        return cls.colors[hash(item) % len(cls.colors)]

    @classmethod
    def getMarker(cls, item = None):
        if item is None or not isinstance(item, collections.Hashable):
            return cls.markers[random.randint(0, len(cls.markers) - 1)]
        return cls.markers[hash(item) % len(cls.markers)]



class _D3PyplotGraph(type):
    """Interface with the pyplot object"""

    def __new__(mcs, *args, **kwargs):
        # import pyplot and register  it
        # import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        mcs.plt = Axes3D
        return type.__new__(mcs, *args, **kwargs)

    def __getattr__(cls, item):
        def decorate(*args, **kwargs):
            o = getattr(cls.plt, item)(*args, **cls.decorate(g_filter = True, **kwargs))
            cls.decorate(**kwargs)
            return o

        return decorate

    def decorate(cls, g_filter = False, g_grid = False, g_xtickslab = None, g_xticks = None,
                 g_xlabel = None, g_ylabel = None, g_title = None, g_xgrid = False, g_ygrid = False,
                 **kwargs):
        if g_filter:
            return kwargs
        ax = cls.plt.gca()
        if g_grid:
            g_xgrid = True
            g_ygrid = True
        if g_xgrid:
            ax.xaxis.grid(True, linestyle = '-', which = 'major', color = 'lightgrey', alpha = 0.4)
        if g_ygrid:
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

class D3Graph(object):
    """Properties for making graphs and interface to graph object"""
    __metaclass__ = _PyplotGraph

    colors = ['b', 'g', 'c', 'm', 'y', 'k', 'aqua', 'blueviolet', 'brown',
              'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'crimson',
              'darkblue', 'darkcyan', 'darkgrey', 'darkgreen', 'darkslateblue', 'darkgoldenrod', 'darkturquoise',
              'deeppink', 'dodgerblue', 'firebrick', 'forestgreen', 'fuchsia', 'green', 'greenyellow', 'hotpink',
              'indianred', 'indigo', 'lightseagreen', 'lightsalmon', 'limegreen', 'maroon', 'mediumaquamarine', 'mediumblue',
              'mediumvioletred', 'navy', 'olive', 'orange', 'orangered', 'orchid', 'purple', 'royalblue',
              'seagreen', 'slateblue', 'sienna', 'steelblue', 'teal', 'tomato']

    markers = ['^', 'd', 'o', 'v', '>', '<', 'p', 's', '*']

    @classmethod
    def getColor(cls, item = None):
        """Get a color for item
        returns random color if item is none
        :param item: hash item to get color
        """
        if item is None or not isinstance(item, collections.Hashable):
            return cls.colors[random.randint(0, len(cls.colors) - 1)]
        return cls.colors[hash(item) % len(cls.colors)]

    @classmethod
    def getMarker(cls, item = None):
        if item is None or not isinstance(item, collections.Hashable):
            return cls.markers[random.randint(0, len(cls.markers) - 1)]
        return cls.markers[hash(item) % len(cls.markers)]
