__author__ = 'francois'

import json
import textwrap

import networkx as nx

from mininet import log


ALL_RESULTS_PATH = 'watchers/watchers.json'
PLOT_PATH = 'watchers/watcher.pdf'


def buildGraph(topo):
    g = nx.Graph()
    g.add_nodes_from([n['name'] for n in topo['hosts'] + topo['switches']])
    tLinks = [e['target'] for e in topo['events']]
    g.add_edges_from([e['hosts'] for e in topo['links'] if e['name'] not in tLinks])
    return g


def getBestMatch(availableSets, matchSet):
    match = 0
    j = 0
    for se, values in enumerate(availableSets):
        m = max(match, len(set(values) & set(matchSet)))
        if m > match:
            j = se
            match = m
    return j


def precision(watcherSet, graphSet):
    return float(len(set(watcherSet) & set(graphSet))) / len(watcherSet)


def recall(watcherSet, graphSet):
    return float(len(set(watcherSet) & set(graphSet))) / len(graphSet)


def getRecallAndPrecision(matches, watcher):
    stats = {}
    l = 0.0
    for color, part in matches.iteritems():
        waAddrs = [p['address'] for p in watcher[color]]
        p = precision(waAddrs, part)
        r = recall(waAddrs, part)
        stats[color] = {'precision': p,
                        'recall': r}
        l += 1
    stats['total'] = {'recall': sum([c['recall'] for c in stats.values()]) / l if l > 0 else 0,
                      'precision': sum([c['precision'] for c in stats.values()]) / l if l > 0 else 0}

    stats['Fmeasure'] = 2 * stats['total']['precision'] * stats['total']['recall'] / (stats['total']['precision'] + stats['total']['recall'])
    return stats


def makeResults(watcher_output, topoFile):
    log.info("Making results from %s with topology %s\n" % (watcher_output, topoFile))
    nameToIp = {}
    topo = json.load(open(topoFile))
    for h in topo['hosts']:
        nameToIp[h['name']] = h['options']['ip']
    topoGraph = buildGraph(topo)
    parts = []
    for part in nx.connected_components(topoGraph):
        parts.append([nameToIp[p] for p in part if nameToIp.has_key(p)])
    assert len(parts) == 2
    watcher = json.load(open(watcher_output))
    colors = watcher['sets'].keys()
    matches = {}
    for c in colors:
        matches[c] = parts[getBestMatch(parts, [p['address'] for p in watcher['sets'][c]])]
    # black = getBestMatch(parts, watcher, 'black')
    # white = 1 - black

    # matches = {'black': parts[black],
    # 'white': parts[white]}

    precisionAndRecall = getRecallAndPrecision(matches, watcher['sets'])
    out = watcher
    out['precisionAndRecall'] = precisionAndRecall
    out['graph'] = parts
    out['totalProbes'] = sum(len(part) for part in parts)
    out['totalTestedProbes'] = sum(len(se) for se in watcher['sets'].values()) + len(watcher['grey'])
    return out


def appendResults(result, outFile = ALL_RESULTS_PATH):
    log.info("Adding results from %s to %s\n" % (result, outFile))
    import os

    if os.path.exists(outFile):
        res = json.load(open(outFile, 'r'))
    else:
        res = []
    res.append(result)
    with open(outFile, 'w') as f:
        json.dump(res, f)
    makeGraphs(res)


def makeGraphs(results, plotPath = PLOT_PATH):
    log.info("Making new graph at %s\n" % plotPath)
    from graphs import Graph as g

    plotter = Plotter(g, plotPath, results)
    xSet = 10, 50, 200
    granularitySet = 0.1, 0.5, 1.0
    metricSet = 0, 1
    # length of grey, precision + recall (per set and total) wrt delay variation
    # paramSet = (balancedMetricWeight, ipMetricWeight, randomMetricWeight, delayMetricWeight, x, granularity)
    # print paramSet
    for granularity in granularitySet:
        log.info("Making new graph : variable is x, granularity : %s\n" % granularity)
        plotter.plotAll(
            variables = {
                'x': None
            },
            parameters = {
                'randomMetricWeight': metricSet,
                'ipMetricWeight': metricSet,
                'balancedMetricWeight': metricSet,
                'delayMetricWeight': metricSet
            },
            grouping = {
                'granularity': granularity
            }

        )
    for x in xSet:
        log.info("Making new graph : variable is granularity, x : %s\n" % x)
        plotter.plotAll(
            variables = {
                'granularity': None
            },
            parameters = {
                'randomMetricWeight': metricSet,
                'ipMetricWeight': metricSet,
                'balancedMetricWeight': metricSet,
                'delayMetricWeight': metricSet
            },
            grouping = {
                'x': x
            }

        )

    # aggregate some results
    log.info("Making new graph : variable is granularity, x : %s\n" % repr(xSet))
    plotter.plotAverage(
        variables = {
            'granularity': None
        },
        parameters = {
            'randomMetricWeight': metricSet,
            'ipMetricWeight': metricSet,
            'balancedMetricWeight': metricSet,
            'delayMetricWeight': metricSet
        },
        grouping = {
            'x': xSet
        }

    )
    plotter.close()


# def plotAll(plotter, **variables):
# # log.info("Making graph for %s\n" % ", ".join(["%s:%s" % (p,v) for p,v in variables.iteritems()]))
# plotter.plotAll(**variables)
# # length of grey, precision + recall (per set and total) wrt granularity
# # plotter.plotAll(granularity = None,
# # x = 10,
#     #                 randomMetricWeight = 1,
#     #                 ipMetricWeight = 1,
#     #                 balancedMetricWeight = 1,
#     #                 delayMetricWeight = 1)
#     # length of grey, precision + recall (per set and total) wrt static factors
#     # plotter.plotAll(x = 10,
#     # granularity = 0.3,
#     #                 randomMetricWeight = None,
#     #                 ipMetricWeight = None,
#     #                 balancedMetricWeight = None,
#     #                 delayMetricWeight = None)


class Plotter(object):
    def __init__(self, graph, plotPath, data):
        self.gr = graph
        self.data = data
        self.plotPath = plotPath
        self.pdf = self.preparePlot()
        # self.prepareData()
        # print(json.dumps(data, indent = 4, separators = (',', ': ')))

    # def prepareData(self):
    # import numpy as np
    #
    # self.greys = np.array([len(exp['greys']) for exp in self.data])
    #     self.totalRecalls = np.array([exp['precisionAndRecall']['total']['recall'] for exp in self.data])
    #     self.totalPrecisions = np.array([exp['precisionAndRecall']['total']['precision'] for exp in self.data])

    def preparePlot(self):
        from matplotlib.backends.backend_pdf import PdfPages

        self.alpha = 0.9
        return PdfPages(self.plotPath)

    def plotAverage(self, variables, parameters = None, grouping = None):
        pass

    def plotAll(self, variables, parameters = None, grouping = None):
        if grouping is None:
            grouping = {}

        self.scatterGrey(variables, parameters, grouping)
        fig = self.gr.gcf()
        fig.set_size_inches(15, 10)
        self.pdf.savefig(bbox_inches = 'tight')  # 'checks/delay.pdf', format = 'pdf', )
        self.gr.close()

        self.scatterPrecisionAndRecall(variables, parameters, grouping)
        fig = self.gr.gcf()
        fig.set_size_inches(15, 20)
        self.pdf.savefig(bbox_inches = 'tight')  # 'checks/delay.pdf', format = 'pdf', )
        self.gr.close()

        self.plotGrey(variables, parameters, grouping)
        fig = self.gr.gcf()
        fig.set_size_inches(15, 10)
        self.pdf.savefig(bbox_inches = 'tight')  # 'checks/delay.pdf', format = 'pdf', )
        self.gr.close()

        self.plotPrecisionAndRecall(variables, parameters, grouping)
        fig = self.gr.gcf()
        fig.set_size_inches(15, 20)
        self.pdf.savefig(bbox_inches = 'tight')  # 'checks/delay.pdf', format = 'pdf', )
        self.gr.close()


    def close(self):
        import datetime

        d = self.pdf.infodict()
        d['Title'] = 'Delays measurement'
        d['Author'] = u'Francois Espinet'
        d['Subject'] = 'Delay measurement'
        d['Keywords'] = 'measurement delays'
        d['ModDate'] = datetime.datetime.today()
        self.pdf.close()

    # @staticmethod
    # def getVariables(params):
    #     return [k for k, v in params.iteritems() if v is None][0]
    #
    # @staticmethod
    # def getParameters(params):
    #     return {k: v for k, v in params.iteritems() if v is not None}

    @staticmethod
    def selectParams(exp, params):
        # return true only if parameters of the experiments are the same as
        # given parameters
        p = exp['parameters']
        for k, value in params.iteritems():
            if not p.has_key(k):
                return False
            if type(value) in (list, tuple):
                if p[k] in value:
                    continue
                if len(value) == 2 and value[0] < p[k] < value[1]:
                    continue
            elif p[k] != value:
                return False
        return True

    @staticmethod
    def printParams(params):
        out = []
        for k, v in params.iteritems():
            if type(v) in (tuple, list):
                o = repr(v)
            else:
                o = v
            out.append("%s=%s" % (k.replace("MetricWeight", ""), o))
        return ", ".join(out)

    def wrapTitle(self, title):
        return "\n".join(textwrap.wrap(title, 60))

    def getParamSet(self, parameters):
        #convert dict of list to list of dicts
        vals = parameters.values()
        keys = parameters.keys()
        import itertools

        pvals = list(itertools.product(*vals))
        return [{keys[i]: v[i] for i in range(len(v))} for v in pvals]

    def setMargins(self):
        margins = list(self.gr.margins())
        if margins[0] < 0.05:
            margins[0] = 0.05
        if margins[1] < 0.05:
            margins[1] = 0.05
        # self.gr.margins(x = 0.05, y = 0.05)
        self.gr.margins(*margins)

    def getPrecisionAndRecall(self, vars, selector):
        import numpy as np

        x = np.array([exp['parameters'][vars] for exp in self.data if self.selectParams(exp, selector)])
        precision = np.array([exp['precisionAndRecall']['total']['precision'] for exp in self.data if self.selectParams(exp, selector)])
        recall = np.array([exp['precisionAndRecall']['total']['recall'] for exp in self.data if self.selectParams(exp, selector)])
        return x, precision, recall

    def getFMeasure(self, vars, selector):
        import numpy as np

        x = np.array([exp['parameters'][vars] for exp in self.data if self.selectParams(exp, selector)])
        fmeasure = np.array([exp['precisionAndRecall']['Fmeasure'] for exp in self.data if self.selectParams(exp, selector)])
        return x, fmeasure

    def scatterPrecisionAndRecall(self, variables, parameters = None, grouping = None):
        self.gr.subplot(2, 1, 1)
        self.graphPrecisionAndRecall(self.gr.scatter, variables, parameters = parameters, grouping = grouping)
        self.gr.subplot(2, 1, 2)
        self.graphFMeasure(self.gr.scatter, variables, parameters = parameters, grouping = grouping)


    def plotGrey(self, variables, parameters = None, grouping = None):
        self.graphGrey(self.gr.plot, variables, parameters = parameters, grouping = grouping)
        self.setMargins()

    def plotPrecisionAndRecall(self, variables, parameters = None, grouping = None):
        self.gr.subplot(2, 1, 1)
        self.graphPrecisionAndRecall(self.gr.plot, variables, parameters = parameters, grouping = grouping)
        self.setMargins()
        self.gr.subplot(2, 1, 2)
        self.graphFMeasure(self.gr.plot, variables, parameters = parameters, grouping = grouping)
        self.setMargins()

    def graphPrecisionAndRecall(self, grapher, variables, parameters = None, grouping = None):
        vars = variables.keys()[0]

        # plot precision & recall
        paramSets = self.getParamSet(parameters)
        #plot all combination of parameters
        for paramSet in paramSets:
            selector = dict(grouping.items() + paramSet.items())
            x, precision, recall = self.getPrecisionAndRecall(vars, selector)
            # x = np.array([exp['parameters'][vars] for exp in self.data if self.selectParams(exp, selector)])
            # precision = np.array([exp['precisionAndRecall']['total']['precision'] for exp in self.data if self.selectParams(exp, selector)])
            # recall = np.array([exp['precisionAndRecall']['total']['recall'] for exp in self.data if self.selectParams(exp, selector)])

            grapher(x, precision, marker = 'd', label = 'Total Precision for %s' % self.printParams(paramSet),
                    color = self.gr.getColor(str(selector)), alpha = self.alpha)
            self.gr.hold = True
            grapher(x, recall, marker = '^', label = 'Total Recall for %s' % self.printParams(paramSet),
                    color = self.gr.getColor(str(selector)), alpha = self.alpha)
            self.gr.hold = True

        self.gr.decorate(g_xlabel = vars,
                         g_ylabel = "Recall/Precision",
                         g_grid = True,
                         g_title = self.wrapTitle("Recall and precision for %s with %s" % (vars, self.printParams(grouping))))
        self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))

    def graphFMeasure(self, grapher, variables, parameters = None, grouping = None):
        vars = variables.keys()[0]

        paramSets = self.getParamSet(parameters)
        # plot Fmeasure
        for paramSet in paramSets:
            selector = dict(grouping.items() + paramSet.items())
            x, fmeasure = self.getFMeasure(vars, selector)
            grapher(x, fmeasure,
                    marker = 'o',
                    label = "Fmeasure for %s" % self.printParams(paramSet),
                    color = self.gr.getColor(str(selector)), alpha = self.alpha)
            self.gr.hold = True
        self.gr.decorate(g_xlabel = vars,
                         g_ylabel = "Fmeasure",
                         g_grid = True,
                         g_title = self.wrapTitle("Fmeasure for %s with %s" % (vars, self.printParams(grouping))))
        self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))

        self.gr.draw()

    def getGreys(self, vars, selector):
        import numpy as np

        x = np.array([exp['parameters'][vars] for exp in self.data if self.selectParams(exp, selector)])
        y1 = np.array([float(len(exp['grey'])) / exp['totalTestedProbes'] for exp in self.data if self.selectParams(exp, selector)])
        y2 = np.array([float(len(exp['grey'])) / exp['totalProbes'] for exp in self.data if self.selectParams(exp, selector)])
        return x, y1, y2

    def scatterGrey(self, variables, parameters = None, grouping = None):
        self.graphGrey(self.gr.scatter, variables, parameters = parameters, grouping = grouping)

    def graphGrey(self, grapher, variables, parameters = None, grouping = None):
        vars = variables.keys()[0]

        paramSets = self.getParamSet(parameters)
        # plot all combination of parameters
        for paramSet in paramSets:
            selector = dict(grouping.items() + paramSet.items())
            x, y1, y2 = self.getGreys(vars, selector)
            grapher(x, y1, marker = 'd', label = 'grey probes / testedProbes for %s' % self.printParams(paramSet),
                    color = self.gr.getColor(str(selector)), alpha = self.alpha)
            self.gr.hold = True
            grapher(x, y2, marker = '^', label = 'grey probes / totalProbes for %s' % self.printParams(paramSet),
                    color = self.gr.getColor(str(selector)), alpha = self.alpha)
            self.gr.hold = True

        self.gr.decorate(g_xlabel = vars,
                         g_ylabel = 'Ratio of grey probes',
                         g_grid = True,
                         g_title = self.wrapTitle("Ratio of grey probes for %s with %s" % (vars, self.printParams(grouping))))
        self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
        self.gr.draw()


if __name__ == "__main__":
    log.setLogLevel('info')
    from argparse import ArgumentParser

    parser = ArgumentParser(description = "Options for starting the custom Mininet network builder")

    # emulation options
    parser.add_argument("--topo",
                        dest = 'tfile',
                        help = 'Topology to load for this simulation',
                        default = None)

    parser.add_argument('--watcher-output',
                        dest = 'watcher_output',
                        default = None,
                        help = "Path to the output file")

    parser.add_argument('--output',
                        dest = 'output',
                        default = ALL_RESULTS_PATH,
                        help = 'Path to the output file')

    parser.add_argument('--graphs',
                        dest = 'graphs',
                        default = None)

    args = parser.parse_args()
    if args.graphs is not None:
        makeGraphs(json.load(open(args.output, 'r')), args.graphs)
        exit()
    out = makeResults(args.watcher_output, args.tfile)
    appendResults(out, args.output)

