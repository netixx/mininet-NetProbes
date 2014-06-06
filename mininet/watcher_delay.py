__author__ = 'francois'

import json

import networkx as nx
import textwrap

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
    stats['total'] = {'recall': sum([c['recall'] for c in stats.values()])/l if l > 0 else 0,
                      'precision': sum([c['precision'] for c in stats.values()])/l if l > 0 else 0}

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
    # length of grey, precision + recall (per set and total) wrt delay variation
    plotter.plotAll(x = None,
                    randomMetricWeight = 1,
                    ipMetricWeight = 1,
                    balancedMetricWeight = 1,
                    delayMetricWeight = 1,
                    granularity = 0.3)
    # length of grey, precision + recall (per set and total) wrt granularity
    plotter.plotAll(granularity = None,
                    x = 10,
                    randomMetricWeight = 1,
                    ipMetricWeight = 1,
                    balancedMetricWeight = 1,
                    delayMetricWeight = 1)
    # length of grey, precision + recall (per set and total) wrt static factors
    # plotter.plotAll(x = 10,
    # granularity = 0.3,
    #                 randomMetricWeight = None,
    #                 ipMetricWeight = None,
    #                 balancedMetricWeight = None,
    #                 delayMetricWeight = None)
    plotter.close()


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
    #     self.greys = np.array([len(exp['greys']) for exp in self.data])
    #     self.totalRecalls = np.array([exp['precisionAndRecall']['total']['recall'] for exp in self.data])
    #     self.totalPrecisions = np.array([exp['precisionAndRecall']['total']['precision'] for exp in self.data])

    def preparePlot(self):
        from matplotlib.backends.backend_pdf import PdfPages

        return PdfPages(self.plotPath)

    def plotAll(self, **variables):
        # for name, method in methods.iteritems():
        #     avgs = {}
        #     ts = {}
        #     steps = {'total': None}
        #     for target, tsteps in method['real_steps'].iteritems():
        #         st = zip(*tsteps)
        #         step_time = np.array((0,) + st[1])
        #         step_values = np.array((0,) + st[0])
        #         steps[target] = (step_time, step_values)
        #         steps['total'] = (step_time, np.minimum(steps['total'][1], step_values)) if steps['total'] is not None else (
        #             step_time, step_values)
        #
        #     for pair in method['pairs']:
        #         avg = map(lambda measure: measure.bw / (1000 ** 2), pair.measures)
        #
        #         t = map(lambda measure: measure.timestamp, pair.measures)
        #         avgs[pair.getPair()] = np.array(avg)
        #         ts[pair.getPair()] = np.array(t)
        #
        #
        #     # plot the data
        #     Graph.subplot(nlines, ncols, line)
        #     for pair in method['pairs']:
        #         Graph.scatter(ts[pair.getPair()], avgs[pair.getPair()],
        #                       color = Graph.getColor(pair.getPair()),
        #                       label = "%s,%s" % pair.getPair())
        #         Graph.hold = True
        #     for target, tsteps in steps.iteritems():
        #         Graph.step(tsteps[0], tsteps[1], 'r', where = 'post', label = target, color = Graph.getColor(target))
        #         Graph.hold = True
        #     Graph.decorate(g_xlabel = 'Time (s)',
        #                    g_ylabel = 'BW estimation with %s (Mbps)' % name)
        #     ax = Graph.gca()
        #     ax.set_yscale('log')
        #     Graph.legend(loc = 2)
        #     Graph.draw()
        #     line += ncols
        # fig = Graph.gcf()
        # fig.set_size_inches(20, 20)
        # pdf.savefig(bbox_inches = 'tight')  # 'checks/delay.pdf', format = 'pdf', )
        # Graph.close()
        self.plotGrey(**variables)
        fig = self.gr.gcf()
        fig.set_size_inches(20, 10)
        self.pdf.savefig(bbox_inches = 'tight')  # 'checks/delay.pdf', format = 'pdf', )
        self.gr.close()

        self.plotPrecisionAndRecall(**variables)
        fig = self.gr.gcf()
        fig.set_size_inches(10, 10)
        self.pdf.savefig(bbox_inches = 'tight')  # 'checks/delay.pdf', format = 'pdf', )
        self.gr.close()

        import datetime

        d = self.pdf.infodict()
        d['Title'] = 'Delays measurement'
        d['Author'] = u'Francois Espinet'
        d['Subject'] = 'Delay measurement'
        d['Keywords'] = 'measurement delays'
        d['ModDate'] = datetime.datetime.today()

    def close(self):
        self.pdf.close()

    @staticmethod
    def getVariables(params):
        return [k for k, v in params.iteritems() if v is None][0]

    @staticmethod
    def getParameters(params):
        return {k: v for k, v in params.iteritems() if v is not None}

    @staticmethod
    def selectParams(exp, params):
        # return true only if parameters of the experiments are the same as
        # given parameters
        p = exp['parameters']
        for k, value in params.iteritems():
            if not p.has_key(k):
                return False
            if p[k] != value:
                return False
        return True

    @staticmethod
    def printParams(params):
        out = []
        for k, v in params.iteritems():
            out.append("%s=%s"%(k,v))
        return ", ".join(out)

    def wrapTitle(self, title):
        return "\n".join(textwrap.wrap(title, 60))

    def plotPrecisionAndRecall(self, **variables):
        vars = self.getVariables(variables)
        params = self.getParameters(variables)
        import numpy as np

        x = np.array([exp['parameters'][vars] for exp in self.data if self.selectParams(exp, params)])
        precision = np.array([exp['precisionAndRecall']['total']['precision'] for exp in self.data if self.selectParams(exp, params)])
        recall = np.array([exp['precisionAndRecall']['total']['recall'] for exp in self.data if self.selectParams(exp, params)])
        #plot precision & recall
        self.gr.subplot(1, 2, 1)
        self.gr.scatter(x, precision, label = 'Total Precision', color = self.gr.getColor('total-precision'))
        self.gr.hold = True
        self.gr.scatter(x, recall, label = 'Total Recall', color = self.gr.getColor('total-recall'))
        self.gr.hold = True
        self.gr.decorate(g_xlabel = vars,
                         g_ylabel = "Recall/Precision",
                         g_grid = True,
                         g_title = self.wrapTitle("%s with %s"%(vars, self.printParams(params))))
        self.gr.legend(loc = 2)
        # plot Fmeasure
        self.gr.subplot(1, 2, 2)
        self.gr.scatter(x, np.array([exp['precisionAndRecall']['Fmeasure'] for exp in self.data if self.selectParams(exp, params)]),
                        label = "Fmeasure")
        self.gr.decorate(g_xlabel= vars,
                         g_ylabel= "Fmeasure",
                         g_grid = True)
        self.gr.legend(loc = 2)
        self.gr.draw()


    def plotGrey(self, **variables):
        vars = self.getVariables(variables)
        params = self.getParameters(variables)
        import numpy as np

        x = np.array([exp['parameters'][vars] for exp in self.data if self.selectParams(exp, params)])
        y1 = np.array([float(len(exp['grey'])) / exp['totalTestedProbes'] for exp in self.data if self.selectParams(exp, params)])
        y2 = np.array([float(len(exp['grey'])) / exp['totalProbes'] for exp in self.data if self.selectParams(exp, params)])
        self.gr.scatter(x, y1, label = 'grey probes / testedProbes', color = self.gr.getColor('testedgrey'))
        self.gr.hold = True
        self.gr.scatter(x, y2, label = 'grey probes / totalProbes', color = self.gr.getColor('totalgrey'))
        self.gr.decorate(g_xlabel = vars,
                         g_ylabel = 'Ratio of grey probes',
                         g_grid = True,
                         g_title = self.wrapTitle("%s with %s" % (vars, self.printParams(params))))
        self.gr.legend(loc = 2)
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

