"""Module for extracting and plotting data about watchers"""

import json
import textwrap
import collections

import networkx as nx

from mininet import log


ALL_RESULTS_PATH = 'watchers/watchers.json'
PLOT_PATH = 'watchers/watchers.pdf'


def _buildGraph(topo):
    g = nx.Graph()
    g.add_nodes_from(n['name'] for n in topo['hosts'] + topo['switches'])
    tLinks = [e['target'] for e in topo['events']]
    for e in topo['links']:
        if e['name'] not in tLinks:
            g.add_edge(*e['hosts'], name = e['name'])
    # g.add_edges_from(e['hosts'] for e in topo['links'] if e['name'] not in tLinks)
    return g, tLinks


def _connectGraph(g, topo, edges):
    for e in topo['links']:
        if e['name'] in edges:
            g.add_edge(*e['hosts'], name = e['name'])
            # g.add_edges_from(e['hosts'] for e in topo['links'] if e['name'] in edges)


def _setMatches(watcherSets, graphSets, watcherPoint):
    wgs = None
    ogs = None
    for gs in graphSets:
        if watcherPoint in gs:
            wgs = gs
        else:
            ogs = gs

    if wgs is None or ogs is None:
        raise RuntimeError("Watcher point is not in graph")

    representatives = sorted([(v['representative']['rttavg'], k) for k, v in watcherSets.iteritems()])
    wgsColor = representatives[0][1]
    ogsColor = representatives[1][1]

    matches = {
        wgsColor: wgs,
        ogsColor: ogs
    }

    return matches

    # # return dict of color:graphSet for color in watcherSets.keys()
    # colors = watcherSets.keys()
    # emptyWatcherSets = [c for c in colors if len(watcherSets[c]) == 0]
    # cs = [c for c in colors if len(watcherSets[c]) > 0]
    # if len(emptyWatcherSets) > 1:
    # raise RuntimeError("More that one set to match is empty")
    # tm = {}
    #
    # # calculate affinities
    # for c in cs:
    # tm[c] = []
    # if len(watcherSets[c]) == 0:
    # continue
    # s = [p['address'] for p in watcherSets[c]]
    # for se, values in enumerate(graphSets):
    # o = float(len(set(values) & set(s)))
    # m = (o / len(values) + o / len(s)) /2
    # tm[c].append((m, se))
    # # highest values at the top of the list
    # tm[c].sort(reverse = True)
    #
    # # check that matches are all different
    # vals = [v[0][1] for v in tm.values()]
    # dup = [i for i, x in enumerate(vals) if vals.count(x) > 1]
    # # check for duplicate values
    # print tm
    # print 0, sorted(graphSets[0])
    # print 1, sorted(graphSets[1])
    # print sorted([p['address'] for p in watcherSets['white']])
    # exit()
    # if len(dup) > 0:
    # # keys = tm.keys()
    # # dup = [keys[i] for i in dup]
    # raise RuntimeError("Error while attributing matches : duplicates found")
    #
    # indexes = range(len(graphSets))
    # # finally return matches
    # matches = {}
    # for k, v in tm.iteritems():
    # bestMatch = v[0][1]
    # matches[k] = graphSets[bestMatch]
    # indexes.remove(bestMatch)
    #
    # if len(matches) != len(colors):
    # #assign remaining set to remaining match
    # if len(matches) + 1 == len(colors) and len(indexes) == 1:
    # color = [c for c in colors if c not in matches.keys()][0]
    #         matches[color] = graphSets[indexes[0]]
    #     else:
    #         raise RuntimeError("Some items were not matched.")
    # return matches


def _precision(watcherSet, graphSet):
    return float(len(set(watcherSet) & set(graphSet))) / len(watcherSet) if len(watcherSet) > 0 else 0


def _recall(watcherSet, graphSet):
    return float(len(set(watcherSet) & set(graphSet))) / len(graphSet)


def _getRecallAndPrecision(matches, watcher):
    stats = {}
    l = 0.0
    for color, part in matches.iteritems():
        # print watcher[color]
        waAddrs = [p['address'] for p in watcher[color]['hosts']]
        p = _precision(waAddrs, part)
        r = _recall(waAddrs, part)
        stats[color] = {'precision': p,
                        'recall': r}
        l += 1
    stats['total'] = {'recall': sum([c['recall'] for c in stats.values()]) / l if l > 0 else 0,
                      'precision': sum([c['precision'] for c in stats.values()]) / l if l > 0 else 0}

    a = (stats['total']['precision'] + stats['total']['recall'])
    stats['Fmeasure'] = 2 * stats['total']['precision'] * stats['total']['recall'] / a if a > 0 else 0
    return stats


def makeResults(watcher_output, topoFile):
    log.output("Making results from %s with topology %s\n" % (watcher_output, topoFile))
    nameToIp = {}
    ipToName = {}
    topo = json.load(open(topoFile))
    for h in topo['hosts']:
        nameToIp[h['name']] = h['options']['ip']
        ipToName[h['options']['ip']] = h['name']
    topoGraph, tLinks = _buildGraph(topo)
    watcher = json.load(open(watcher_output))
    out = watcher
    _makeSetResults(watcher, topoGraph, out, nameToIp)
    _connectGraph(topoGraph, topo, tLinks)
    _makeLinkResults(watcher, topoGraph, out, ipToName, nameToIp)
    out['tlinks'] = tLinks
    out['topoFile'] = topoFile
    return out


def _makeLinkResults(watcher, topoGraph, out, ipToName, nameToIp):
    reps = sorted([(v['representative']['rttavg'], k) for k, v in watcher['sets'].iteritems()])
    wgs = (ipToName[v['address']] for v in watcher['sets'][reps[0][1]]['hosts'])
    ogs = (ipToName[v['address']] for v in watcher['sets'][reps[1][1]]['hosts'])
    it = (wgs, 'minus'), (ogs, 'plus')
    w = watcher['watcher']
    for ps, sign in it:
        for p in ps:
            path = nx.shortest_path(topoGraph, source = w, target = p)
            edges = zip(path[1:], path[:-1])
            dist = 0
            for e in edges:
                d = topoGraph.get_edge_data(*e)
                if not d.has_key(sign):
                    d[sign] = 0
                d[sign] += 1
                if not d.has_key('distance'):
                    d['distance'] = dist
                topoGraph.add_edge(e[0], e[1], **d)
                dist += 1
    o = {}
    signs = zip(*it)[1]
    # print signs
    for edge in topoGraph.edges(data = True):
        data = edge[2]
        o[data['name']] = {sign: data[sign] for sign in signs if data.has_key(sign)}
        o[data['name']]['distance'] = data['distance'] if data.has_key('distance') else 0
    out['edges'] = o


def _makeSetResults(watcher, topoGraph, out, nameToIp):
    parts = []
    for part in nx.connected_components(topoGraph):
        parts.append([nameToIp[p] for p in part if nameToIp.has_key(p)])
    assert len(parts) == 2

    matches = _setMatches(watcher['sets'], parts, nameToIp[watcher['watcher']])

    precisionAndRecall = _getRecallAndPrecision(matches, watcher['sets'])
    out['precisionAndRecall'] = precisionAndRecall
    out['graph'] = parts
    out['totalProbes'] = sum(len(part) for part in parts)
    out['totalTestedProbes'] = sum(len(se['hosts']) for se in watcher['sets'].values()) + len(watcher['grey'])
    # return out


def appendResults(result, outFile = ALL_RESULTS_PATH):
    log.output("Adding results from %s to %s\n" % (json.dumps(result, indent = 4, separators = (',', ':')), outFile))
    import os

    if os.path.exists(outFile):
        res = json.load(open(outFile, 'r'))
    else:
        res = []
    res.append(result)
    with open(outFile, 'w') as f:
        json.dump(res, f)
        # makeGraphs(res)


def exclusive(parameterSet):
    ps = []

    for p in parameterSet:
        if sum(v for k, v in p.iteritems() if k in ('randomMetricWeight', 'ipMetricWeight', 'balancedMetricWeight', 'delayMetricWeight')) == 1:
            ps.append(p)

    return ps


def makeGraphs(results, plotPath = PLOT_PATH):
    pp = plotPath.replace(".pdf", "")
    log.output("Making new graph at %s\n" % pp)
    from graphs import Graph as g
    # from graphs import D3Graph as d3g

    linkplotter = LinkPlotter(g, pp + "-link.pdf", results)
    # setplotter = SetPlotter(g, pp + "-set.pdf", results)
    try:
        makeGraphsLinks(linkplotter)
        # makeGraphsGranularitySampleSize(plotter)
        # log.output("Making new graph at %s\n" % plotPath)
        # from graphs import Graph as g
        # # from graphs import D3Graph as d3g
        #
        # plotter = Plotter(g, plotPath, results)
        # # xSet = 99, 95
        # # granularitySet = 1
        # plotter.plotAllPlot(
        # variables  = {
        # 'x' : None
        # },
        # parameters = {
        #
        # },
        # grouping = {
        #
        # }
        # )
        # plotter.close()
    finally:
        linkplotter.close()


def makeGraphsLinks(plotter):
    metricSet = 0, 1
    bucketSet = 'probabilistic-bucket', 'ordered-bucket'
    selectionSet = exclusive,  # None
    sampleSizeSet = 0.1, 0.2, 0.3
    granularitySet = 1, 4

    def max(x):
        return plotter.np.percentile(x, 100)
    def percent99(x):
        return plotter.np.percentile(x, 99)
    def std(x):
        return plotter.np.std(x)
    def maxMinus1(x):
        return plotter.np.max(x) - 1
    def std2(x):
        return 2*plotter.np.std(x)
    def stdy(x):
        return 1.8 * plotter.np.std(x)

    # percentileSet = 70, 90, 95, 99, 100
    # selectionMethods = (lambda x: plotter.np.percentile(x, 100),
    #                     lambda x: plotter.np.percentile(x, 99),
    #                     lambda x: plotter.np.std(x),
    #                     lambda x: plotter.np.max(x) - 1
    # )
    selectionMethods = (
        max,
        percent99,
        std,
        std2,
        stdy,
        maxMinus1
    )
    # for selectionMethod in selectionMethods:
    for paramSelection in selectionSet:
        plotter.plotLinksMetricSelection(
            variables = {
                'sampleSize': [0.05, 1]
            },
            parameters = {
                'randomMetricWeight': metricSet,
                'ipMetricWeight': metricSet,
                'balancedMetricWeight': metricSet,
                'delayMetricWeight': metricSet,
                'bucket_type': bucketSet,
                'granularity': granularitySet
            },
            grouping = {
            },
            parameterSetSelection = paramSelection,
            electionMethod = selectionMethods
        )
    for paramSelection in selectionSet:
        for bucketType in bucketSet:
            for granularity in granularitySet:
                plotter.plotLinksMetric(
                    variables = {
                        'sampleSize': [0.05, 1]
                    },
                    parameters = {
                        'randomMetricWeight': metricSet,
                        'ipMetricWeight': metricSet,
                        'balancedMetricWeight': metricSet,
                        'delayMetricWeight': metricSet,
                    },
                    grouping = {
                        'bucket_type': bucketType,
                        'granularity': granularity
                    },
                    parameterSetSelection = paramSelection,
                    electionMethod = selectionMethods[0]
                )
    for paramSelection in selectionSet:
        for sampleSize in sampleSizeSet:
            for bucketType in bucketSet:
                for granularity in granularitySet:
                    plotter.plotLinksScores(
                        variables = {
                        },
                        parameters = {
                            'randomMetricWeight': metricSet,
                            'ipMetricWeight': metricSet,
                            'balancedMetricWeight': metricSet,
                            'delayMetricWeight': metricSet,
                        },
                        grouping = {
                            'bucket_type': bucketType,
                            'sampleSize': sampleSize,
                            'granularity': granularity
                        },
                        parameterSetSelection = paramSelection
                    )


def makeGraphsGranularitySampleSize(plotter):
    # xSet = 10, 20, 50, 100, 200
    x = 100
    # granularitySet = 0.1, 0.3, 0.5, 0.8, 1.0
    granularitySet = 1, 4
    metricSet = 0, 1
    selectionSet = exclusive,  # None
    sampleSizeSet = 0.01, 0.1, 0.2, 0.3
    bucketSet = 'probabilistic-bucket', 'ordered-bucket'
    # length of grey, precision + recall (per set and total) wrt delay variation
    # length of grey, precision + recall (per set and total) wrt granularity

    # paramSet = (balancedMetricWeight, ipMetricWeight, randomMetricWeight, delayMetricWeight, x, granularity)
    # print paramSet
    # print selection of result, then all results (None)
    for paramSelection in selectionSet:
        for granularity in granularitySet:
            for bucketType in bucketSet:
                log.output("Making new graph : variable is sampleSize, granularity : %s\n" % granularity)
                plotter.plotAllPlot(
                    variables = {
                        'sampleSize': None
                    },
                    parameters = {
                        'randomMetricWeight': metricSet,
                        'ipMetricWeight': metricSet,
                        'balancedMetricWeight': metricSet,
                        'delayMetricWeight': metricSet,
                    },
                    grouping = {
                        'granularity': granularity,
                        'x': x,
                        'bucket_type': bucketType
                    },
                    parameterSetSelection = paramSelection
                )
        for sampleSize in sampleSizeSet:
            for bucketType in bucketSet:
                # log.output("Making new graph : variable is granularity, x : %s\n" % x)
                plotter.plotFMeasurePlot(
                    variables = {
                        'granularity': None
                    },
                    parameters = {
                        'randomMetricWeight': metricSet,
                        'ipMetricWeight': metricSet,
                        'balancedMetricWeight': metricSet,
                        'delayMetricWeight': metricSet,
                    },
                    grouping = {
                        'sampleSize': sampleSize,
                        'x': x,
                        'bucket_type': bucketType
                    },
                    parameterSetSelection = paramSelection
                )
                # log.output("Making new graph : variable is sampleSize, granularity : %s\n" % granularity)
                # plotter.plotAllPlot(
                # variables = {
                # 'sampleSize': None
                # },
                # parameters = {
                # 'randomMetricWeight': metricSet,
                # 'ipMetricWeight': metricSet,
                # 'balancedMetricWeight': metricSet,
                # 'delayMetricWeight': metricSet,
                # 'bucket_type' : bucketSet
                # },
                # grouping = {
                # 'granularity': granularity,
                # 'x': x,
                # },
                #     parameterSetSelection = paramSelection
                # )


def makeGraphsOld(plotter):
    xSet = 10, 20, 50, 100, 200
    granularitySet = 0.1, 0.3, 0.5, 0.8, 1.0
    metricSet = 0, 1
    selectionSet = exclusive,  # None
    # length of grey, precision + recall (per set and total) wrt delay variation
    # length of grey, precision + recall (per set and total) wrt granularity

    # paramSet = (balancedMetricWeight, ipMetricWeight, randomMetricWeight, delayMetricWeight, x, granularity)
    # print paramSet
    # print selection of result, then all results (None)
    for paramSelection in selectionSet:
        for granularity in granularitySet:
            log.output("Making new graph : variable is x, granularity : %s\n" % granularity)
            plotter.plotAllPlot(
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
                },
                parameterSetSelection = paramSelection

            )
        for x in xSet:
            log.output("Making new graph : variable is granularity, x : %s\n" % x)
            plotter.plotAllPlot(
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
                    'x': x,
                },
                parameterSetSelection = paramSelection

            )


class Plotter(object):
    def __init__(self, graph, plotPath, data, d3 = None):
        self.gr = graph
        self.gr3d = d3
        self.data = data
        self.plotPath = plotPath
        self.pdf = self.preparePlot()
        import numpy as np

        self.np = np

    def preparePlot(self):
        from matplotlib.backends.backend_pdf import PdfPages

        self.alpha = 0.9
        log.output('Creating new graph at %s\n' % self.plotPath)
        return PdfPages(self.plotPath)


    def close(self):
        import datetime

        d = self.pdf.infodict()
        d['Title'] = 'Delays measurement'
        d['Author'] = u'Francois Espinet'
        d['Subject'] = 'Delay measurement'
        d['Keywords'] = 'measurement delays'
        d['ModDate'] = datetime.datetime.today()
        self.pdf.close()
        log.output('Graph successfully written to %s\n' % self.plotPath)

    @staticmethod
    def selectParams(exp, params, variables = None):
        # return true only if parameters of the experiments are the same as
        # given parameters
        p = exp['parameters']
        if variables is not None:
            d = dict(params.items() + variables.items())
        else:
            d = params
        for k, value in d.iteritems():
            if not p.has_key(k):
                return False
            if value is None:
                continue
            elif type(value) in (list, tuple):
                if len(value) == 2:
                    if value[0] <= p[k] <= value[1]:
                        continue
                    else:
                        return False
                if p[k] in value:
                    continue
                else:
                    return False
            elif p[k] != value:
                return False
        return True

    @staticmethod
    def printParams(params, addiionalString = None):
        out = []
        for k, v in params.iteritems():
            if type(v) in (tuple, list):
                o = repr(v)
            else:
                o = v
            out.append("%s=%s" % (k.replace("MetricWeight", ""), o))
        s = ", ".join(out)
        if addiionalString is not None:
            s = ", ".join((s, addiionalString))
        return s

    @staticmethod
    def printVariables(variables, addiionalString = None):
        out = []
        for v in variables.keys():
            if type(v) in (tuple, list):
                o = repr(v)
            else:
                o = v
            out.append("%s" % o)
        s = ", ".join(out)
        if addiionalString is not None:
            s = ", ".join((s, addiionalString))
        return s

    def wrapTitle(self, title):
        return "\n".join(textwrap.wrap(title, 60))

    def getParamSet(self, parameters):
        # convert dict of list to list of dicts
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


    def sort(self, array, row = 0):
        if array.size > 0:
            array = array[:, array[row].argsort()]
        return array


class LinkPlotter(Plotter):
    def getLinksScores(self, selector):

        exps = [exp for exp in self.data if self.selectParams(exp, selector)]
        if len(exps) == 0:
            return self.np.array([]), self.np.array([]), self.np.array([]), []
        tlinks = set()

        vals = self.np.array(
            [[self._sumLink(link) for link in exp['edges'].values()] for exp in exps]
        )
        vals = self.np.ma.masked_array(vals, self.np.isnan(vals))
        v = self.np.mean(vals, axis = 0)
        dev = self.np.std(vals, axis = 0)

        for exp in exps:
            tlinks.update(exp['tlinks'])

        r = self.np.array([
            [l['distance'] for l in exps[0]['edges'].values()],
            [l for l in exps[0]['edges'].keys()],
            v,
            dev
        ])
        self.sort(r)
        return r[1], r[2].astype(self.np.float32, copy = False), r[3].astype(self.np.float32, copy = False), tlinks

    def getLinksTopScores(self, *args):

        mkeys, scores, dev, percentile = args
        if len(mkeys) == 0 or len(scores) == 0 or len(dev) == 0:
            return mkeys, scores, dev

        sth = self.np.percentile(scores, 80)  # / 2.0

        keepIndexes = self.np.where(scores > sth)[0]
        keys = [l for i, l in enumerate(mkeys) if i in keepIndexes]
        r = self.np.array([
            keys,
            scores[keepIndexes],
            dev[keepIndexes]
        ])
        self.sort(r)

        return r[0], r[1], r[2]


    def _sumLink(self, link):
        p = 0
        m = 0
        if link.has_key('plus'):
            p += link['plus']
        if link.has_key('minus'):
            m -= link['minus']
        if p == 0 and m == 0:
            return float('nan')
        return p + m

    def getRawLinksMetric(self, variables, selector, electionMethod = None):

        variable = variables.keys()[0]
        f = {}
        if electionMethod is None:
            electionMethod = self.np.max
        for exp in self.data:
            if self.selectParams(exp, selector, variables):
                links = self.np.array([link for link in exp['edges'].iterkeys()])
                scores = self.np.array([self._sumLink(link) for link in exp['edges'].itervalues()])
                keepIndexes = self.np.where(self.np.logical_not(self.np.isnan(scores)))[0]
                topScore = electionMethod(scores[keepIndexes])
                scores = self.np.ma.masked_array(scores, self.np.isnan(scores))
                keepTopIndexes = self.np.ma.where((scores >= topScore))[0]
                linksScores = self.np.array([
                    links[keepTopIndexes],
                    scores[keepTopIndexes]
                ])
                candidates = linksScores[0]
                tlinks = exp['tlinks']
                s = 0
                for l in tlinks:
                    if l in candidates:
                        s += 1
                s = float(s) / len(candidates) if len(candidates) > 0 else 0
                if not f.has_key(exp['parameters'][variable]):
                    f[exp['parameters'][variable]] = []
                f[exp['parameters'][variable]].append(s)
        return f

    def getLinksMetric(self, variables, selector, electionMethod = None):

        f = self.getRawLinksMetric(variables, selector, electionMethod)
        x = []
        y = []
        dev = []
        err = []
        for k in sorted(f.keys()):
            v = f[k]
            x.append(k)
            m = self.np.mean(v)
            std = self.np.std(v)
            y.append(m)
            dev.append(std)
            err.append(std / m if m > 0 else 0)

        return self.np.array(x), self.np.array(y), self.np.array(dev), self.np.array(err)

    def plotLinksMetricSelection(self, variables = None, parameters = None, grouping = None, parameterSetSelection = None, electionMethod = None):
        log.output("Making new graph MetricSelection with variables : %s, parameters : %s, grouping : %s\n" % (
            self.printVariables(variables), self.printParams(parameters), self.printParams(grouping)))
        paramSets = self.getParamSet(parameters)
        if callable(parameterSetSelection):
            paramSets = parameterSetSelection(paramSets)
        if not isinstance(electionMethod, collections.Iterable):
            electionMethod = [electionMethod]

        for p in electionMethod:
            mets = {}
            # get all individual data for this percentile
            for paramSet in paramSets:
                selector = dict(grouping.items() + paramSet.items())
                d = self.getRawLinksMetric(variables, selector, p)
                for k, v in d.iteritems():
                    if not mets.has_key(k):
                        mets[k] = {'mean': [], 'std': []}
                    mets[k]['mean'].append(self.np.mean(v))
                    mets[k]['std'].append(self.np.std(v))
            stp = 'election: %s' % p.__name__
            x = []
            y = []
            dev = []
            err = []
            for k in sorted(mets.keys()):
                v = mets[k]
                x.append(k)
                m = self.np.mean(v['mean'])
                std = self.np.mean(v['std'])
                y.append(m)
                dev.append(std)
                err.append(std / m if m > 0 else 0)
            self.gr.subplot(2, 1, 1)
            x = self.np.array(x)
            self.gr.errorbar(x, self.np.array(y), yerr = self.np.array(dev),
                             marker = self.gr.getMarker(stp),
                             label = stp,
                             color = self.gr.getColor(stp), alpha = self.alpha
            )
            self.gr.hold = True
            self.gr.subplot(2, 1, 2)
            self.gr.plot(x, self.np.array(err),
                         marker = self.gr.getMarker(stp),
                         label = stp,
                         color = self.gr.getColor(stp), alpha = self.alpha
            )
            self.gr.hold = True

        self.gr.subplot(2, 1, 1)
        self.setMargins()
        self.gr.decorate(g_xlabel = self.printVariables(variables),
                         g_ylabel = "Metric",
                         g_grid = True,
                         g_title = self.wrapTitle("Aggregated metrics for %s" % (self.printVariables(variables))))
        self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))

        self.gr.subplot(2, 1, 2)
        self.setMargins()
        self.gr.decorate(g_xlabel = self.printVariables(variables),
                         g_ylabel = "Error",
                         g_grid = True,
                         g_title = self.wrapTitle("Aggregated error (avg(std)/avg(metrics)) for %s" % (self.printVariables(variables))))
        self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))

        fig = self.gr.gcf()
        fig.set_size_inches(15, 18)
        self.pdf.savefig(bbox_inches = 'tight')
        self.gr.close()


    def plotLinksMetric(self, variables = None, parameters = None, grouping = None, parameterSetSelection = None, electionMethod = None):
        log.output("Making new graph Metric with variables : %s, parameters : %s, grouping : %s\n" % (
            self.printVariables(variables), self.printParams(parameters), self.printParams(grouping)))
        paramSets = self.getParamSet(parameters)
        if callable(parameterSetSelection):
            paramSets = parameterSetSelection(paramSets)

        if not isinstance(electionMethod, collections.Iterable):
            electionMethod = [electionMethod]
        # plot all combination of parameters
        for paramSet in paramSets:
            for p in electionMethod:
                stp = 'selection = %s' % p.__name__
                selector = dict(grouping.items() + paramSet.items())
                x, metric, dev, relerr = self.getLinksMetric(variables, selector, p)
                self.gr.subplot(3, 1, 1)
                self.gr.errorbar(x, metric, yerr = dev,
                                 marker = self.gr.getMarker(str(paramSet) + stp),
                                 label = '%s' % self.printParams(paramSet, stp),
                                 color = self.gr.getColor(str(paramSet) + stp), alpha = self.alpha
                )
                # self.gr.hold = True
                self.gr.subplot(3, 1, 2)
                self.gr.plot(x, relerr,
                             marker = self.gr.getMarker(str(paramSet) + stp),
                             label = '%s' % self.printParams(paramSet, stp),
                             color = self.gr.getColor(str(paramSet) + stp), alpha = self.alpha
                )
                self.gr.subplot(3, 1, 3)
                self.gr.plot(x, dev,
                             marker = self.gr.getMarker(str(paramSet) + stp),
                             label = '%s' % self.printParams(paramSet, stp),
                             color = self.gr.getColor(str(paramSet) + stp), alpha = self.alpha
                )
        variable = variables.keys()[0]
        # grouping['percentile'] = percentile

        self.gr.subplot(3, 1, 1)
        self.setMargins()
        self.gr.decorate(g_xlabel = variable,
                         g_ylabel = "Metric",
                         g_grid = True,
                         g_title = self.wrapTitle("Metric for %s with %s " % (self.printVariables(variables), self.printParams(grouping))))
        self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))

        self.gr.subplot(3, 1, 2)
        self.setMargins()
        self.gr.decorate(g_xlabel = variable,
                         g_ylabel = "Relative incertitude",
                         g_grid = True,
                         g_title = self.wrapTitle(
                             "Relative incertitude (std/metric) for %s with %s " % (self.printVariables(variables), self.printParams(grouping))))
        self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))

        self.gr.subplot(3, 1, 3)
        self.setMargins()
        self.gr.decorate(g_xlabel = variable,
                         g_ylabel = "Absolute incertitude",
                         g_grid = True,
                         g_title = self.wrapTitle(
                             "Absolute incertitude for %s with %s " % (self.printVariables(variables), self.printParams(grouping))))
        self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))

        fig = self.gr.gcf()
        fig.set_size_inches(15, 25)
        self.pdf.savefig(bbox_inches = 'tight')  # 'checks/delay.pdf', format = 'pdf', )
        self.gr.close()

    def plotLinksScores(self, variables = None, parameters = None, grouping = None, parameterSetSelection = None):
        log.output("Making new graph LinksScores with variables : %s, parameters : %s, grouping : %s\n" % (
            self.printVariables(variables), self.printParams(parameters), self.printParams(grouping)))
        paramSets = self.getParamSet(parameters)
        if callable(parameterSetSelection):
            paramSets = parameterSetSelection(paramSets)
        mapping = {}
        i = 0
        percentile = 80
        xticks = set()
        xticksTop = set()
        for paramSet in paramSets:
            selector = dict(grouping.items() + paramSet.items())
            links, scores, dev, tlinks = self.getLinksScores(selector)
            for link in links:
                if not mapping.has_key(link):
                    mapping[link] = i
                    i += 1
            x = [mapping[link] for link in links]
            xticks.update(links)
            self.gr.subplot(3, 1, 1)
            self.gr.plot(x, scores,
                         label = '%s (target %s)' % (self.printParams(paramSet), repr(tlinks)),
                         color = self.gr.getColor(str(paramSet)), alpha = self.alpha
            )
            self.gr.subplot(3, 1, 2)
            self.gr.plot(x, dev,
                         label = '%s (target %s)' % (self.printParams(paramSet), repr(tlinks)),
                         color = self.gr.getColor(str(paramSet)), alpha = self.alpha
            )
            self.gr.subplot(3, 1, 3)
            topLinks, topScores, topDev = self.getLinksTopScores(links, scores, dev, percentile)
            topX = [mapping[link] for link in topLinks]
            xticksTop.update(topLinks)
            self.gr.plot(topX, topScores,
                         label = '%s (target %s)' % (self.printParams(paramSet), repr(tlinks)),
                         color = self.gr.getColor(str(paramSet)), alpha = self.alpha
            )

        xticks = zip(*sorted([(mapping[link], link) for link in xticks]))
        xticksTop = zip(*sorted([(mapping[link],link) for link in xticksTop]))
        self.gr.subplot(3, 1, 1)

        if len(xticks) == 2:
            self.gr.xticks(xticks[0],xticks[1], rotation = 90, fontsize = 8)
        self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
        self.gr.decorate(g_xlabel = 'links',
                         g_ylabel = "Link score",
                         g_grid = True,
                         g_title = self.wrapTitle("Link scores with %s" % self.printParams(grouping))
        )

        self.gr.subplot(3, 1, 2)
        if len(xticks) == 2:
            self.gr.xticks(xticks[0],xticks[1], rotation = 90, fontsize = 8)
        self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
        self.gr.decorate(g_xlabel = 'links',
                         g_ylabel = "Link score error",
                         g_grid = True,
                         g_title = self.wrapTitle("Error on scores with %s" % self.printParams(grouping))
        )

        self.gr.subplot(3, 1, 3)
        if len(xticksTop) == 2:
            self.gr.xticks(xticksTop[0],xticksTop[1], rotation = 90, fontsize = 8)
        self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
        self.gr.decorate(g_xlabel = 'links',
                         g_ylabel = "Link score",
                         g_grid = True,
                         g_title = self.wrapTitle("Link scores (top %s%% values) with %s" % (percentile, self.printParams(grouping)))
        )

        fig = self.gr.gcf()
        fig.set_size_inches(30, 30)
        self.pdf.savefig(bbox_inches = 'tight')  # 'checks/delay.pdf', format = 'pdf', )
        self.gr.close()


class SetPlotter(Plotter):
    def plotFMeasurePlot(self, grouping = None, **args):
        if grouping is None:
            grouping = {}

        self.graphAvgFMeasure(grouping = grouping, **args)
        self.setMargins()
        fig = self.gr.gcf()
        fig.set_size_inches(15, 10)
        self.pdf.savefig(bbox_inches = 'tight')  # 'checks/delay.pdf', format = 'pdf', )
        self.gr.close()

    def plotAllPlot(self, grouping = None, **args):
        if grouping is None:
            grouping = {}
        self.plotGrey(grouping = grouping, **args)
        fig = self.gr.gcf()
        fig.set_size_inches(15, 10)
        self.pdf.savefig(bbox_inches = 'tight')  # 'checks/delay.pdf', format = 'pdf', )
        self.gr.close()

        self.plotPrecisionAndRecall(grouping = grouping, **args)
        fig = self.gr.gcf()
        fig.set_size_inches(15, 25)
        self.pdf.savefig(bbox_inches = 'tight')  # 'checks/delay.pdf', format = 'pdf', )
        self.gr.close()

    def plotAllScatter(self, grouping = None, **args):
        if grouping is None:
            grouping = {}

        self.plotAllPlot(grouping = grouping, **args)

        self.scatterGrey(grouping = grouping, **args)
        fig = self.gr.gcf()
        fig.set_size_inches(15, 10)
        self.pdf.savefig(bbox_inches = 'tight')
        self.gr.close()

        self.scatterPrecisionAndRecall(grouping = grouping, **args)
        fig = self.gr.gcf()
        fig.set_size_inches(20, 20)
        self.pdf.savefig(bbox_inches = 'tight')
        self.gr.close()

    def getAvgFMeasure(self, variable, selector):
        f = {}
        for exp in self.data:
            if self.selectParams(exp, selector):
                if not f.has_key(exp['parameters'][variable]):
                    f[exp['parameters'][variable]] = []
                f[exp['parameters'][variable]].append(exp['precisionAndRecall']['Fmeasure'])
        x = f.keys()
        fmeasure = [self.np.mean(v) for v in f.values()]
        stdfmeasure = [self.np.std(v) for v in f.values()]
        d = self.np.array([
            x,
            fmeasure,
            stdfmeasure
        ])
        d = self.sort(d)
        x = d[0]
        fmeasure = d[1]
        err = d[2]
        return x, fmeasure, err

    def getFMeasure(self, variable, selector):

        d = self.np.array([
            [exp['parameters'][variable] for exp in self.data if self.selectParams(exp, selector)],
            [exp['precisionAndRecall']['Fmeasure'] for exp in self.data if self.selectParams(exp, selector)],
        ])
        d = self.sort(d)
        x = d[0]
        fmeasure = d[1]
        return x, fmeasure

    def scatterPrecisionAndRecall(self, **args):
        self.gr.subplot(2, 1, 1)
        self.graphTotalPrecisionAndRecall(grapher = self.gr.scatter, **args)
        self.gr.subplot(2, 1, 2)
        self.graphTotalFMeasure(grapher = self.gr.scatter, **args)


    def plotGrey(self, **args):
        self.graphGrey(grapher = self.gr.plot, **args)
        self.setMargins()

    def plotPrecisionAndRecall(self, **args):
        grs = 4
        i = 1
        self.gr.subplot(grs, 1, i)
        self.graphAvgFMeasure(**args)
        self.setMargins()
        i += 1
        self.gr.subplot(grs, 1, i)
        self.graphFMeasureError(grapher = self.gr.plot, **args)
        self.setMargins()
        i += 1
        self.gr.subplot(grs, 1, i)
        self.graphTotalPrecision(grapher = self.gr.plot, **args)
        self.setMargins()
        i += 1
        self.gr.subplot(grs, 1, i)
        self.graphTotalRecall(grapher = self.gr.plot, **args)
        self.setMargins()
        i += 1


    def graphTotalPrecisionAndRecall(self, grapher = None, variables = None, parameters = None, grouping = None, parameterSetSelection = None):
        if grapher is None or variables is None:
            return
        variable = variables.keys()[0]

        # plot precision & recall
        paramSets = self.getParamSet(parameters)
        if callable(parameterSetSelection):
            paramSets = parameterSetSelection(paramSets)

        # plot all combination of parameters
        for paramSet in paramSets:

            selector = dict(grouping.items() + paramSet.items())
            x, precision, recall = self.getTotalPrecisionAndRecall(variable, selector)

            grapher(x, precision, marker = 'd', label = 'Total Precision for %s' % self.printParams(paramSet),
                    color = self.gr.getColor(str(paramSet)), alpha = self.alpha)
            self.gr.hold = True
            grapher(x, recall, marker = '^', label = 'Total Recall for %s' % self.printParams(paramSet),
                    color = self.gr.getColor(str(paramSet)), alpha = self.alpha)
            self.gr.hold = True

        self.gr.decorate(g_xlabel = variable,
                         g_ylabel = "Recall/Precision",
                         g_grid = True,
                         g_title = self.wrapTitle(
                             "Total Recall and precision for %s with %s" % (self.printVariables(variables), self.printParams(grouping))))
        self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))

    def graphTotalPrecision(self, grapher = None, variables = None, parameters = None, grouping = None, parameterSetSelection = None):
        if grapher is None or variables is None:
            return
        variable = variables.keys()[0]

        # plot precision & recall
        paramSets = self.getParamSet(parameters)
        if callable(parameterSetSelection):
            paramSets = parameterSetSelection(paramSets)

        # plot all combination of parameters
        for paramSet in paramSets:

            selector = dict(grouping.items() + paramSet.items())
            x, precision, recall = self.getTotalPrecisionAndRecall(variable, selector)

            grapher(x, precision, marker = 'd', label = 'Total Precision for %s' % self.printParams(paramSet),
                    color = self.gr.getColor(str(paramSet)), alpha = self.alpha)
            self.gr.hold = True

        self.gr.decorate(g_xlabel = variable,
                         g_ylabel = "Precision",
                         g_grid = True,
                         g_title = self.wrapTitle("Total Precision for %s with %s" % (self.printVariables(variables), self.printParams(grouping))))
        self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))

    def graphTotalRecall(self, grapher = None, variables = None, parameters = None, grouping = None, parameterSetSelection = None):
        if grapher is None or variables is None:
            return
        variable = variables.keys()[0]

        # plot precision & recall
        paramSets = self.getParamSet(parameters)
        if callable(parameterSetSelection):
            paramSets = parameterSetSelection(paramSets)

        # plot all combination of parameters
        for paramSet in paramSets:

            selector = dict(grouping.items() + paramSet.items())
            x, precision, recall = self.getTotalPrecisionAndRecall(variable, selector)

            grapher(x, recall, marker = '^', label = 'Total Recall for %s' % self.printParams(paramSet),
                    color = self.gr.getColor(str(paramSet)), alpha = self.alpha)
            self.gr.hold = True

        self.gr.decorate(g_xlabel = variable,
                         g_ylabel = "Recall",
                         g_grid = True,
                         g_title = self.wrapTitle("Total Recall for %s with %s" % (self.printVariables(variables), self.printParams(grouping))))
        self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))


    def graphPerSetPrecisionAndRecall(self, grapher = None, variables = None, parameters = None, grouping = None, parameterSetSelection = None):
        if grapher is None or variables is None:
            return
        variable = variables.keys()[0]

        # plot precision & recall
        paramSets = self.getParamSet(parameters)
        if callable(parameterSetSelection):
            paramSets = parameterSetSelection(paramSets)

        # plot all combination of parameters
        for paramSet in paramSets:

            selector = dict(grouping.items() + paramSet.items())
            for color in ('black', 'white'):
                x, p, r = self.getPrecisionAndRecall(variable, selector, color)
                grapher(x, p, marker = '>', label = 'Precision for set %s %s' % (color, self.printParams(paramSet)),
                        color = self.gr.getColor(color + str(paramSet)), alpha = self.alpha)
                self.gr.hold = True
                grapher(x, r, marker = '<', label = 'Recall for set %s %s' % (color, self.printParams(paramSet)),
                        color = self.gr.getColor(color + str(paramSet)), alpha = self.alpha)
                self.gr.hold = True

        self.gr.decorate(g_xlabel = variable,
                         g_ylabel = "Recall/Precision",
                         g_grid = True,
                         g_title = self.wrapTitle(
                             "Per set Recall and precision for %s with %s" % (self.printVariables(variables), self.printParams(grouping))))
        self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))


    def graphAvgFMeasure(self, variables = None, parameters = None, grouping = None, parameterSetSelection = None):
        variable = variables.keys()[0]

        paramSets = self.getParamSet(parameters)
        if callable(parameterSetSelection):
            paramSets = parameterSetSelection(paramSets)
        # plot Fmeasure
        for paramSet in paramSets:
            selector = dict(grouping.items() + paramSet.items())
            x, fmeasure, yerr = self.getAvgFMeasure(variable, selector)
            self.gr.errorbar(x, fmeasure,
                             yerr = yerr,
                             marker = self.gr.getMarker(),
                             label = "Fmeasure for %s" % self.printParams(paramSet),
                             color = self.gr.getColor(str(paramSet)), alpha = self.alpha)
            self.gr.hold = True
        self.gr.decorate(g_xlabel = variable,
                         g_ylabel = "Fmeasure",
                         g_grid = True,
                         g_title = self.wrapTitle("Avged Fmeasure for %s with %s" % (variable, self.printParams(grouping))))
        self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
        # self.gr.yscale('log')

        self.gr.draw()

    def graphFMeasureError(self, grapher = None, variables = None, parameters = None, grouping = None, parameterSetSelection = None):
        if grapher is None or variables is None:
            return
        variable = variables.keys()[0]

        paramSets = self.getParamSet(parameters)
        if callable(parameterSetSelection):
            paramSets = parameterSetSelection(paramSets)
        # plot Fmeasure
        for paramSet in paramSets:
            selector = dict(grouping.items() + paramSet.items())
            x, fmeasure, yerr = self.getAvgFMeasure(variable, selector)
            grapher(x, yerr,
                    marker = self.gr.getMarker(),
                    label = "Error for %s" % self.printParams(paramSet),
                    color = self.gr.getColor(str(paramSet)), alpha = self.alpha)
            self.gr.hold = True
        self.gr.decorate(g_xlabel = variable,
                         g_ylabel = "Std deviation",
                         g_grid = True,
                         g_title = self.wrapTitle(
                             "Std deviation for FMeasure for %s with %s" % (self.printVariables(variables), self.printParams(grouping))))
        self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
        # self.gr.yscale('log')

        self.gr.draw()

    def graphTotalFMeasure(self, grapher = None, variables = None, parameters = None, grouping = None, parameterSetSelection = None):
        if grapher is None or variables is None:
            return
        variable = variables.keys()[0]

        paramSets = self.getParamSet(parameters)
        if callable(parameterSetSelection):
            paramSets = parameterSetSelection(paramSets)
        # plot Fmeasure
        for paramSet in paramSets:
            selector = dict(grouping.items() + paramSet.items())
            x, fmeasure = self.getFMeasure(variable, selector)
            grapher(x, fmeasure,
                    marker = self.gr.getMarker(),
                    label = "Fmeasure for %s" % self.printParams(paramSet),
                    color = self.gr.getColor(str(paramSet)), alpha = self.alpha)
            self.gr.hold = True
        self.gr.decorate(g_xlabel = variable,
                         g_ylabel = "Fmeasure",
                         g_grid = True,
                         g_title = self.wrapTitle("Fmeasure for %s with %s" % (self.printVariables(variables), self.printParams(grouping))))
        self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))

        self.gr.draw()

    def getGreys(self, variable, selector):

        d = self.np.array([
                              [exp['parameters'][variable] for exp in self.data if self.selectParams(exp, selector)],
                              [float(len(exp['grey'])) / exp['totalTestedProbes'] for exp in self.data if self.selectParams(exp, selector)],
                              [float(len(exp['grey'])) / exp['totalProbes'] for exp in self.data if self.selectParams(exp, selector)]
                          ],
        )

        self.sort(d)
        x = d[0]
        y1 = d[1]
        y2 = d[2]
        return x, y1, y2

    def scatterGrey(self, **args):
        self.graphGrey(grapher = self.gr.scatter, **args)

    def graphGrey(self, grapher = None, variables = None, parameters = None, grouping = None, parameterSetSelection = None):
        if grapher is None or variables is None:
            return
        variable = variables.keys()[0]

        paramSets = self.getParamSet(parameters)
        if callable(parameterSetSelection):
            paramSets = parameterSetSelection(paramSets)
        # plot all combination of parameters
        for paramSet in paramSets:
            selector = dict(grouping.items() + paramSet.items())
            x, y1, y2 = self.getGreys(variable, selector)
            grapher(x, y1, marker = 'd', label = 'grey probes' + self.printParams(paramSet),
                    color = self.gr.getColor(str(paramSet)), alpha = self.alpha)
            self.gr.hold = True

        self.gr.decorate(g_xlabel = variable,
                         g_ylabel = 'Ratio of grey probes',
                         g_grid = True,
                         g_title = self.wrapTitle(
                             "Ratio of grey probes (gp/testedProbes) for %s with %s" % (self.printVariables(variables), self.printParams(grouping))))
        self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
        self.gr.draw()


    def getTotalPrecisionAndRecall(self, variable, selector):

        d = self.np.array([
            [exp['parameters'][variable] for exp in self.data if self.selectParams(exp, selector)],
            [exp['precisionAndRecall']['total']['precision'] for exp in self.data if self.selectParams(exp, selector)],
            [exp['precisionAndRecall']['total']['recall'] for exp in self.data if self.selectParams(exp, selector)]
        ])
        d = self.sort(d)
        x = d[0]
        precision = d[1]
        recall = d[2]
        return x, precision, recall

    def getPrecisionAndRecall(self, variable, selector, color):

        d = self.np.array([
            [exp['parameters'][variable] for exp in self.data if self.selectParams(exp, selector)],
            [exp['precisionAndRecall'][color]['precision'] for exp in self.data if self.selectParams(exp, selector)],
            [exp['precisionAndRecall'][color]['recall'] for exp in self.data if self.selectParams(exp, selector)]
        ])
        d = self.sort(d)
        x = d[0]
        precision = d[1]
        recall = d[2]
        return x, precision, recall


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

