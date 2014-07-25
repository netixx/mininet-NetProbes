"""Module for extracting and plotting data about watchers"""

import json
import textwrap
import collections
import itertools

import networkx as nx

from mininet import log


ALL_RESULTS_PATH = 'watchers/watchers.json'
PLOT_PATH = 'watchers/watchers.pdf'

GRAPH_ROOT = 's1'


def _buildGraph(topo, params):
    g = nx.Graph()
    for e in topo['events']:
        events.replaceParams(e, params)
    g.add_nodes_from(n['name'] for n in topo['hosts'] + topo['switches'])
    tLinks = [e['target'] for e in topo['events']]
    for e in topo['links']:
        if e['name'] not in tLinks:
            g.add_edge(*e['hosts'], name = e['name'])
    # g.add_edges_from(e['hosts'] for e in topo['links'] if e['name'] not in tLinks)
    return g, tLinks


def _connectGraph(g, topo, edges):
    r = {}
    for e in topo['links']:
        if e['name'] in edges:
            g.add_edge(*e['hosts'], name = e['name'])
            r[e['name']] = e['hosts']
            # g.add_edges_from(e['hosts'] for e in topo['links'] if e['name'] in edges)

    return r


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


def _precision(watcherSet, graphSet):
    return float(len(set(watcherSet) & set(graphSet))) / len(watcherSet) if len(watcherSet) > 0 else 0


def _recall(watcherSet, graphSet):
    return float(len(set(watcherSet) & set(graphSet))) / len(graphSet)


def _rand_index(matches, watcher):
    a = 0
    b = 0
    c = 0
    d = 0
    wa = [set(p['address'] for p in watcher[color]['hosts']) for color in matches.iterkeys()]
    # remove probe that are not in the watcher set
    gt = [[p for p in g if any(p in w for w in wa)] for g in matches.itervalues()]
    # gt = [set(p for p in g) for g in matches.itervalues()]
    S = itertools.chain(*gt)

    # add rest of the world
    # wa.append([p for g in gt for p in g if not any(p in w for w in wa)])

    t = 0
    for pair in itertools.combinations(S, 2):
        t += 1
        sameWa = False
        sameGt = False
        for w in wa:
            if pair[0] in w and pair[1] in w:
                # pair in in same set for watcher
                sameWa = True
                break
        for g in gt:
            if pair[0] in g and pair[1] in g:
                # pair is in same set for ground truth
                sameGt = True
                break
        # each elements of the pair belong to the same set in both partitions
        # print pair, sameWa, sameGt
        if sameWa and sameGt:
            a += 1
        # each element of the pair belong to the different sets in both partitions
        elif not sameWa and not sameGt:
            b += 1
        elif sameWa and not sameGt:
            c += 1
        elif not sameWa and sameGt:
            d += 1

    # return (a + b) / nCk(len(S), 2)
    den = float(a + b + c + d)
    # if den != t:
    # log.error("Error in rand measure")
    r = (a + b) / den if den != 0 else 0
    return r


def _getSetMetrics(matches, watcher):
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

    stats['randindex'] = _rand_index(matches, watcher)

    return stats


import events


def makeResults(watcher_output, topoFile, substParams):
    log.output("Making results from %s with topology %s\n" % (watcher_output, topoFile))
    nameToIp = {}
    ipToName = {}
    topo = json.load(open(topoFile))
    for h in topo['hosts']:
        nameToIp[h['name']] = h['options']['ip']
        ipToName[h['options']['ip']] = h['name']
    topoGraph, tLinks = _buildGraph(topo, substParams)
    watcher = json.load(open(watcher_output))
    out = watcher
    out['substParams'] = substParams
    try:
        _makeSetResults(watcher, topoGraph, out, nameToIp)
        thosts = _connectGraph(topoGraph, topo, tLinks)
        _makeLinkResults(watcher, topoGraph, out, ipToName, GRAPH_ROOT)
        out['tlinkswdep'] = _depth_links(topoGraph, thosts)
    except:
        raise

    out['parameters']['depth'] = out['tlinkswdep'][0][0]
    out['tlinks'] = tLinks
    out['topoFile'] = topoFile
    return out


def _depth_links(topoGraph, tlinks, root = GRAPH_ROOT):
    l = []
    for link, hosts in tlinks.iteritems():
        l.append((_link_depth(topoGraph, hosts, root), link))

    return l


def _link_depth(topoGraph, link, root):
    return max(len(nx.shortest_path(topoGraph, source = root, target = h)) for h in link) - 1


def _makeLinkResults(watcher, topoGraph, out, ipToName, grRoot = GRAPH_ROOT):
    reps = sorted([(v['representative']['rttavg'], k) for k, v in watcher['sets'].iteritems()])
    wgs = (ipToName[v['address']] for v in watcher['sets'][reps[0][1]]['hosts'])
    ogs = (ipToName[v['address']] for v in watcher['sets'][reps[1][1]]['hosts'])
    it = (wgs, 'minus'), (ogs, 'plus')
    w = watcher['watcher']
    rootWatcherPath = set(nx.shortest_path(topoGraph, source = w, target = grRoot))
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
                if e in rootWatcherPath:
                    d['rootwatcher'] = True
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

    setMetrics = _getSetMetrics(matches, watcher['sets'])
    # for backward compatibility
    out['precisionAndRecall'] = setMetrics
    out['setStatistics'] = setMetrics
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


def ge1(parameterSet):
    ps = []

    for p in parameterSet:
        if sum(v for k, v in p.iteritems() if k in ('randomMetricWeight', 'ipMetricWeight', 'balancedMetricWeight', 'delayMetricWeight')) >= 1:
            ps.append(p)

    return ps


def makeGraphs(results, plotPath = PLOT_PATH):
    pp = plotPath.replace(".pdf", "")
    log.output("Making new graphs at %s\n" % pp)
    from graphs import Graph as g

    linkplotter = LinkPlotter(g, pp + "-link.pdf", results)

    try:
        makeGraphsLinks(linkplotter)
    finally:
        linkplotter.close()

    setplotter = SetPlotter(g, pp + "-set.pdf", results)
    try:
        makeGraphsSet(setplotter)
    finally:
        setplotter.close()


def makeGraphsLinks(plotter):
    metricSet = 0, 1
    metricSetBis = 0,

    def max(x):
        return plotter.np.percentile(x, 100)

    def std(x):
        return plotter.np.mean(x) + plotter.np.std(x)

    def maxMinus1(x):
        return plotter.np.max(x) - 1

    def std3(x):
        return plotter.np.mean(x) + 3 * plotter.np.std(x)

    def mean(x):
        return plotter.np.mean(x) + 4.8 * plotter.np.mean(x)


    metrics1 = (
        plotter.newMetric(name = 'naive 1/N', call = plotter.metricNaive),
        plotter.newMetric(name = 'proportional', call = plotter.metricProbabilistic),
        plotter.newMetric(name = 'argmax', call = plotter.metricGreater, kwargs = {'electionMethod': max})
    )

    metrics1bis = (
        plotter.newMetric(name = 'naive 1/depth', call = plotter.metricNaive, kwargs = {'assumeLocation': True}),
        plotter.newMetric(name = 'proportional', call = plotter.metricProbabilistic, kwargs = {'assumeLocation': True}),
        plotter.newMetric(name = 'argmax', call = plotter.metricGreater, kwargs = {'electionMethod': max,
                                                                                   'assumeLocation': True})
    )

    defaultMetric = plotter.newMetric(name = 'argmax', call = plotter.metricGreater, kwargs = {'electionMethod': max})
    # defaultMetric = plotter.newMetric(name = 'std3', kwargs = {'electionMethod': std3})

    # Compare selection metrics wrt sample size
    plotter.barLinksMetric(
        variables = {
            'sampleSize': None,
        },
        parameters = {
            'randomMetricWeight': metricSet,
            'ipMetricWeight': metricSet,
            'balancedMetricWeight': metricSet,
            'delayMetricWeight': metricSetBis
        },
        grouping = {
            # 'degree': ''
        },
        parameterSetSelection = ge1,
        metrics = defaultMetric,
        decoration = {
            'g_xlabel': "Number of measures",
            'g_title' : "Probability of finding target link in proposed links"
        }
    )

    bestSampleSize = 50
    # compare link selection policy for sampleSize = max(previous)
    plotter.compareMetrics(
        variables = {
            'sampleSize': bestSampleSize
            # 'sampleSize': None
        },
        grouping = {
            'sampleSize': bestSampleSize
        },
        metrics = metrics1,
        decoration = {
            'g_ylogscale': True,
            'g_xlabel' : "Number of measures",
            'g_title' : "Probability of finding target link in proposed links"
        }
    )

    # compare link selection policy for sampleSize = max(previous)
    plotter.compareMetrics(
        variables = {
            # 'sampleSize': bestSampleSize
            'sampleSize': None
        },
        grouping = {
            # 'sampleSize': bestSampleSize
        },
        metrics = metrics1bis,
        decoration = {
            'g_xlabel': "Number of measures",
            'g_title': "Probability of finding target link in candidates when located between the root and the probe"
        }
    )


    # compare best link selection policy wrt to depth for degree
    plotter.compareTopos(
        variables = {
            'depth': None
        },
        grouping = {
            'sampleSize': bestSampleSize
        },
        decoration = {
            'g_xlabel': "depth of problem",
            'g_title' : "Impact of topology over probability"
        },
        # metrics = defaultMetric
    )

    # ========================================================================================== #
    # extras
    # ========================================================================================== #
    plotter.bar3dLinksMetric(
        variables = {
            'depth': None,
            'sampleSize': None
        },
        parameters = {
            'randomMetricWeight': metricSet,
            'ipMetricWeight': metricSet,
            'balancedMetricWeight': metricSet,
            'delayMetricWeight': metricSetBis
        },
        grouping = {
        },
        parameterSetSelection = ge1,
        metrics = defaultMetric
    )

    metrics2 = (
        plotter.newMetric(name = 'max', kwargs = {'electionMethod': max}),
        plotter.newMetric(name = 'maxMinus1', kwargs = {'electionMethod': maxMinus1}),
        plotter.newMetric(name = 'mean5', kwargs = {'electionMethod': mean}),
        plotter.newMetric(name = 'std3', kwargs = {'electionMethod': std3}),
        plotter.newMetric(name = 'std', kwargs = {'electionMethod': std}),
    )

    # compare link selection policy for sampleSize = max(previous)
    plotter.compareMetrics(
        variables = {
            # 'sampleSize': bestSampleSize
            'sampleSize': None
        },
        grouping = {
            # 'sampleSize': bestSampleSize
        },
        metrics = metrics2
    )

    # Compare selection metrics wrt sample size
    # plotter.compareOrderSelectionMetrics(
    # variables = {
    #         'sampleSize': None,
    #     },
    #     parameters = {
    #         'randomMetricWeight': metricSet,
    #         'ipMetricWeight': metricSet,
    #         'balancedMetricWeight': metricSet,
    #         'delayMetricWeight': metricSetBis
    #     },
    #     grouping = {
    #         # 'degree': ''
    #     },
    #     parameterSetSelection = ge1,
    #     metrics = defaultMetric
    # )

    # Compare selection metrics (order) wrt sample size
    # plotter.cdfSelectionMetrics(
    #     variables = {
    #         'sampleSize': None,
    #     },
    #     parameters = {
    #         'randomMetricWeight': metricSet,
    #         'ipMetricWeight': metricSet,
    #         'balancedMetricWeight': metricSet,
    #         'delayMetricWeight': metricSetBis
    #     },
    #     grouping = {
    #         # 'degree': ''
    #     },
    #     parameterSetSelection = ge1,
    #     metrics = defaultMetric
    # )

    # Compare selection metrics (order) wrt sample size
    # plotter.compareOrderSelectionMetrics(
    #     variables = {
    #         'depth': None,
    #     },
    #     parameters = {
    #         'randomMetricWeight': metricSet,
    #         'ipMetricWeight': metricSet,
    #         'balancedMetricWeight': metricSet,
    #         'delayMetricWeight': metricSetBis
    #     },
    #     grouping = {
    #         # 'degree': ''
    #     },
    #     parameterSetSelection = ge1,
    #     metrics = defaultMetric
    # )

    # Compare selection metrics (order) wrt sample size
    # plotter.cdfSelectionMetrics(
    #     variables = {
    #         'depth': None,
    #     },
    #     parameters = {
    #         'randomMetricWeight': metricSet,
    #         'ipMetricWeight': metricSet,
    #         'balancedMetricWeight': metricSet,
    #         'delayMetricWeight': metricSetBis
    #     },
    #     grouping = {
    #         # 'degree': ''
    #     },
    #     parameterSetSelection = ge1,
    #     metrics = defaultMetric
    # )

    # check for best std factor
    def stdy(x, y):
        return plotter.np.mean(x) + y * plotter.np.std(x)

    from functools import partial

    metrics3 = [plotter.newMetric(name = 'std%s' % y, kwargs = {'electionMethod': partial(stdy, y = y)}) for y in (1, 2, 3, 3.5, 3.8, 4, 4.5, 5, 6)]

    # compare link selection policy for sampleSize = max(previous)
    plotter.compareMetrics(
        variables = {
            'sampleSize': None
            # 'depth': None
        },
        grouping = {
            # 'sampleSize': bestSampleSize
        },
        metrics = metrics3,
        decoration = {
            'title': 'Standard deviation factor selection'
        },
        annotate = False
    )

    # check for best mean factor
    def meany(x, y):
        return y * plotter.np.mean(x)

    metrics4 = [plotter.newMetric(name = 'mean%s' % (y / 10.0), kwargs = {'electionMethod': partial(meany, y = y / 10.0)}) for y in range(40, 60,
                                                                                                                                          2)]

    # compare link selection policy for sampleSize = max(previous)
    plotter.compareMetrics(
        variables = {
            'sampleSize': None
            # 'depth': None
        },
        grouping = {
            # 'sampleSize': bestSampleSize
        },
        metrics = metrics4,
        decoration = {
            'title': "Mean factor selection"
        },
        annotate = False
    )


def makeGraphsSet(plotter):
    # plotter.bar3dRandMeasure()
    plotter.barRandMeasure(
        variables = {
            'sampleSize': None
        },
        parameters = {
            "": {}
        },
        grouping = {

        }
    )


from collections import namedtuple


def convert(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


class Plotter(object):
    def __init__(self, graph, plotPath, data, d3 = None):
        self.gr = graph
        self.data = data
        self.plotPath = plotPath
        self.pdf = self.preparePlot()
        import numpy as np

        self.np = np

    def graphHeight(self, nGraphs = 1):
        return self._grHeight * nGraphs + self._titleHeight

    def preparePlot(self):
        from matplotlib.backends.backend_pdf import PdfPages

        self.alpha = 0.9
        self.alpha3d = 0.85
        self.alphaSurf = 0.7

        self.graphWidth = 10
        self._grHeight = 4
        self._titleHeight = 1.5
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
    def printParams(params, addiionalString = None, removeZeros = True):
        out = []
        for k, v in params.iteritems():
            if type(v) in (tuple, list):
                o = repr(v)
            else:
                o = v
            if removeZeros and v:
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

    def wrapLegend(self, legend):
        return "\n  ".join(textwrap.wrap(legend, 30))


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

    def jitter(self, array):
        import numpy as np

        if array.size <= 1:
            return array
        maxJitter = (array.max() - array.min()) / 100.0
        # print array
        jit = (np.random.random_sample((array.size,)) - 0.5) * 2 * maxJitter
        # array += jit
        return array + jit

    def printTitle(self, title, variables, grouping = None):
        return self.wrapTitle(
            title.format(variables = variables if len(variables) > 0 else "",
                         grouping = "with %s" % grouping if grouping is not None and len(grouping) > 0 else "")
        )

    def removeGraphScale(self, scaleDict):
        keys = sorted(scaleDict.keys())
        self.gr.decorate(
            g_xticks = [scaleDict[k] for k in keys],
            g_xtickslab = keys
        )

    def removeScale(self, scaleDict, values):
        o = []
        for i, value in enumerate(values):
            if not scaleDict.has_key(value):
                scaleDict[value] = len(scaleDict) + 1
            o.append(scaleDict[value])

        return self.np.array(o)

    def decorate(self, args, **kwargs):
        if args is not None:
            kwargs.update(args)

        self.gr.decorate(**kwargs)


def logGraph(func):
    def f(self, **kwargs):
        log.output("Making new graph %s with %s%s%s ... " % (
            func.__name__,
            "variables : %s" % self.printVariables(kwargs['variables']) if kwargs.has_key('variables') and len(kwargs['variables']) > 0 else "",
            ", parameters: %s" % self.printParams(kwargs['parameters']) if kwargs.has_key('parameters') and len(kwargs['parameters']) > 0 else "",
            ", grouping: %s" % self.printParams(kwargs['grouping']) if kwargs.has_key('grouping') and len(kwargs['grouping']) > 0 else "")
        )
        func(self, **kwargs)
        log.output("done\n\n")

    return f


class LinkPlotter(Plotter):
    def printMetrics(self, metrics):
        return " ,".join(m['name'] for m in metrics)

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

    def _getScores(self, edges, assumeLocation):
        if assumeLocation:
            links = self.np.array([link for link, data in edges.iteritems() if data.has_key('rootwatcher') and data['rootwatcher']])
            scores = self.np.array([self._sumLink(link) for link in edges.itervalues() if link.has_key('rootwatcher') and link['rootwatcher']])
        else:
            links = self.np.array([link for link in edges.iterkeys()])
            scores = self.np.array([self._sumLink(link) for link in edges.itervalues()])
        return links, scores

    def metricGreater(self, exp, variables, electionMethod = None, assumeLocation = False):
        if electionMethod is None:
            electionMethod = self.np.max

        links, scores = self._getScores(exp['edges'], assumeLocation)
        keepNumberIndexes = self.np.where(self.np.logical_not(self.np.isnan(scores)))[0]
        if len(keepNumberIndexes) < 1:
            return (), 0
        topScore = electionMethod(scores[keepNumberIndexes])
        scores = self.np.ma.masked_array(scores, self.np.isnan(scores))
        keepSelectedIndexes = self.np.ma.where((scores >= topScore))[0]
        linksScores = self.np.array([
            links[keepSelectedIndexes],
            scores[keepSelectedIndexes]
        ])
        candidates = set(linksScores[0])
        tlinks = exp['tlinks']
        s = 0
        for l in tlinks:
            if l in candidates:
                s += 1
        s = float(s) / len(candidates) if len(candidates) > 0 else 0
        key = convert({v: exp['parameters'][v] for v in variables.keys()})
        return key, s

    def metricNaive(self, exp, variables, assumeLocation = False):
        links, rscores = self._getScores(exp['edges'], assumeLocation)
        scores = len(rscores)
        scores = 1.0 / scores if scores > 0 else 0
        nlinks = len(exp['tlinks'])
        key = convert({v: exp['parameters'][v] for v in variables.keys()})
        return key, nlinks * scores


    def metricProbabilistic(self, exp, variables, assumeLocation = False):
        links, scores = self._getScores(exp['edges'], assumeLocation)
        keepIndexes = self.np.where(self.np.logical_and(self.np.logical_not(self.np.isnan(scores)), self.np.greater(scores, 0)))[0]

        keepLinks = links[keepIndexes]
        linksScores = scores[keepIndexes]
        s = 0
        tlinks = set(exp['tlinks'])
        totScores = float(sum(linksScores))
        if totScores > 0:
            for i, link in self.np.ndenumerate(keepLinks):
                if link in tlinks:
                    s += linksScores[i]
            s /= totScores
        else:
            s = 0
        key = convert({v: exp['parameters'][v] for v in variables.keys()})
        return key, s

    def getRawLinksMetric(self, variables, selector, metric = None):

        f = {}
        if metric is None:
            metric = self.newMetric()
        for exp in self.data:
            if self.selectParams(exp, selector, variables):
                key, s = metric['call'](exp, variables, **metric['kwargs'])
                if not f.has_key(key):
                    f[key] = []
                f[key].append(s)
        return f

    def getLinksMetric(self, variables, selector, metric = None):
        f = self.getRawLinksMetric(variables, selector, metric)
        return self._mergeRawData(f)

    def getSumMetricOrder(self, variables, selector, paramSets, metric = None):
        f = self.getRawMetricOrder(variables, selector, paramSets, metric = metric)
        f2 = {}
        for key, value in f.iteritems():
            vkeys = sorted(value.keys())
            f2[key] = (self.np.array(zip(*vkeys)), self.np.array([sum(value[k]) for k in vkeys]))
        return f2

    def getListMetricOrder(self, variables, selector, paramSets, metric = None):
        f = self.getRawMetricOrder(variables, selector, paramSets, metric = metric)
        f2 = {}
        for key, value in f.iteritems():
            vkeys = sorted(value.keys())
            f2[key] = (self.np.array(zip(*vkeys)), self.np.array([value[k] for k in vkeys]))
        return f2


    def getRawMetricOrder(self, variables, selector, paramSets, metric = None):
        f = {}
        if metric is None:
            metric = self.newMetric()
        for pset in paramSets:
            keyp = convert(pset)
            for exp in self.data:
                if self.selectParams(exp, dict(selector.items() + pset.items()), variables):
                    key, s = metric['call'](exp, variables, **metric['kwargs'])
                    if not f.has_key(key):
                        f[key] = {}
                    if not f[key].has_key(keyp):
                        f[key][keyp] = []
                    f[key][keyp].append(s)
                    # if self.selectParams(exp, selector.items(), dict(variables + pset.items())):
                    # key, s = metric['call'](exp, variables, **metric['kwargs'])
                    # if not f.has_key(key):
                    # f[key] = []
                    # f[key].append(s)
        # getattr(var, key)
        f2 = {}
        import operator

        for var, values in f.iteritems():
            sorted_var = sorted([(v, key) for key, value in values.iteritems() for v in value], key = operator.itemgetter(0))
            # print sorted_var
            # sorted_var = sorted(value.iteritems(), key = operator.itemgetter(1))
            s = 0
            currentScore = 0
            # tot = float(len(sorted_var))
            for i, (score, param) in enumerate(sorted_var):
                if not f2.has_key(param):
                    f2[param] = {}
                if not f2[param].has_key(var):
                    f2[param][var] = []
                if score > currentScore:
                    currentScore = score
                    s += 1
                f2[param][var].append(s)
        return f2

    def getTopoMetrics(self, variables, selector):
        import os

        metric = self.newMetric()
        f = {}
        for exp in self.data:
            if self.selectParams(exp, selector, variables):
                topo = os.path.basename(exp['topoFile']).replace(".json", "").replace('nox-', '')
                if not f.has_key(topo):
                    f[topo] = {}
                key, s = metric['call'](exp, variables, **metric['kwargs'])
                if not f[topo].has_key(key):
                    f[topo][key] = []
                f[topo][key].append(s)

        f2 = {}
        for topo, data in f.iteritems():
            f2[topo] = self._mergeRawData(data)

        return f2


    def _mergeRawData(self, data):
        y = []
        dev = []
        err = []
        num = []
        keys = sorted(data.keys())
        x = zip(*keys)
        for k in keys:
            v = data[k]
            # x.append(k)
            m = self.np.mean(v)
            std = self.np.std(v)
            y.append(m)
            dev.append(std)
            err.append(std / m if m > 0 else 0)
            num.append(len(v))

        return self.np.array(x), self.np.array(y), self.np.array(dev), self.np.array(err), self.np.array(num)

    def newMetric(self, name = 'max', call = None, kwargs = None):
        return {
            'name': name,
            'call': call if call is not None else self.metricGreater,
            'kwargs': kwargs if kwargs is not None else {}
        }

    @logGraph
    def compareMetrics(self, variables = None, grouping = None, metrics = None, annotate = True, decoration = None):
        if metrics is None:
            metrics = self.newMetric()

        if type(metrics) is dict:
            metrics = [metrics]
        plot = False
        basewidth = (0.95 / len(metrics))
        width = basewidth
        # offset = - basewidth*len(metrics)/2.0
        offset = 0
        scaleDict = {}
        for electionMethod in metrics:
            stp = 'selection = %s' % electionMethod['name']
            selector = dict(grouping.items())
            x, metric, dev, relerr, nsamples = self.getLinksMetric(variables, selector, metric = electionMethod)
            if len(x) > 0 and len(x[0]) > 0:
                x = self.removeScale(scaleDict, x[0])
                # if len(x) > 1:
                # width = (x[1:] - x[:-1]).min() * basewidth
                plot = True
                self.gr.subplot(1, 1, 1)
                # metric = self.np.log(metric)
                self.gr.bar(x + offset, metric, yerr = dev,
                            width = width,
                            label = stp,
                            color = self.gr.getColor(stp),
                            alpha = self.alpha,
                )
                if annotate:
                    for i, num in enumerate(nsamples):
                        self.gr.annotate(num,
                                         xy = (x[i] + offset + width / 2.0, metric[i] / 2.0),
                                         # textcoords = 'offset points',
                                         # xytext = (4, 4),
                                         ha = 'center',
                                         va = 'center'
                        )
                offset += width
        if plot:
            self.gr.subplot(1, 1, 1)
            self.removeGraphScale(scaleDict)
            self.decorate(decoration,
                          g_xlabel = self.printVariables(variables),
                          g_ylabel = "Metric value",
                          g_ygrid = True,
                          g_title = self.printTitle("Comparison of metrics %s wrt {variables} {grouping}" % self.printMetrics(metrics),
                                                    self.printVariables(variables),
                                                    self.printParams(grouping)),
                          # g_ylim =
            )
            self.setMargins()
            self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))

            fig = self.gr.gcf()
            fig.set_size_inches(self.graphWidth, self.graphHeight())
            self.pdf.savefig(bbox_inches = 'tight')  # 'checks/delay.pdf', format = 'pdf', )
            self.gr.close()


    @logGraph
    def cdfSelectionMetrics(self, variables = None, parameters = None, grouping = None, parameterSetSelection = None, electionMethod = None,
                            decoration = None):
        paramSets = self.getParamSet(parameters)
        if callable(parameterSetSelection):
            paramSets = parameterSetSelection(paramSets)

        if not isinstance(electionMethod, collections.Iterable):
            electionMethod = [electionMethod]
        plot = False
        graphs = {}
        for p in electionMethod:
            stp = 'selection = %s' % p.__name__
            f = self.getListMetricOrder(variables, grouping, paramSets)
            for tupleParamSet, values in f.iteritems():
                paramSet = tupleParamSet._asdict()
                x = values[0]
                if len(x) > 0 and len(x[0]) > 0:
                    x = x[0]
                    y = values[1]
                    plot = True
                    for i, xi in enumerate(x):
                        if not graphs.has_key(xi):
                            graphs[xi] = len(graphs) + 1
                        cgr = graphs[xi]
                        self.gr.subplot(len(graphs), 1, cgr)
                        s = self.np.sort(y[i])
                        s = self.jitter(s)
                        yvals = 1 - (self.np.arange(len(s)) + 0.5) / float(len(s))
                        self.gr.plot(s, yvals,
                                     color = self.gr.getColor(str(paramSet) + stp),
                                     label = self.wrapLegend('%s' % self.printParams(paramSet, stp))
                        )

        if plot:
            for xi, ngr in graphs.iteritems():
                self.gr.subplot(len(graphs), 1, ngr)
                self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
                self.decorate(decoration,
                              g_xlabel = "Metric score",
                              g_ylabel = "1 - CDF",
                              g_grid = True,
                              g_title = self.printTitle(
                                  "CDF of metric scores for {variables}=%s" % xi,
                                  self.printVariables(variables))
                )

            fig = self.gr.gcf()
            fig.set_size_inches(self.graphWidth, self.graphHeight(len(graphs)))
            self.pdf.savefig(bbox_inches = 'tight')
        self.gr.close()


    @logGraph
    def compareOrderSelectionMetrics(self, variables = None, parameters = None, grouping = None, parameterSetSelection = None, electionMethod = None,
                                     decoration = None):
        paramSets = self.getParamSet(parameters)
        if callable(parameterSetSelection):
            paramSets = parameterSetSelection(paramSets)

        if not isinstance(electionMethod, collections.Iterable):
            electionMethod = [electionMethod]
        plot = False
        basewidthfact = 0.9
        # basewidth = 1
        # width = basewidth
        offset = 0
        scaleDict = {}
        for p in electionMethod:
            stp = 'selection = %s' % p.__name__
            f = self.getSumMetricOrder(variables, grouping, paramSets)
            width = basewidthfact * (1.0 / len(f))
            for tupleParamSet, values in f.iteritems():
                paramSet = tupleParamSet._asdict()
                x = values[0]
                if len(x) > 0 and len(x[0]) > 0:
                    x = x[0]
                    y = values[1]
                    x = self.removeScale(scaleDict, x)
                    # width = (x[1:] - x[:-1]).min() * basewidth
                    plot = True
                    self.gr.subplot(1, 1, 1)
                    self.gr.bar(x + offset, y,
                                width = width,
                                label = self.wrapLegend('%s' % self.printParams(paramSet, stp)),
                                color = self.gr.getColor(str(paramSet) + stp),
                                alpha = self.alpha,
                                log = bool(decoration.get('g_ylogscale')) if decoration is not None else False
                    )
                offset += width
        if plot:
            self.gr.subplot(1, 1, 1)
            self.removeGraphScale(scaleDict)
            self.setMargins()
            self.decorate(decoration,
                          g_xlabel = self.printVariables(variables),
                          g_ylabel = "Metric Score (order)",
                          g_grid = True,
                          g_title = self.printTitle(
                              "Order of metrics for {variables} {grouping}", self.printVariables(variables), self.printParams(grouping))
            )
            self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))

            fig = self.gr.gcf()
            fig.set_size_inches(self.graphWidth, self.graphHeight())
            self.pdf.savefig(bbox_inches = 'tight')
        self.gr.close()

    @logGraph
    def barLinksMetric(self, variables = None, parameters = None, grouping = None, parameterSetSelection = None, metrics = None,
                       decoration = None):

        paramSets = self.getParamSet(parameters)
        if callable(parameterSetSelection):
            paramSets = parameterSetSelection(paramSets)

        if metrics is None:
            metrics = self.newMetric()

        if type(metrics) is dict:
            metrics = [metrics]
        plot = False
        width = (1.0 / len(paramSets))
        # width = basewidth
        offset = 0
        scaleDict = {}
        for paramSet in paramSets:
            for metric in metrics:
                stp = 'selection = %s' % metric['name']
                selector = dict(grouping.items() + paramSet.items())
                x, metric, dev, relerr, nsamples = self.getLinksMetric(variables, selector, metric = metric)
                if len(x) > 0:
                    x = self.removeScale(scaleDict, x[0])

                    # if len(x) > 1:
                    # width = (x[1:] - x[:-1]).min() * basewidth
                    plot = True
                    self.gr.subplot(3, 1, 1)
                    self.gr.bar(x + offset, metric, yerr = dev,
                                width = width,
                                # marker = self.gr.getMarker(str(paramSet) + stp),
                                label = self.wrapLegend('%s' % self.printParams(paramSet, stp)),
                                color = self.gr.getColor(str(paramSet) + stp),
                                alpha = self.alpha,
                                log = bool(decoration.get('g_ylogscale')) if decoration is not None else False
                    )
                    for i, num in enumerate(nsamples):
                        self.gr.annotate(num,
                                         xy = (x[i] + offset + width / 2.0, metric[i] / 2.0),
                                         # textcoords = 'offset points',
                                         # xytext = (4, 4),
                                         ha = 'center',
                                         va = 'center'
                        )
                    self.gr.subplot(3, 1, 2)
                    self.gr.bar(x + offset, relerr,
                                width = width,
                                # marker = self.gr.getMarker(str(paramSet) + stp),
                                label = self.wrapLegend('%s' % self.printParams(paramSet, stp)),
                                color = self.gr.getColor(str(paramSet) + stp), alpha = self.alpha,
                                log = bool(decoration.get('g_ylogscale')) if decoration is not None else False
                    )
                    self.gr.subplot(3, 1, 3)
                    self.gr.bar(x + offset, dev,
                                width = width,
                                # marker = self.gr.getMarker(str(paramSet) + stp),
                                label = self.wrapLegend('%s' % self.printParams(paramSet, stp)),
                                color = self.gr.getColor(str(paramSet) + stp), alpha = self.alpha,
                                log = bool(decoration.get('g_ylogscale')) if decoration is not None else False
                    )
                    offset += width
        if plot:
            self.gr.subplot(3, 1, 1)
            self.removeGraphScale(scaleDict)
            self.setMargins()
            self.decorate(decoration,
                          g_xlabel = self.printVariables(variables),
                          g_ylabel = "Metric",
                          g_grid = True,
                          g_title = self.printTitle(
                              "Metric for {variables} {grouping}", self.printVariables(variables), self.printParams(grouping))
            )
            self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
            # ya.set_major_locator(MaxNLocator(integer = True))

            self.gr.subplot(3, 1, 2)
            self.removeGraphScale(scaleDict)
            self.setMargins()
            self.decorate(decoration,
                          g_xlabel = self.printVariables(variables),
                          g_ylabel = "Coefficient of variation",
                          g_grid = True,
                          g_title = self.printTitle(
                              "Coefficient of variation (std/metric) for {variables} {grouping}",
                              self.printVariables(variables), self.printParams(grouping))
            )
            self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))

            self.gr.subplot(3, 1, 3)
            self.removeGraphScale(scaleDict)
            self.setMargins()
            self.decorate(decoration,
                          g_xlabel = self.printVariables(variables),
                          g_ylabel = "Absolute incertitude (std)",
                          g_grid = True,
                          g_title = self.printTitle(
                              "Absolute incertitude (std) for {variables} {grouping}", self.printVariables(variables), self.printParams(grouping))
            )
            self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))

            fig = self.gr.gcf()
            fig.set_size_inches(self.graphWidth, 3 * self.graphHeight())
            self.pdf.savefig(bbox_inches = 'tight')  # 'checks/delay.pdf', format = 'pdf', )
        self.gr.close()

    @logGraph
    def bar3dLinksMetric(self, variables = None, parameters = None, grouping = None, parameterSetSelection = None, metrics = None,
                         decoration = None):


        paramSets = self.getParamSet(parameters)
        if callable(parameterSetSelection):
            paramSets = sorted(parameterSetSelection(paramSets))

        if metrics is None:
            metrics = self.newMetric()

        if type(metrics) is dict:
            metrics = [metrics]

        basewidth = 0.5 * (1.0 / len(paramSets))
        width = basewidth
        offset = 0

        axes = {}

        axes[1] = self.gr.subplot3d(3, 1, 1)
        axes[2] = self.gr.subplot3d(3, 1, 2)
        axes[3] = self.gr.subplot3d(3, 1, 3)
        plot = False
        for paramSet in paramSets:
            for metric in metrics:
                stp = 'selection = %s' % metric['name']
                selector = dict(grouping.items() + paramSet.items())
                x, metric, dev, relerr, nsamples = self.getLinksMetric(variables, selector, metric = metric)
                if len(x) > 0:
                    plot = True
                    X = x[0]
                    Y = x[1]
                    axes[1].plot(xs = X + offset + width / 2.0, ys = Y, zs = metric,
                                 zdir = 'z',
                                 marker = self.gr.getMarker(str(paramSet) + stp),
                                 label = self.wrapLegend('%s' % self.printParams(paramSet, stp)),
                                 color = self.gr.getColor(str(paramSet) + stp),
                                 linestyle = 'None',
                                 alpha = self.alpha
                    )
                    axes[1].bar(X + offset, metric, zs = Y,
                                zdir = 'y',
                                # label = '%s' % self.printParams(paramSet, stp),
                                color = self.gr.getColor(str(paramSet) + stp),
                                width = width,
                                alpha = self.alpha
                    )
                    for i, num in enumerate(nsamples):
                        axes[1].text(X[i] + offset, Y[i], metric[i] / 2.0,
                                     num,
                                     alpha = 1,
                                     zorder = 10000

                        )
                    axes[2].plot(xs = X + offset + width / 2.0, ys = Y, zs = relerr,
                                 zdir = 'z',
                                 marker = self.gr.getMarker(str(paramSet) + stp),
                                 label = self.wrapLegend('%s' % self.printParams(paramSet, stp)),
                                 color = self.gr.getColor(str(paramSet) + stp),
                                 linestyle = 'None',
                                 alpha = self.alpha
                    )
                    axes[2].bar(X + offset, relerr, zs = Y,
                                zdir = 'y',
                                # label = '%s' % self.printParams(paramSet, stp),
                                color = self.gr.getColor(str(paramSet) + stp),
                                width = width,
                                alpha = self.alpha
                    )
                    axes[3].plot(xs = X + offset + width / 2.0, ys = Y, zs = dev,
                                 zdir = 'z',
                                 marker = self.gr.getMarker(str(paramSet) + stp),
                                 label = self.wrapLegend('%s' % self.printParams(paramSet, stp)),
                                 color = self.gr.getColor(str(paramSet) + stp),
                                 linestyle = 'None',
                                 alpha = self.alpha
                    )
                    axes[3].bar(X + offset, dev, zs = Y,
                                zdir = 'y',
                                # label = '%s' % self.printParams(paramSet, stp),
                                color = self.gr.getColor(str(paramSet) + stp),
                                width = width,
                                alpha = self.alpha
                    )
                    offset += width
        if plot:
            xlab = variables.keys()[0]
            ylab = variables.keys()[1]
            self.decorate(decoration,
                          axes = axes[1],
                          g_xlabel = xlab,
                          g_ylabel = ylab,
                          g_zlabel = "Metric",
                          g_title = self.printTitle(
                              "Metric for {variables} {grouping}", self.printVariables(variables), self.printParams(grouping))
            )
            axes[1].legend(loc = 'center left', bbox_to_anchor = (1, 0.5), numpoints = 1)

            self.decorate(decoration,
                          axes = axes[2],
                          g_xlabel = xlab,
                          g_ylabel = ylab,
                          g_zlabel = "Coefficient of variation ",
                          g_title = self.printTitle(
                              "Coefficient of variation (std/metric) for {variables} {grouping}", self.printVariables(variables), self.printParams(
                                  grouping))
            )
            axes[2].legend(loc = 'center left', bbox_to_anchor = (1, 0.5), numpoints = 1)
            #
            self.decorate(decoration,
                          axes = axes[3],
                          g_xlabel = xlab,
                          g_ylabel = ylab,
                          g_zlabel = "Absolute incertitude",
                          g_title = self.printTitle(
                              "Absolute incertitude (std) for {variables} {grouping}", self.printVariables(variables), self.printParams(grouping))
            )
            axes[3].legend(loc = 'center left', bbox_to_anchor = (1, 0.5), numpoints = 1)
            # if azimut is not None or elevation is not None:
            # for ax in axes.itervalues():
            # if azimut is not None:
            # ax.view_init(elev = elevation, azim = azimut)

            fig = self.gr.gcf()
            fig.set_size_inches(12, 15)
            self.pdf.savefig(bbox_inches = 'tight')
        self.gr.close()

    @logGraph
    def plotLinksMetric(self, variables = None, parameters = None, grouping = None, parameterSetSelection = None, metrics = None,
                        decoration = None):
        paramSets = self.getParamSet(parameters)
        if callable(parameterSetSelection):
            paramSets = sorted(parameterSetSelection(paramSets))

        if metrics is None:
            metrics = self.newMetric()

        if type(metrics) is dict:
            metrics = [metrics]
        plot = False
        # plot all combination of parameters
        # plot = False
        for paramSet in paramSets:
            for metric in metrics:
                stp = 'selection = %s' % metric['name']
                selector = dict(grouping.items() + paramSet.items())
                x, metric, dev, relerr, nsamples = self.getLinksMetric(variables, selector, metric = metric)
                if len(x) > 0:
                    x = x[0]
                    plot = True
                    x = self.jitter(x)
                    self.gr.subplot(3, 1, 1)
                    self.gr.errorbar(x, metric, yerr = dev,
                                     marker = self.gr.getMarker(str(paramSet) + stp),
                                     label = self.wrapLegend('%s' % self.printParams(paramSet, stp)),
                                     color = self.gr.getColor(str(paramSet) + stp),
                                     alpha = self.alpha
                    )
                    # for i, num in enumerate(nsamples):
                    # self.gr.annotate(num, xy = (x[i], metric[i]), textcoords = 'offset points', xytext = (4, 4))#, xytext = (x[i],
                    # metric[i])
                    self.gr.subplot(3, 1, 2)
                    self.gr.plot(x, relerr,
                                 marker = self.gr.getMarker(str(paramSet) + stp),
                                 label = self.wrapLegend('%s' % self.printParams(paramSet, stp)),
                                 color = self.gr.getColor(str(paramSet) + stp), alpha = self.alpha
                    )
                    for i, num in enumerate(nsamples):
                        self.gr.annotate(num,
                                         xy = (x[i], relerr[i]),
                                         textcoords = 'offset points',
                                         xytext = (4, 4))
                    self.gr.subplot(3, 1, 3)
                    self.gr.plot(x, dev,
                                 marker = self.gr.getMarker(str(paramSet) + stp),
                                 label = self.wrapLegend('%s' % self.printParams(paramSet, stp)),
                                 color = self.gr.getColor(str(paramSet) + stp), alpha = self.alpha
                    )
            if plot:
                self.gr.subplot(3, 1, 1)
                self.setMargins()
                self.decorate(decoration,
                              g_xlabel = self.printVariables(variables),
                              g_ylabel = "Metric",
                              g_grid = True,
                              g_title = self.printTitle(
                                  "Metric for {variables} {grouping}", self.printVariables(variables), self.printParams(grouping))
                )
                self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))

                self.gr.subplot(3, 1, 2)
                self.setMargins()
                self.decorate(decoration,
                              g_xlabel = self.printVariables(variables),
                              g_ylabel = "Coefficient of variation",
                              g_grid = True,
                              g_title = self.printTitle(
                                  "Coefficient of variation (std/metric) for {variables} {grouping}",
                                  self.printVariables(variables), self.printParams(grouping))
                )
                self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))

                self.gr.subplot(3, 1, 3)
                self.setMargins()
                self.decorate(decoration,
                              g_xlabel = self.printVariables(variables),
                              g_ylabel = "Absolute incertitude (std)",
                              g_grid = True,
                              g_title = self.printTitle(
                                  "Absolute incertitude for {variables} {grouping}", self.printVariables(variables), self.printParams(grouping))
                )
                self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
        if plot:
            fig = self.gr.gcf()
            fig.set_size_inches(15, 25)
            self.pdf.savefig(bbox_inches = 'tight')  # 'checks/delay.pdf', format = 'pdf', )
        self.gr.close()

    @logGraph
    def compareTopos(self, variables = None, grouping = None, decoration = None):
        plot = False
        f = self.getTopoMetrics(variables, grouping)
        for topo, data in f.iteritems():
            x, metric, dev, relerr, nsamples = data
            if len(x) > 0:
                x = x[0]
                plot = True
                # x = self.jitter(x)
                self.gr.subplot(1, 1, 1)
                self.gr.errorbar(x, metric, yerr = dev,
                                 marker = self.gr.getMarker(topo),
                                 label = topo,
                                 color = self.gr.getColor(topo),
                                 alpha = self.alpha
                )
                for i, num in enumerate(nsamples):
                    self.gr.annotate(num, xy = (x[i], metric[i]), textcoords = 'offset points', xytext = (4, 4))  # , xytext = (x[i],
                    # metric[i])
                    # self.gr.subplot(3, 1, 2)
                    # self.gr.plot(x, relerr,
                    # marker = self.gr.getMarker(topo),
                    # label = '%s' % self.printParams(paramSet, stp),
                    # color = self.gr.getColor(str(paramSet) + stp), alpha = self.alpha
                    # )
                    # for i, num in enumerate(nsamples):
                    # self.gr.annotate(num,
                    # xy = (x[i], relerr[i]),
                    # textcoords = 'offset points',
                    # xytext = (4, 4))
                    # self.gr.subplot(3, 1, 3)
                    # self.gr.plot(x, dev,
                    # marker = self.gr.getMarker(str(paramSet) + stp),
                    # label = '%s' % self.printParams(paramSet, stp),
                    # color = self.gr.getColor(str(paramSet) + stp), alpha = self.alpha
                    # )
        if plot:
            self.gr.subplot(1, 1, 1)
            self.setMargins()
            self.decorate(decoration,
                          g_xlabel = self.printVariables(variables),
                          g_ylabel = "Metric",
                          g_grid = True,
                          g_title = self.printTitle(
                              "Topology comparison for {variables} {grouping}", self.printVariables(variables), self.printParams(grouping))
            )
            self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))

            # self.gr.subplot(3, 1, 2)
            # self.setMargins()
            # self.gr.decorate(g_xlabel = self.printVariables(variables),
            # g_ylabel = "Relative incertitude",
            # g_grid = True,
            # g_title = self.wrapTitle(
            # "Relative incertitude (std/metric) for %s with %s " % (
            # self.printVariables(variables), self.printParams(grouping))))
            # self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
            #
            # self.gr.subplot(3, 1, 3)
            # self.setMargins()
            # self.gr.decorate(g_xlabel = self.printVariables(variables),
            # g_ylabel = "Absolute incertitude",
            # g_grid = True,
            # g_title = self.wrapTitle(
            # "Absolute incertitude for %s with %s " % (self.printVariables(variables), self.printParams(grouping))))
            # self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
        if plot:
            fig = self.gr.gcf()
            fig.set_size_inches(self.graphWidth, self.graphHeight())
            self.pdf.savefig(bbox_inches = 'tight')  # 'checks/delay.pdf', format = 'pdf', )
        self.gr.close()


class SetPlotter(Plotter):
    def getAvgRandIndex(self, variables, selector):
        # variable = variables.keys()[0]
        f = {}
        for exp in self.data:
            if self.selectParams(exp, selector):
                key = tuple(exp['parameters'][v] for v in variables.keys())
                if not f.has_key(key):
                    f[key] = []
                f[key].append(exp['setStatistics']['randindex'])
        keys = sorted(f.keys())
        x = zip(*keys)
        rand = []
        dev = []
        for k in keys:
            v = f[k]
            m = self.np.mean(v)
            std = self.np.std(v)
            rand.append(m)
            dev.append(std)

        # d = self.sort(d)
        # x = d[0]
        # rand = d[1]
        # err = d[2]
        return self.np.array(x), self.np.array(rand), self.np.array(dev)

    @logGraph
    def barRandMeasure(self, variables = None, parameters = None, grouping = None, decoration = None):

        plot = False
        width = 0.7 * (1.0 / len(parameters))
        # width = basewidth
        offset = 0
        scaleDict = {}
        for name, paramSet in parameters.iteritems():
            selector = dict(grouping.items() + paramSet.items())
            x, rand, dev = self.getAvgRandIndex(variables, selector)
            if len(x) > 0:
                x = self.removeScale(scaleDict, x[0])
                # if len(x) > 1:
                # width = (x[1:] - x[:-1]).min() * basewidth
                plot = True
                self.gr.subplot(1, 1, 1)
                self.gr.bar(x + offset, rand, yerr = dev,
                            width = width,
                            label = self.wrapLegend('%s' % self.printParams(paramSet, name)),
                            color = self.gr.getColor(str(paramSet)),
                            alpha = self.alpha,
                            log = bool(decoration.get('g_ylogscale')) if decoration is not None else False
                )
                # for i, num in enumerate(nsamples):
                # self.gr.annotate(num,
                # xy = (x[i] + offset + width / 2.0, metric[i] / 2.0),
                # textcoords = 'offset points',
                # xytext = (4, 4),
                # ha = 'center',
                # va = 'center'
                # )
                # self.gr.subplot(2, 1, 2)
                # self.gr.bar(x + offset, dev,
                # width = width,
                # # marker = self.gr.getMarker(str(paramSet) + stp),
                # label = '%s' % self.printParams(paramSet),
                # color = self.gr.getColor(str(paramSet)), alpha = self.alpha
                # )
                offset += width
        if plot:
            self.gr.subplot(1, 1, 1)
            self.removeGraphScale(scaleDict)
            self.setMargins()
            self.decorate(decoration,
                          g_xlabel = self.printVariables(variables),
                          g_ylabel = "Rand Index",
                          g_grid = True,
                          g_title = self.printTitle(
                              "Rand index for {variables} {grouping}", self.printVariables(variables), self.printParams(grouping))
            )
            self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
            # ya.set_major_locator(MaxNLocator(integer = True))

            # self.gr.subplot(2, 1, 2)
            # self.setMargins()
            # self.gr.decorate(g_xlabel = self.printVariables(variables),
            # g_ylabel = "Std dev",
            # g_grid = True,
            # g_title = title if title else self.wrapTitle(
            # "Standart deviation for %s with %s " % (
            # self.printVariables(variables), self.printParams(grouping))))
            # self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))

            fig = self.gr.gcf()
            fig.set_size_inches(self.graphWidth, self.graphHeight())
            self.pdf.savefig(bbox_inches = 'tight')  # 'checks/delay.pdf', format = 'pdf', )
        self.gr.close()

    @logGraph
    def bar3dRandMeasure(self, variables = None, parameters = None, grouping = None, parameterSetSelection = None, decoration = None):

        paramSets = self.getParamSet(parameters)
        if callable(parameterSetSelection):
            paramSets = parameterSetSelection(paramSets)

        basewidth = 0.5 * (1.0 / len(paramSets))
        width = basewidth
        offset = 0

        axes = {}

        axes[1] = self.gr.subplot3d(2, 1, 1)
        axes[2] = self.gr.subplot3d(2, 1, 2)
        plot = False
        for paramSet in paramSets:
            selector = dict(grouping.items() + paramSet.items())
            x, metric, dev = self.getAvgRandIndex(variables, selector)
            if len(x) > 0:
                plot = True
                X = x[0]
                Y = x[1]
                axes[1].plot(xs = X + offset + width / 2.0, ys = Y, zs = metric,
                             zdir = 'z',
                             marker = self.gr.getMarker(str(paramSet)),
                             label = self.wrapLegend('%s' % self.printParams(paramSet, stp)),
                             color = self.gr.getColor(str(paramSet)),
                             linestyle = 'None',
                             alpha = self.alpha
                )
                axes[1].bar(X + offset, metric, zs = Y,
                            zdir = 'y',
                            # label = '%s' % self.printParams(paramSet, stp),
                            color = self.gr.getColor(str(paramSet)),
                            width = width,
                            alpha = self.alpha
                )
                axes[2].plot(xs = X + offset + width / 2.0, ys = Y, zs = dev,
                             zdir = 'z',
                             marker = self.gr.getMarker(str(paramSet)),
                             label = self.wrapLegend('%s' % self.printParams(paramSet, stp)),
                             color = self.gr.getColor(str(paramSet)),
                             linestyle = 'None',
                             alpha = self.alpha
                )
                axes[2].bar(X + offset, dev, zs = Y,
                            zdir = 'y',
                            # label = '%s' % self.printParams(paramSet, stp),
                            color = self.gr.getColor(str(paramSet)),
                            width = width,
                            alpha = self.alpha
                )
                offset += width
        if plot:
            xlab = variables.keys()[0]
            ylab = variables.keys()[1]
            self.decorate(decoration,
                          axes = axes[1],
                          g_xlabel = xlab,
                          g_ylabel = ylab,
                          g_zlabel = "Rand Index",
                          g_title = self.printTitle(
                              "Rand Index for {variables} {grouping}", self.printVariables(variables), self.printParams(grouping))
            )
            axes[1].legend(loc = 'center left', bbox_to_anchor = (1, 0.5), numpoints = 1)

            self.decorate(decoration,
                          axes = axes[2],
                          g_xlabel = xlab,
                          g_ylabel = ylab,
                          g_zlabel = "Std dev",
                          g_title = self.printTitle(
                              "Std deviation for {variables} {grouping}", self.printVariables(variables), self.printParams(
                                  grouping))
            )
            axes[2].legend(loc = 'center left', bbox_to_anchor = (1, 0.5), numpoints = 1)
            #
            # if azimut is not None or elevation is not None:
            # for ax in axes.itervalues():
            # if azimut is not None:
            # ax.view_init(elev = elevation, azim = azimut)

            fig = self.gr.gcf()
            fig.set_size_inches(13, 17)
            self.pdf.savefig(bbox_inches = 'tight')  # 'checks/delay.pdf', format = 'pdf', )
        self.gr.close()


if __name__ == "__main__":
    log.setLogLevel('info')
    from argparse import ArgumentParser

    parser = ArgumentParser(description = "Options for starting the custom Mininet network builder")

    # # emulation options
    # parser.add_argument("--topo",
    # dest = 'tfile',
    # help = 'Topology to load for this simulation',
    # default = None)

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

