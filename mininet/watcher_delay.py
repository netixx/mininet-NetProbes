"""Module for extracting and plotting data about watchers"""

import json
import textwrap
import collections
import math
import itertools

import networkx as nx

from mininet import log


ALL_RESULTS_PATH = 'watchers/watchers.json'
PLOT_PATH = 'watchers/watchers.pdf'


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
    # matches[color] = graphSets[indexes[0]]
    # else:
    # raise RuntimeError("Some items were not matched.")
    # return matches


def _precision(watcherSet, graphSet):
    return float(len(set(watcherSet) & set(graphSet))) / len(watcherSet) if len(watcherSet) > 0 else 0


def _recall(watcherSet, graphSet):
    return float(len(set(watcherSet) & set(graphSet))) / len(graphSet)


def _true_pos(white, good):
    return len(set(white) & set(good))


def _true_neg(black, bad):
    return len(set(black) & set(bad))


def _false_neg(black, good):
    return len(set(black) & set(good))


def _false_pos(white, bad):
    return len(set(white) & set(bad))


# def _conf_matrix(matches, watcher):
# mat = []
# for color, part in matches.iteritems():
# waAddrs = [p['address'] for p in watcher[color]['hosts']]
# mat.append([waAddrs, part])
#
# white = mat[0][0]
# good = mat[0][1]
# black = mat[1][0]
# bad = mat[1][1]
#
# tp = _true_pos(white, good)
# tn = _true_neg(black, bad)
# fn = _false_neg(black, good)
# fp = _false_pos(white, bad)
#
# return tp, tn, fp, fn


def nCk(n, k):
    f = math.factorial
    return f(n) / f(k) / f(n - k)


def _rand_index(matches, watcher):
    a = 0
    b = 0
    c = 0
    d = 0
    wa = [[p['address'] for p in watcher[color]['hosts']] for color in matches.iterkeys()]
    #remove probe that are not in the watcher set
    gt = [p for g in matches.itervalues() for p in g if any(p in w for w in wa)]
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
        #each element of the pair belong to the different sets in both partitions
        elif not sameWa and not sameGt:
            b += 1
        elif sameWa and not sameGt:
            c += 1
        elif not sameWa and sameGt:
            d += 1

    # return (a + b) / nCk(len(S), 2)
    den = float(a + b + c + d)
    if den != t:
        raise log.error("Error in rand measure")
    return (a + b) / den if den != 0 else 0


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

    # if len(matches) == 2:
    #     tp, tn, fp, fn = _conf_matrix(matches, watcher)
    #     print 'tp,tn,fp,fn',tp, tn, fp, fn
    #     stats['accuracy'] = float(tp + tn) / float(tp + tn + fp + fn)
    #     # stats['mcc'] = float(tp * tn - fp * fn) / math.sqrt((tp + fp) * (fp + fn) * (tn + fp) * (tn + fn))
    #     stats['binRecall'] = tp / float(tp + fn)
    #     stats['binPrecision'] = tp / float(tp + fp)
    #     stats['binFmeasure'] = 2 * tp / float(2 * tp + fp + fn)
    # print 'fm', stats['Fmeasure'], 'rand', stats['randindex']
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
    try:
        _makeSetResults(watcher, topoGraph, out, nameToIp)
        thosts = _connectGraph(topoGraph, topo, tLinks)
        _makeLinkResults(watcher, topoGraph, out, ipToName, nameToIp)
        out['tlinkswdep'] = _depth_links(topoGraph, thosts)
    except:
        raise

    out['parameters']['depth'] = out['tlinkswdep'][0][0]
    out['tlinks'] = tLinks
    out['topoFile'] = topoFile
    return out


def _depth_links(topoGraph, tlinks, root = 's1'):
    l = []
    for link, hosts in tlinks.iteritems():
        l.append((_link_depth(topoGraph, hosts, root), link))

    return l


def _link_depth(topoGraph, link, root):
    return max(len(nx.shortest_path(topoGraph, source = root, target = h)) for h in link)


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
    log.output("Making new graph at %s\n" % pp)
    from graphs import Graph as g
    # from graphs import D3Graph as d3g

    linkplotter = LinkPlotter(g, pp + "-link.pdf", results)

    try:
        makeGraphsLinks(linkplotter)
    finally:
        linkplotter.close()

    setplotter = SetPlotter(g, pp + "-set.pdf", results)
    try:
        makeGraphsGranularitySampleSize(setplotter)
    finally:
        setplotter.close()


def makeGraphsLinks(plotter):
    metricSet = 0, 1
    bucketType = 'probabilistic-power-bucket'  #,'probabilistic-bucket', 'ordered-bucket'
    selectionSet = ge1,  # exclusive,  # None
    sampleSizeSet = 10, 20, 50
    granularity = 1  #, 4
    depthSet = range(1, 10)

    def max(x):
        return plotter.np.percentile(x, 100)

    def percent99(x):
        return plotter.np.percentile(x, 99)

    def std(x):
        return plotter.np.std(x)

    def maxMinus1(x):
        return plotter.np.max(x) - 1

    def std2(x):
        return 2 * plotter.np.std(x)

    def stdy(x):
        return 1.8 * plotter.np.std(x)

    selectionMethods = (
        max,
        # percent99,
        std,
        std2,
        # stdy,
        maxMinus1
    )
    # for selectionMethod in selectionMethods:
    # for paramSelection in selectionSet:
    #     for bucketType in bucketSet:
    #         plotter.plotLinksMetricSelection(
    #             variables = {
    #                 'sampleSize': None
    #             },
    #             parameters = {
    #                 'randomMetricWeight': metricSet,
    #                 'ipMetricWeight': metricSet,
    #                 'balancedMetricWeight': metricSet,
    #                 'delayMetricWeight': metricSet,
    #                 'granularity': granularitySet
    #             },
    #             grouping = {
    #                 'bucket_type': bucketType
    #             },
    #             parameterSetSelection = paramSelection,
    #             electionMethod = selectionMethods
    #         )
    for paramSelection in selectionSet:
        for sampleSize in sampleSizeSet:
            plotter.plotLinksMetric(
                variables = {
                    'depth': None
                },
                parameters = {
                    'randomMetricWeight': metricSet,
                    'ipMetricWeight': metricSet,
                    'balancedMetricWeight': metricSet,
                    'delayMetricWeight': metricSet
                },
                grouping = {
                    'bucket_type': bucketType,
                    'granularity': granularity,
                    'sampleSize': sampleSize
                },
                parameterSetSelection = paramSelection,
                electionMethod = max
            )

    # for paramSelection in selectionSet:
    #     for depth in depthSet:
    #         for sampleSize in sampleSizeSet:
    #             plotter.plotLinksScores(
    #                 variables = {
    #                 },
    #                 parameters = {
    #                     'randomMetricWeight': metricSet,
    #                     'ipMetricWeight': metricSet,
    #                     'balancedMetricWeight': metricSet,
    #                     'delayMetricWeight': metricSet,
    #                 },
    #                 grouping = {
    #                     'bucket_type': bucketType,
    #                     'sampleSize': sampleSize,
    #                     'granularity': granularity,
    #                     'depth': depth
    #                 },
    #                 parameterSetSelection = paramSelection
    #             )

    for paramSelection in selectionSet:
        plotter.plotLinksMetric(
            variables = {
                'depth': None,
                'sampleSize': None
            },
            parameters = {
                'randomMetricWeight': metricSet,
                'ipMetricWeight': metricSet,
                'balancedMetricWeight': metricSet,
                'delayMetricWeight': metricSet
            },
            grouping = {
                'bucket_type': bucketType,
                'granularity': granularity
            },
            parameterSetSelection = paramSelection,
            electionMethod = max
        )


def makeGraphsGranularitySampleSize(plotter):
    granularity = 1
    metricSet = 0, 1
    selectionSet = ge1,  # exclusive,  # None
    bucketType = 'probabilistic-power-bucket'  #,'probabilistic-bucket', 'ordered-bucket',
    sampleSizeSet = 10, 20, 50
    # length of grey, precision + recall (per set and total) wrt delay variation
    # length of grey, precision + recall (per set and total) wrt granularity

    # for paramSelection in selectionSet:
    #     for granularity in granularitySet:
    #         for bucketType in bucketSet:
    #             plotter.plotAllPlot(
    #                 variables = {
    #                     'sampleSize': None
    #                 },
    #                 parameters = {
    #                     'randomMetricWeight': metricSet,
    #                     'ipMetricWeight': metricSet,
    #                     'balancedMetricWeight': metricSet,
    #                     'delayMetricWeight': metricSet,
    #                 },
    #                 grouping = {
    #                     'granularity': granularity,
    #                     'bucket_type': bucketType
    #                 },
    #                 parameterSetSelection = paramSelection
    #             )
    for paramSelection in selectionSet:
        for sampleSize in sampleSizeSet:
            plotter.plotAllPlot(
                variables = {
                    'depth': None
                },
                parameters = {
                    'randomMetricWeight': metricSet,
                    'ipMetricWeight': metricSet,
                    'balancedMetricWeight': metricSet,
                    'delayMetricWeight': metricSet,

                },
                grouping = {
                    'granularity': granularity,
                    'bucket_type': bucketType,
                    'sampleSize': sampleSize
                },
                parameterSetSelection = paramSelection
            )


class Plotter(object):
    def __init__(self, graph, plotPath, data, d3 = None):
        self.gr = graph
        self.data = data
        self.plotPath = plotPath
        self.pdf = self.preparePlot()
        import numpy as np

        self.np = np

    def preparePlot(self):
        from matplotlib.backends.backend_pdf import PdfPages

        self.alpha = 0.9
        self.alpha3d = 0.75
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

    def jitter(self, array):
        import numpy as np

        if array.size <= 1:
            return array
        maxJitter = (array.max() - array.min()) / 100.0
        # print array
        jit = (np.random.random_sample((array.size,)) - 0.5) * 2 * maxJitter
        # array += jit
        return array + jit


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
        r = self.sort(r, 1)
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
        r = self.sort(r)

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
                key = tuple(exp['parameters'][v] for v in variables.keys())
                if not f.has_key(key):
                    f[key] = []
                f[key].append(s)
        return f

    def getLinksMetric(self, variables, selector, electionMethod = None):

        f = self.getRawLinksMetric(variables, selector, electionMethod)
        x = []
        y = []
        dev = []
        err = []
        num = []
        keys = sorted(f.keys())
        x = zip(*keys)
        for k in keys:
            v = f[k]
            # x.append(k)
            m = self.np.mean(v)
            std = self.np.std(v)
            y.append(m)
            dev.append(std)
            err.append(std / m if m > 0 else 0)
            num.append(len(v))

        return self.np.array(x), self.np.array(y), self.np.array(dev), self.np.array(err), self.np.array(num)

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
            if len(x) < 1:
                return
            self.gr.subplot(3, 1, 1)
            x = self.np.array(x)
            x = self.jitter(x)
            self.gr.errorbar(x, self.np.array(y), yerr = self.np.array(dev),
                             marker = self.gr.getMarker(stp),
                             label = stp,
                             color = self.gr.getColor(stp), alpha = self.alpha
            )
            self.gr.hold = True
            self.gr.subplot(3, 1, 2)
            self.gr.plot(x, self.np.array(err),
                         marker = self.gr.getMarker(stp),
                         label = stp,
                         color = self.gr.getColor(stp), alpha = self.alpha
            )
            self.gr.hold = True
            self.gr.subplot(3, 1, 3)
            self.gr.plot(x, self.np.array(dev),
                         marker = self.gr.getMarker(stp),
                         label = stp,
                         color = self.gr.getColor(stp), alpha = self.alpha
            )
            self.gr.hold = True

        self.gr.subplot(3, 1, 1)
        self.setMargins()
        self.gr.decorate(g_xlabel = self.printVariables(variables),
                         g_ylabel = "Metric",
                         g_grid = True,
                         g_title = self.wrapTitle("Aggregated metrics for %s with %s" % (self.printVariables(variables), self.printParams(grouping))))
        self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))

        self.gr.subplot(3, 1, 2)
        self.setMargins()
        self.gr.decorate(g_xlabel = self.printVariables(variables),
                         g_ylabel = "Error",
                         g_grid = True,
                         g_title = self.wrapTitle("Aggregated error (avg(std)/avg(metrics)) for %s with %s" % (
                             self.printVariables(variables), self.printParams(grouping))))
        self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))

        self.gr.subplot(3, 1, 3)
        self.setMargins()
        self.gr.decorate(g_xlabel = self.printVariables(variables),
                         g_ylabel = "Std deviation",
                         g_grid = True,
                         g_title = self.wrapTitle("Aggregated error (avg(std)) for %s with %s" % (
                             self.printVariables(variables), self.printParams(grouping))))
        self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))

        fig = self.gr.gcf()
        fig.set_size_inches(15, 18)
        self.pdf.savefig(bbox_inches = 'tight')
        self.gr.close()

    def pointsToSurface(self, X, Y, Z, xi = None, yi = None):
        step = 1
        if xi is None:
            minX = X.min()
            maxX = X.max()
            if not minX < maxX:
                return None, None, None
            xi = self.np.arange(minX, maxX + step, step)
        if yi is None:
            minY = Y.min()
            maxY = Y.max()
            if not minY < maxY:
                return None, None, None
            yi = self.np.arange(minY, maxY + step, step)

        from matplotlib.mlab import griddata

        zi = griddata(X, Y, Z, xi, yi, interp = 'nn')
        # zi = griddata((X, Y), Z, (xi, yi), method = 'linear')
        xim, yim = self.np.meshgrid(xi, yi)
        return xim, yim, zi

    def _plotLinkMetric3d(self, variables = None, paramSets = None, grouping = None, electionMethod = None, elevation = None, azimut = None):
        axes = {}

        axes[1] = self.gr.subplot3d(3, 1, 1)
        axes[2] = self.gr.subplot3d(3, 1, 2)
        axes[3] = self.gr.subplot3d(3, 1, 3)
        # plot all combination of parameters
        plot = False
        for paramSet in paramSets:
            for p in electionMethod:
                stp = 'selection = %s' % p.__name__
                selector = dict(grouping.items() + paramSet.items())
                x, metric, dev, relerr, nsamples = self.getLinksMetric(variables, selector, p)
                if len(x) > 0:
                    plot = True
                    # if len(x.shape) > 1 and x.shape[0] == 2:
                    X = x[0]
                    Y = x[1]

                    axes[1].scatter(xs = X, ys = Y, zs = metric,
                                    zdir = 'z',
                                    marker = self.gr.getMarker(str(paramSet) + stp),
                                    label = '%s' % self.printParams(paramSet, stp),
                                    color = self.gr.getColor(str(paramSet) + stp),
                                    # alpha = 1
                    )
                    axes[2].scatter(xs = X, ys = Y, zs = relerr,
                                    zdir = 'z',
                                    marker = self.gr.getMarker(str(paramSet) + stp),
                                    label = '%s' % self.printParams(paramSet, stp),
                                    color = self.gr.getColor(str(paramSet) + stp),
                    )
                    axes[3].scatter(xs = X, ys = Y, zs = dev,
                                    zdir = 'z',
                                    marker = self.gr.getMarker(str(paramSet) + stp),
                                    label = '%s' % self.printParams(paramSet, stp),
                                    color = self.gr.getColor(str(paramSet) + stp),
                    )
                    xim, yim, metrici = self.pointsToSurface(X, Y, metric)
                    if xim is not None:
                        axes[1].plot_wireframe(xim, yim, metrici,
                                               rstride = 14,
                                               cstride = 14,
                                               color = self.gr.getColor(str(paramSet) + stp),
                                               label = '%s' % self.printParams(paramSet, stp),
                                               alpha = self.alpha3d
                        )
                        xim, yim, relerri = self.pointsToSurface(X, Y, relerr)
                        axes[2].plot_wireframe(xim, yim, relerri,
                                             rstride = 14,
                                             cstride = 14,
                                             color = self.gr.getColor(str(paramSet) + stp),
                                             label = '%s' % self.printParams(paramSet, stp),
                                             alpha = self.alpha3d
                        )
                        xim, yim, devi = self.pointsToSurface(X, Y, dev)
                        axes[3].plot_wireframe(xim, yim, devi,
                                             rstride = 14,
                                             cstride = 14,
                                             color = self.gr.getColor(str(paramSet) + stp),
                                             label = '%s' % self.printParams(paramSet, stp),
                                             alpha = self.alpha3d
                        )
        if plot:
            xlab = variables.keys()[0]
            ylab = variables.keys()[1]
            # self.setMargins()
            self.gr.decorate(axes = axes[1],
                             g_xlabel = xlab,
                             g_ylabel = ylab,
                             g_zlabel = "Metric",
                             g_title = self.wrapTitle("Metric for %s with %s " % (self.printVariables(variables), self.printParams(grouping))))
            # axes[1].legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
            axes[1].legend(loc = 'center left', bbox_to_anchor = (1, 0.5))

            # self.setMargins()
            self.gr.decorate(axes = axes[2],
                             g_xlabel = xlab,
                             g_ylabel = ylab,
                             g_zlabel = "Relative incertitude",
                             g_title = self.wrapTitle(
                                 "Relative incertitude (std/metric) for %s with %s " % (self.printVariables(variables), self.printParams(
                                     grouping))))
            axes[2].legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
            #
            # # self.setMargins()
            self.gr.decorate(axes = axes[3],
                             g_xlabel = xlab,
                             g_ylabel = ylab,
                             g_zlabel = "Absolute incertitude",
                             g_title = self.wrapTitle(
                                 "Absolute incertitude for %s with %s " % (self.printVariables(variables), self.printParams(grouping))))
            axes[3].legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
            # if azimut is not None or elevation is not None:
            for ax in axes.itervalues():
                if azimut is not None:
                    ax.view_init(elev = elevation, azim = azimut)

        return plot


    def plotLinksMetric(self, variables = None, parameters = None, grouping = None, parameterSetSelection = None, electionMethod = None):
        log.output("Making new graph Metric with variables : %s, parameters : %s, grouping : %s\n" % (
            self.printVariables(variables), self.printParams(parameters), self.printParams(grouping)))
        paramSets = self.getParamSet(parameters)
        if callable(parameterSetSelection):
            paramSets = parameterSetSelection(paramSets)

        if not isinstance(electionMethod, collections.Iterable):
            electionMethod = [electionMethod]
        plot = False
        if len(variables) == 2:
            # for az in range(0, 91, 45):
            plot = self._plotLinkMetric3d(variables = variables,
                                          paramSets = paramSets,
                                          grouping = grouping,
                                          electionMethod = electionMethod)
                # if plot:
                #     fig = self.gr.gcf()
                #     fig.set_size_inches(15, 25)
                #     self.pdf.savefig(bbox_inches = 'tight')  # 'checks/delay.pdf', format = 'pdf', )
                # self.gr.close()
            # return
        else:
            # plot all combination of parameters
            # plot = False
            for paramSet in paramSets:
                for p in electionMethod:
                    stp = 'selection = %s' % p.__name__
                    selector = dict(grouping.items() + paramSet.items())
                    x, metric, dev, relerr, nsamples = self.getLinksMetric(variables, selector, p)
                    if len(x) > 0:
                        x = x[0]
                        plot = True
                        x = self.jitter(x)
                        self.gr.subplot(3, 1, 1)
                        self.gr.errorbar(x, metric, yerr = dev,
                                         marker = self.gr.getMarker(str(paramSet) + stp),
                                         label = '%s' % self.printParams(paramSet, stp),
                                         color = self.gr.getColor(str(paramSet) + stp),
                                         alpha = self.alpha
                        )
                        # for i, num in enumerate(nsamples):
                        #     self.gr.annotate(num, xy = (x[i], metric[i]), textcoords = 'offset points', xytext = (4, 4))#, xytext = (x[i],
                        # metric[i])
                        self.gr.subplot(3, 1, 2)
                        self.gr.plot(x, relerr,
                                     marker = self.gr.getMarker(str(paramSet) + stp),
                                     label = '%s' % self.printParams(paramSet, stp),
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
                                     label = '%s' % self.printParams(paramSet, stp),
                                     color = self.gr.getColor(str(paramSet) + stp), alpha = self.alpha
                        )
            if plot:
                self.gr.subplot(3, 1, 1)
                self.setMargins()
                self.gr.decorate(g_xlabel = self.printVariables(variables),
                                 g_ylabel = "Metric",
                                 g_grid = True,
                                 g_title = self.wrapTitle("Metric for %s with %s " % (self.printVariables(variables), self.printParams(grouping))))
                self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))

                self.gr.subplot(3, 1, 2)
                self.setMargins()
                self.gr.decorate(g_xlabel = self.printVariables(variables),
                                 g_ylabel = "Relative incertitude",
                                 g_grid = True,
                                 g_title = self.wrapTitle(
                                     "Relative incertitude (std/metric) for %s with %s " % (
                                         self.printVariables(variables), self.printParams(grouping))))
                self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))

                self.gr.subplot(3, 1, 3)
                self.setMargins()
                self.gr.decorate(g_xlabel = self.printVariables(variables),
                                 g_ylabel = "Absolute incertitude",
                                 g_grid = True,
                                 g_title = self.wrapTitle(
                                     "Absolute incertitude for %s with %s " % (self.printVariables(variables), self.printParams(grouping))))
                self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
        if plot:
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
            if len(links) >= 1:
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

        if len(xticks) >= 1:
            xticks = zip(*sorted([(mapping[link], link) for link in xticks]))
            xticksTop = zip(*sorted([(mapping[link], link) for link in xticksTop]))
            self.gr.subplot(3, 1, 1)

            if len(xticks) == 2:
                self.gr.xticks(xticks[0], xticks[1], rotation = 90, fontsize = 8)
                self.gr.xlim(min(xticks[0]), max(xticks[0]))
            self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
            self.gr.decorate(g_xlabel = 'links',
                             g_ylabel = "Link score",
                             g_grid = True,
                             g_title = self.wrapTitle("Link scores with %s" % self.printParams(grouping))
            )

            self.gr.subplot(3, 1, 2)
            if len(xticks) == 2:
                self.gr.xticks(xticks[0], xticks[1], rotation = 90, fontsize = 8)
                self.gr.xlim(min(xticks[0]), max(xticks[0]))
            self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
            self.gr.decorate(g_xlabel = 'links',
                             g_ylabel = "Link score error",
                             g_grid = True,
                             g_title = self.wrapTitle("Error on scores with %s" % self.printParams(grouping))
            )

            self.gr.subplot(3, 1, 3)
            if len(xticksTop) == 2:
                self.gr.xticks(xticksTop[0], xticksTop[1], rotation = 90, fontsize = 8)
                self.gr.xlim(min(xticksTop[0]), max(xticksTop[0]))
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
    def getAvgFMeasure(self, variables, selector):
        variable = variables.keys()[0]
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

    def getAvgRandIndex(self, variables, selector):
        variable = variables.keys()[0]
        f = {}
        for exp in self.data:
            if self.selectParams(exp, selector):
                if not f.has_key(exp['parameters'][variable]):
                    f[exp['parameters'][variable]] = []
                f[exp['parameters'][variable]].append(exp['setStatistics']['randindex'])
        x = f.keys()
        randIndex = [self.np.mean(v) for v in f.values()]
        stdrandIndex = [self.np.std(v) for v in f.values()]
        d = self.np.array([
            x,
            randIndex,
            stdrandIndex
        ])
        d = self.sort(d)
        x = d[0]
        rand = d[1]
        err = d[2]
        return x, rand, err

    def getFMeasure(self, variables, selector):
        variable = variables.keys()[0]
        d = self.np.array([
            [exp['parameters'][variable] for exp in self.data if self.selectParams(exp, selector, variables)],
            [exp['precisionAndRecall']['Fmeasure'] for exp in self.data if self.selectParams(exp, selector, variables)],
        ])
        d = self.sort(d)
        x = d[0]
        fmeasure = d[1]
        return x, fmeasure

    def getTotalPrecisionAndRecall(self, variables, selector):
        variable = variables.keys()[0]
        d = self.np.array([
            [exp['parameters'][variable] for exp in self.data if self.selectParams(exp, selector, variables)],
            [exp['precisionAndRecall']['total']['precision'] for exp in self.data if self.selectParams(exp, selector, variables)],
            [exp['precisionAndRecall']['total']['recall'] for exp in self.data if self.selectParams(exp, selector, variables)]
        ])
        d = self.sort(d)
        x = d[0]
        precision = d[1]
        recall = d[2]
        return x, precision, recall

    def getPrecisionAndRecall(self, variables, selector, color):
        variable = variables.keys()[0]
        d = self.np.array([
            [exp['parameters'][variable] for exp in self.data if self.selectParams(exp, selector, variables)],
            [exp['precisionAndRecall'][color]['precision'] for exp in self.data if self.selectParams(exp, selector, variables)],
            [exp['precisionAndRecall'][color]['recall'] for exp in self.data if self.selectParams(exp, selector, variables)]
        ])
        d = self.sort(d)
        x = d[0]
        precision = d[1]
        recall = d[2]
        return x, precision, recall

    def getGreys(self, variables, selector):
        variable = variables.keys()[0]
        d = self.np.array([
                              [exp['parameters'][variable] for exp in self.data if self.selectParams(exp, selector, variables)],
                              [float(len(exp['grey'])) / exp['totalTestedProbes'] for exp in self.data if
                               self.selectParams(exp, selector, variables)],
                              [float(len(exp['grey'])) / exp['totalProbes'] for exp in self.data if self.selectParams(exp, selector, variables)]
                          ],
        )

        self.sort(d)
        x = d[0]
        y1 = d[1]
        y2 = d[2]
        return x, y1, y2

    def plotFMeasurePlot(self, grouping = None, **args):
        if grouping is None:
            grouping = {}

        if self.graphAvgFMeasure(grouping = grouping, **args):
            self.setMargins()
            fig = self.gr.gcf()
            fig.set_size_inches(15, 10)
            self.pdf.savefig(bbox_inches = 'tight')
        self.gr.close()

    def plotAllPlot(self, grouping = None, **args):
        if grouping is None:
            grouping = {}
        if self.plotGrey(grouping = grouping, **args):
            fig = self.gr.gcf()
            fig.set_size_inches(15, 10)
            self.pdf.savefig(bbox_inches = 'tight')
        self.gr.close()

        if self.plotPrecisionAndRecall(grouping = grouping, **args):
            fig = self.gr.gcf()
            fig.set_size_inches(15, 30)
            self.pdf.savefig(bbox_inches = 'tight')
        self.gr.close()

    def plotAllScatter(self, grouping = None, **args):
        if grouping is None:
            grouping = {}

        self.plotAllPlot(grouping = grouping, **args)

        if self.scatterGrey(grouping = grouping, **args):
            fig = self.gr.gcf()
            fig.set_size_inches(15, 10)
            self.pdf.savefig(bbox_inches = 'tight')
        self.gr.close()

        if self.scatterPrecisionAndRecall(grouping = grouping, **args):
            fig = self.gr.gcf()
            fig.set_size_inches(20, 20)
            self.pdf.savefig(bbox_inches = 'tight')
        self.gr.close()


    def scatterPrecisionAndRecall(self, **args):
        p = []
        self.gr.subplot(2, 1, 1)
        p.append(self.graphTotalPrecisionAndRecall(grapher = self.gr.scatter, **args))
        self.gr.subplot(2, 1, 2)
        p.append(self.graphTotalFMeasure(grapher = self.gr.scatter, **args))
        return any(item for item in p)


    def plotGrey(self, **args):
        p = self.graphGrey(grapher = self.gr.plot, **args)
        self.setMargins()
        return p

    def plotPrecisionAndRecall(self, **args):
        p = []
        grs = 5
        i = 1
        self.gr.subplot(grs, 1, i)
        p.append(self.graphAvgRandIndex(**args))
        self.setMargins()
        i += 1
        self.gr.subplot(grs, 1, i)
        p.append(self.graphAvgFMeasure(**args))
        self.setMargins()
        i += 1
        self.gr.subplot(grs, 1, i)
        p.append(self.graphFMeasureError(grapher = self.gr.plot, **args))
        self.setMargins()
        i += 1
        self.gr.subplot(grs, 1, i)
        p.append(self.graphTotalPrecision(grapher = self.gr.plot, **args))
        self.setMargins()
        i += 1
        self.gr.subplot(grs, 1, i)
        p.append(self.graphTotalRecall(grapher = self.gr.plot, **args))
        self.setMargins()
        i += 1
        # true if one is true
        return any(item for item in p)


    def graphTotalPrecisionAndRecall(self, grapher = None, variables = None, parameters = None, grouping = None, parameterSetSelection = None):
        if grapher is None or variables is None:
            return
        log.output("Making new graph Total Precision and Recall with variables : %s, parameters : %s, grouping : %s\n" % (
            self.printVariables(variables), self.printParams(parameters), self.printParams(grouping)))
        plot = False
        # plot precision & recall
        paramSets = self.getParamSet(parameters)
        if callable(parameterSetSelection):
            paramSets = parameterSetSelection(paramSets)

        # plot all combination of parameters
        for paramSet in paramSets:

            selector = dict(grouping.items() + paramSet.items())
            x, precision, recall = self.getTotalPrecisionAndRecall(variables, selector)
            if len(x) > 0:
                x = self.jitter(x)
                plot = True
                grapher(x, precision, marker = 'd', label = 'Total Precision for %s' % self.printParams(paramSet),
                        color = self.gr.getColor(str(paramSet)), alpha = self.alpha)
                self.gr.hold = True
                grapher(x, recall, marker = '^', label = 'Total Recall for %s' % self.printParams(paramSet),
                        color = self.gr.getColor(str(paramSet)), alpha = self.alpha)
                self.gr.hold = True

        self.gr.decorate(g_xlabel = self.printVariables(variables),
                         g_ylabel = "Recall/Precision",
                         g_grid = True,
                         g_title = self.wrapTitle(
                             "Total Recall and precision for %s with %s" % (self.printVariables(variables), self.printParams(grouping))))
        self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
        return plot

    def graphTotalPrecision(self, grapher = None, variables = None, parameters = None, grouping = None, parameterSetSelection = None):
        if grapher is None or variables is None:
            return
        log.output("Making new graph TotalPrecision with variables : %s, parameters : %s, grouping : %s\n" % (
            self.printVariables(variables), self.printParams(parameters), self.printParams(grouping)))
        plot = False
        # plot precision & recall
        paramSets = self.getParamSet(parameters)
        if callable(parameterSetSelection):
            paramSets = parameterSetSelection(paramSets)

        # plot all combination of parameters
        for paramSet in paramSets:

            selector = dict(grouping.items() + paramSet.items())
            x, precision, recall = self.getTotalPrecisionAndRecall(variables, selector)
            if len(x) > 0:
                x = self.jitter(x)
                plot = True
                grapher(x, precision, marker = 'd', label = 'Total Precision for %s' % self.printParams(paramSet),
                        color = self.gr.getColor(str(paramSet)), alpha = self.alpha)
                self.gr.hold = True

        self.gr.decorate(g_xlabel = self.printVariables(variables),
                         g_ylabel = "Precision",
                         g_grid = True,
                         g_title = self.wrapTitle("Total Precision for %s with %s" % (self.printVariables(variables), self.printParams(grouping))))
        self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
        return plot

    def graphTotalRecall(self, grapher = None, variables = None, parameters = None, grouping = None, parameterSetSelection = None):
        if grapher is None or variables is None:
            return

        log.output("Making new graph Total Recall with variables : %s, parameters : %s, grouping : %s\n" % (
            self.printVariables(variables), self.printParams(parameters), self.printParams(grouping)))
        plot = False
        # plot precision & recall
        paramSets = self.getParamSet(parameters)
        if callable(parameterSetSelection):
            paramSets = parameterSetSelection(paramSets)

        # plot all combination of parameters
        for paramSet in paramSets:

            selector = dict(grouping.items() + paramSet.items())
            x, precision, recall = self.getTotalPrecisionAndRecall(variables, selector)
            if len(x) > 0:
                x = self.jitter(x)
                plot = True
                grapher(x, recall, marker = '^', label = 'Total Recall for %s' % self.printParams(paramSet),
                        color = self.gr.getColor(str(paramSet)), alpha = self.alpha)
                self.gr.hold = True

        self.gr.decorate(g_xlabel = self.printVariables(variables),
                         g_ylabel = "Recall",
                         g_grid = True,
                         g_title = self.wrapTitle("Total Recall for %s with %s" % (self.printVariables(variables), self.printParams(grouping))))
        self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
        return plot

    def graphPerSetPrecisionAndRecall(self, grapher = None, variables = None, parameters = None, grouping = None, parameterSetSelection = None):
        if grapher is None or variables is None:
            return

        log.output("Making new graph per set Precision and Recall with variables : %s, parameters : %s, grouping : %s\n" % (
            self.printVariables(variables), self.printParams(parameters), self.printParams(grouping)))

        plot = False
        # plot precision & recall
        paramSets = self.getParamSet(parameters)
        if callable(parameterSetSelection):
            paramSets = parameterSetSelection(paramSets)

        # plot all combination of parameters
        for paramSet in paramSets:

            selector = dict(grouping.items() + paramSet.items())
            for color in ('black', 'white'):
                x, p, r = self.getPrecisionAndRecall(variables, selector, color)
                if len(x) > 0:
                    plot = True
                    grapher(x, p, marker = '>', label = 'Precision for set %s %s' % (color, self.printParams(paramSet)),
                            color = self.gr.getColor(color + str(paramSet)), alpha = self.alpha)
                    self.gr.hold = True
                    grapher(x, r, marker = '<', label = 'Recall for set %s %s' % (color, self.printParams(paramSet)),
                            color = self.gr.getColor(color + str(paramSet)), alpha = self.alpha)
                    self.gr.hold = True

        self.gr.decorate(g_xlabel = self.printVariables(variables),
                         g_ylabel = "Recall/Precision",
                         g_grid = True,
                         g_title = self.wrapTitle(
                             "Per set Recall and precision for %s with %s" % (self.printVariables(variables), self.printParams(grouping))))
        self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
        return plot


    def graphAvgRandIndex(self, variables = None, parameters = None, grouping = None, parameterSetSelection = None):
        variable = variables.keys()[0]
        log.output("Making new graph Avg Rand Index with variables : %s, parameters : %s, grouping : %s\n" % (
            self.printVariables(variables), self.printParams(parameters), self.printParams(grouping)))
        paramSets = self.getParamSet(parameters)
        if callable(parameterSetSelection):
            paramSets = parameterSetSelection(paramSets)
        plot = False
        # plot RandIndex
        for paramSet in paramSets:
            selector = dict(grouping.items() + paramSet.items())
            x, rand, yerr = self.getAvgRandIndex(variables, selector)
            if len(x) > 0:
                x = self.jitter(x)
                plot = True
                self.gr.errorbar(x, rand,
                                 yerr = yerr,
                                 marker = self.gr.getMarker(),
                                 label = "Rand Index for %s" % self.printParams(paramSet),
                                 color = self.gr.getColor(str(paramSet)), alpha = self.alpha)
                self.gr.hold = True
        self.gr.decorate(g_xlabel = variable,
                         g_ylabel = "Rand Index",
                         g_grid = True,
                         g_title = self.wrapTitle("Avged Rand Index for %s with %s" % (variable, self.printParams(grouping))))
        self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
        # self.gr.yscale('log')

        self.gr.draw()
        return plot


    def graphAvgFMeasure(self, variables = None, parameters = None, grouping = None, parameterSetSelection = None):
        variable = variables.keys()[0]
        log.output("Making new graph AvgFMeasure with variables : %s, parameters : %s, grouping : %s\n" % (
            self.printVariables(variables), self.printParams(parameters), self.printParams(grouping)))
        paramSets = self.getParamSet(parameters)
        if callable(parameterSetSelection):
            paramSets = parameterSetSelection(paramSets)
        plot = False
        # plot Fmeasure
        for paramSet in paramSets:
            selector = dict(grouping.items() + paramSet.items())
            x, fmeasure, yerr = self.getAvgFMeasure(variables, selector)
            if len(x) > 0:
                x = self.jitter(x)
                plot = True
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
        return plot

    def graphFMeasureError(self, grapher = None, variables = None, parameters = None, grouping = None, parameterSetSelection = None):
        if grapher is None or variables is None:
            return

        log.output("Making new graph FMeasure Error with variables : %s, parameters : %s, grouping : %s\n" % (
            self.printVariables(variables), self.printParams(parameters), self.printParams(grouping)))
        plot = False
        paramSets = self.getParamSet(parameters)
        if callable(parameterSetSelection):
            paramSets = parameterSetSelection(paramSets)
        # plot Fmeasure
        for paramSet in paramSets:
            selector = dict(grouping.items() + paramSet.items())
            x, fmeasure, yerr = self.getAvgFMeasure(variables, selector)
            if len(x) > 0:
                x = self.jitter(x)
                plot = True
                grapher(x, yerr,
                        marker = self.gr.getMarker(),
                        label = "Error for %s" % self.printParams(paramSet),
                        color = self.gr.getColor(str(paramSet)), alpha = self.alpha)
                self.gr.hold = True
        self.gr.decorate(g_xlabel = self.printVariables(variables),
                         g_ylabel = "Std deviation",
                         g_grid = True,
                         g_title = self.wrapTitle(
                             "Std deviation for FMeasure for %s with %s" % (self.printVariables(variables), self.printParams(grouping))))
        self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
        # self.gr.yscale('log')

        self.gr.draw()
        return plot

    def graphTotalFMeasure(self, grapher = None, variables = None, parameters = None, grouping = None, parameterSetSelection = None):
        if grapher is None or variables is None:
            return
        log.output("Making new graph Total FMeasure with variables : %s, parameters : %s, grouping : %s\n" % (
            self.printVariables(variables), self.printParams(parameters), self.printParams(grouping)))
        plot = False
        paramSets = self.getParamSet(parameters)
        if callable(parameterSetSelection):
            paramSets = parameterSetSelection(paramSets)
        # plot Fmeasure
        for paramSet in paramSets:
            selector = dict(grouping.items() + paramSet.items())
            x, fmeasure = self.getFMeasure(variables, selector)
            if len(x) > 0:
                x = self.jitter(x)
                plot = True
                grapher(x, fmeasure,
                        marker = self.gr.getMarker(),
                        label = "Fmeasure for %s" % self.printParams(paramSet),
                        color = self.gr.getColor(str(paramSet)), alpha = self.alpha)
                self.gr.hold = True
        self.gr.decorate(g_xlabel = self.printVariables(variables),
                         g_ylabel = "Fmeasure",
                         g_grid = True,
                         g_title = self.wrapTitle("Fmeasure for %s with %s" % (self.printVariables(variables), self.printParams(grouping))))
        self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))

        self.gr.draw()
        return plot

    def scatterGrey(self, **args):
        self.graphGrey(grapher = self.gr.scatter, **args)

    def graphGrey(self, grapher = None, variables = None, parameters = None, grouping = None, parameterSetSelection = None):
        if grapher is None or variables is None:
            return
        log.output("Making new graph Grey probes with variables : %s, parameters : %s, grouping : %s\n" % (
            self.printVariables(variables), self.printParams(parameters), self.printParams(grouping)))
        plot = False
        paramSets = self.getParamSet(parameters)
        if callable(parameterSetSelection):
            paramSets = parameterSetSelection(paramSets)
        # plot all combination of parameters
        for paramSet in paramSets:
            selector = dict(grouping.items() + paramSet.items())
            x, y1, y2 = self.getGreys(variables, selector)
            if len(x) > 0:
                plot = True
                grapher(x, y1, marker = 'd', label = 'grey probes' + self.printParams(paramSet),
                        color = self.gr.getColor(str(paramSet)), alpha = self.alpha)
                self.gr.hold = True

        self.gr.decorate(g_xlabel = self.printVariables(variables),
                         g_ylabel = 'Ratio of grey probes',
                         g_grid = True,
                         g_title = self.wrapTitle(
                             "Ratio of grey probes (gp/testedProbes) for %s with %s" % (self.printVariables(variables), self.printParams(grouping))))
        self.gr.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
        self.gr.draw()
        return plot


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

