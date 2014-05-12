"""Module for checking properties on Mininet virtual network
Currently, bandwidth and delays can be check with the methods provided by
the measures module"""

import time
import datetime
import random
import os
import json

from mininet.log import info, error
import events
from measures import Traceroute, Ping, DelayStats, IPerf, Spruce, IGI, Assolo, Abing
import vars


def check(net, level):
    """Check virtual network
    :param net : the network to test
    :param level : level of check, the higher the thorougher
    """
    if level >= 1:
        net.pingAll()
    if level >= 2:
        checkDelay(net)
        checkBw(net)


def checkDelay(net):
    """Check delays on net
    :param net : net to check
    """
    info("Checking delays consistency.\n")
    try:
        Delay.check(net)
    except Exception as e:
        error('Could not check delays %s\n' % e)


def checkBw(net):
    """Check Bandwidth on net
    :param net : net to check
    """
    info("Checking bandwidth consistency.\n")
    try:
        Bandwidth.check(net)
    except Exception as e:
        error('Could not check bandwidth %s\n' % e)


class Bandwidth(object):
    """Set of bandwidth check to perform"""
    net = None
    PAIRS = [('h1', 'h7'), ('h2', 'h6')]
    #bitrate in Mbps
    STEPS = [100, 10, 1, 0.1]
    SAMPLE_NUMBER = 20
    time_start = None

    _START_WAIT = 0.1
    _STOP_WAIT = 0.2
    _SAMPLE_WAIT = 0.01

    methods = {
        'iperf': {
            'method': IPerf.bw,
            'options': {}
        },
        'spruce': {
            'method': Spruce.bw,
            'options': {'binDir': os.path.join(vars.testBinPath, 'spruce')}
        },
        'igi': {
            'method': IGI.bw,
            'options': {'binDir': os.path.join(vars.testBinPath, 'igi')}
        },
        'assolo': {
            'method': Assolo.bw,
            'options': {'binDir': os.path.join(vars.testBinPath, 'assolo'),
                        'duration': 10}
        },
        'abing': {
            'method': Abing.bw,
            'options': {'binDir': os.path.join(vars.testBinPath, 'abing')}
        },
        # 'yaz' : {
        #     'method' : Yaz.bw,
        #     'options': {'binDir': os.path.join(vars.testBinPath)}
        # }
    }

    @classmethod
    def _getTimeStamp(cls):
        return time.time() - cls.time_start


    @classmethod
    def check(cls, net):
        """Check this net
        :param net: net to check
        """
        cls.net = net
        # get a baseline
        pairs = cls._strToNodes(cls.PAIRS)
        for name, method in cls.methods.iteritems():
            method['pairs'] = [cls.HostStats(pair[0], pair[1], name) for pair in pairs]
        info("&&& Selected pairs : %s\n" % ", ".join(["(%s, %s)" % pair for pair in cls.PAIRS]))
        info("&&& Selected methods : %s\n" % ", ".join(cls.methods.keys()))
        cls._getBaselines(cls.methods)
        info("&&& Steps for this run %s\n" % (", ".join(["%sMpbs" % step for step in cls.STEPS])))
        info("&&& Running tests\n")
        for name, method in cls.methods.iteritems():
            info("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
            info("&& Running test for method %s\n" % name)
            cls._runSteps(method)
        info("&&& All tests are done\n")
        cls.makeResults(cls.methods, saveResults = True)


    @classmethod
    def _runSteps(cls, method):
        cls.time_start = time.time()
        method['real_steps'] = []
        for step in cls.STEPS:
            cls.__runStep(step, method)
        #for step graph
        method['real_steps'].append((cls.STEPS[-1], cls._getTimeStamp()))
        cls._resetEvent()

    @classmethod
    def __runStep(cls, step, method):
        info("&& Testing next step %sMbps\n" % step)
        method['real_steps'].append((step, cls._getTimeStamp()))
        cls._makeEvent(step)
        time.sleep(cls._START_WAIT)
        cls.__getSamples(step, method['pairs'], method['method'], method['options'])
        time.sleep(cls._STOP_WAIT)


    @classmethod
    def __getSamples(cls, step, pairs, method, options = {}):
        info("& Getting samples : ")
        options['bw'] = "%sM" % step
        for i in range(1, cls.SAMPLE_NUMBER + 1):
            info("%s " % i)
            for pair in pairs:
                bw = method((pair.host, pair.target), **options)
                bw.step = step
                bw.timestamp = cls._getTimeStamp()
                # if ping.sent > 0:
                pair.measures.append(bw)
                info("({:.2f}M) ".format(bw.bw / (1000.0 ** 2)))
                time.sleep(cls._SAMPLE_WAIT)
        info("\n")

    @classmethod
    def _getBaselines(cls, methods):
        info("&& Getting baselines\n")
        for name, method in methods.iteritems():
            info("& Getting baseline for method %s\n" % name)
            cls.__setBaseline(method['pairs'], method['method'], method['options'])
        info("&& Baselines done\n")

    @classmethod
    def __setBaseline(cls, pairs, method, options = {}):
        for pair in pairs:
            pair.baseline = method((pair.host, pair.target), **options)


    @classmethod
    def _makeEvent(cls, delay):
        events.runEvent(cls.__getEvent(delay), cls.net)

    @classmethod
    def __getEvent(cls, bw, target = 'l11'):
        return events.NetEvent(target = target,
                               variations = {'bw': bw})

    @classmethod
    def _resetEvent(cls, target = 'l11'):
        events.resetTarget(cls.net.get(target))


    @classmethod
    def _strToNodes(cls, pairs):
        return [(cls.net.getNodeByName(s1), cls.net.getNodeByName(s2)) for s1, s2 in pairs]

    class HostStats(object):
        """Storage for results"""

        def __init__(self, host, target, method = '', baseline = None, measures = []):
            self.host = host
            self.target = target
            self.measures = measures
            self.baseline = baseline
            self.method = method

        def subtrackBaseline(self, ping):
            return DelayStats(ping.timestamp,
                              ping.step,
                              ping.sent,
                              ping.received,
                              ping.bw - self.baseline.bw)

        def getPair(self):
            return self.host.name, self.target.name

        def getStrPair(self):
            return "%s,%s" % self.getPair()

        def printAll(self):
            return "\n%s -> %s\nbaseline for %s: \n   %s\nmeasures : \n   %s\n" % (self.host.name,
                                                                                   self.target.name,
                                                                                   self.method,
                                                                                   self.baseline.printAll() if self.baseline is not None else '',
                                                                                   "\n   ".join([m.printAll() for m in self.measures]))

        def toDict(self):
            return {'host': self.host.name,
                    'target': self.target.name,
                    'baseline': self.baseline.toDict(),
                    'measures': [m.toDict() for m in self.measures]}

    @classmethod
    def saveResults(cls, methods):
        """Save results to json file
        :param methods: results to save"""
        info('Saving bandwidth results\n')
        import json

        results = {}
        for name, method in methods.iteritems():
            results[name] = {}
            results[name]['pairs'] = []
            results[name]['real_steps'] = method['real_steps']
            for pair in method['pairs']:
                results[name]['pairs'].append(pair.toDict())
        fn = "checks/bw_%s.json" % datetime.datetime.now()
        json.dump(results, open(fn, 'w'))
        return fn

    @classmethod
    def loadResults(cls, jsonFile):
        """Load results from json file
        :param jsonFile : file to load"""
        ms = json.load(open(jsonFile, 'r'))
        methods = {}

        class Host(object):
            def __init__(self, name):
                self.name = name
        for name, method_data in ms.iteritems():
            methods[name] = {}
            methods[name]['pairs'] = []
            methods[name]['real_steps'] = method_data['real_steps']
            for pairdata in method_data['pairs']:
                s = cls.HostStats(method = name,
                                  host = Host(pairdata['host']),
                                  target = Host(pairdata['target']),
                                  baseline = DelayStats(**pairdata['baseline']),
                                  measures = [DelayStats(**m) for m in pairdata['measures']])
                methods[name]['pairs'].append(s)
        info("Loading bw results done.\n")
        return methods

    @classmethod
    def makeResults(cls, methods, saveResults = True):
        """Make result to graphics
        :param methods : results to process
        :param saveResults : save results to json file ?"""
        if saveResults:
            try:
                fn = cls.saveResults(methods)
                info("Saved bandwidth results to file %s\n" % fn)
            except Exception as e:
                error("Could not save bandwidth results : %s\n" % e)

        import numpy as np
        from matplotlib.backends.backend_pdf import PdfPages

        nmethods = len(methods)
        line = 1
        ncols = 1
        nlines = nmethods
        # nbins = 15
        #dict(step : dict(method, data))
        # mboxes = {}
        try:
            pdf = PdfPages('checks/bw.pdf')
            for name, method in methods.iteritems():
                info("Result of measures for method %s:\n" % name)
                for pair in method['pairs']:
                    info(pair.printAll())
                avgs = {}
                ts = {}
                steps = zip(*method['real_steps'])
                step_time = np.array((0,) + steps[1])
                step_values = np.array((0,) + steps[0])

                for pair in method['pairs']:
                    avg = map(lambda measure: measure.bw / (1000 ** 2), pair.measures)

                    t = map(lambda measure: measure.timestamp, pair.measures)
                    avgs[pair.getPair()] = np.array(avg)
                    ts[pair.getPair()] = np.array(t)


                # plot the data
                Graph.subplot(nlines, ncols, line)
                for pair in method['pairs']:
                    Graph.plot(ts[pair.getPair()], avgs[pair.getPair()], color = Graph.Graph.getColor(pair.getPair()) + '.',
                             label = "%s,%s" % pair.getPair())
                    Graph.hold(True)
                Graph.step(step_time, step_values, 'r', where = 'post')
                Graph.xlabel('Time (s)', fontsize = 10)
                Graph.ylabel('BW estimation with %s (Mbps)' % name, fontsize = 10)
                ax = Graph.gca()
                ax.set_yscale('log')
                Graph.legend(loc = 2)
                Graph.draw()
                line += ncols
            fig = Graph.gcf()
            fig.set_size_inches(20, 20)
            pdf.savefig(bbox_inches = 'tight')  #'checks/delay.pdf', format = 'pdf', )
            Graph.close()

            d = pdf.infodict()
            d['Title'] = 'Delays measurement'
            d['Author'] = u'Francois Espinet'
            d['Subject'] = 'Delay measurement'
            d['Keywords'] = 'measurement delays'
            d['ModDate'] = datetime.datetime.today()
        finally:
            pdf.close()
        Graph.show()


class Delay(object):
    net = None
    PAIRS = [('h1', 'h7'), ('h2', 'h6')]
    #delays (in ms) to test
    STEPS = [0, 10, 50, 100, 500]
    SAMPLE_NUMBER = 50
    _FUDGE_FACTOR = 0.8

    #deadline after which the command terminates
    SAMPLE_DEADLINE = 5
    #number of packets per sample/baseline
    PACKET_NUMBER = 4
    BL_PACKET_NUMBER = 30
    #rate at which to send the packets
    SEND_SPEED = 0.3
    # consider packet lost if it arrives after WAIT_FACTOR * step * 2
    WAIT_FACTOR = 3

    #account for linux ICMP port unreachable message per second limit (in ms if > 10)
    UDP_SEND_WAIT = 900

    #formula for times (in seconds) : (time to send all probes + time to wait for the last reply) * number of pairs to test + fudge factor
    times = [SAMPLE_NUMBER * (int(PACKET_NUMBER * _FUDGE_FACTOR) + step * 2 / 1000) * len(PAIRS) for step in STEPS]
    time_start = None

    _START_WAIT = 0.1
    _STOP_WAIT = 0.2
    _SAMPLE_WAIT = 0.01
    methods = {
        'udp-trrt': {'method': Traceroute.ping,
                     'options': {'npackets': PACKET_NUMBER,
                                 'sendwait': UDP_SEND_WAIT},
                     'blOptions': {'npackets': BL_PACKET_NUMBER,
                                   'sendwait': UDP_SEND_WAIT}
        },
        'udplite-trrt': {'method': Traceroute.ping,
                         'options': {'npackets': PACKET_NUMBER,
                                     'proto': Traceroute.P_UDPLITE,
                                     'sendwait': UDP_SEND_WAIT},
                         'blOptions': {'npackets': BL_PACKET_NUMBER,
                                       'proto': Traceroute.P_UDPLITE,
                                       'sendwait': UDP_SEND_WAIT}
        },
        'icmp-trrt': {'method': Traceroute.ping,
                      'options': {'npackets': PACKET_NUMBER,
                                  'proto': Traceroute.P_ICMP},
                      'blOptions': {'npackets': BL_PACKET_NUMBER,
                                    'proto': Traceroute.P_ICMP}
        },
        'icmp-ping': {'method': Ping.ping,
                      'options': {'deadline': SAMPLE_DEADLINE,
                                  'npackets': PACKET_NUMBER,
                                  'sendspeed': SEND_SPEED},
                      'blOptions': {'npackets': BL_PACKET_NUMBER,
                                    'sendspeed': SEND_SPEED}
        },
        'tcp-trrt': {'method': Traceroute.ping,
                     'options': {'npackets': PACKET_NUMBER,
                                 'proto': Traceroute.P_TCP},
                     'blOptions': {'npackets': BL_PACKET_NUMBER,
                                   'proto': Traceroute.P_TCP}
        }
    }

    @classmethod
    def _getTimeStamp(cls):
        return time.time() - cls.time_start


    @classmethod
    def check(cls, net):
        cls.net = net
        # get a baseline
        pairs = cls._strToNodes(cls.PAIRS)
        for name, method in cls.methods.iteritems():
            method['pairs'] = [cls.HostStats(pair[0], pair[1], name) for pair in pairs]
        info("&&& Selected pairs : %s\n" % ", ".join(["(%s, %s)" % pair for pair in cls.PAIRS]))
        info("&&& Selected methods : %s\n" % ", ".join(cls.methods.keys()))
        cls._getBaselines(cls.methods)
        info("&&& Steps for this run %s, time to complete : %s \n" % (", ".join(["%sms" % step for step in cls.STEPS]),
                                                                      "%s" % datetime.timedelta(seconds = sum(cls.times) * len(cls.methods)) ))
        info("&&& Running tests\n")
        for name, method in cls.methods.iteritems():
            info("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
            info("&& Running test for method %s\n" % name)
            cls._runSteps(method)
        info("&&& All tests are done\n")
        cls.makeResults(cls.methods, saveResults = True)


    @classmethod
    def _runSteps(cls, method):
        cls.time_start = time.time()
        method['real_steps'] = []
        for step in cls.STEPS:
            cls.__runStep(step, method)
        #for step graph
        method['real_steps'].append((cls.STEPS[-1], cls._getTimeStamp()))
        cls._resetEvent()

    @classmethod
    def getWaitTime(cls, step):
        return 5.0  #float(2 * step * cls.WAIT_FACTOR) / 1000.0

    @classmethod
    def __runStep(cls, step, method):
        info("&& Testing next step %sms\n" % step)
        method['real_steps'].append((step, cls._getTimeStamp()))
        cls._makeEvent(step)
        time.sleep(cls._START_WAIT)
        method['options']['wait'] = cls.getWaitTime(step)
        cls._getSamples(step, method['pairs'], method['method'], method['options'])
        time.sleep(cls._STOP_WAIT)


    @classmethod
    def _getSamples(cls, step, pairs, method, options = {}):
        info("& Getting samples : ")
        for i in range(1, cls.SAMPLE_NUMBER + 1):
            info("%s " % i)
            for pair in pairs:
                ping = method(pair.host, pair.target, **options)
                ping.step = step
                ping.timestamp = cls._getTimeStamp()
                # if ping.sent > 0:
                pair.measures.append(ping)
                info("({:.2f}) ".format(ping.rttavg))
                time.sleep(cls._SAMPLE_WAIT)
        info("\n")

    @classmethod
    def _getBaselines(cls, methods):
        info("&& Getting baselines\n")
        for name, method in methods.iteritems():
            info("& Getting baseline for method %s\n" % name)
            cls.__setBaseline(method['pairs'], method['method'], method['blOptions'])
        info("&& Baselines done\n")

    @classmethod
    def __setBaseline(cls, pairs, method, options = {}):
        for pair in pairs:
            pair.baseline = method(pair.host, pair.target, **options)


    @classmethod
    def _makeEvent(cls, delay):
        events.runEvent(cls.__getEvent(delay), cls.net)

    @classmethod
    def __getEvent(cls, delay, target = 'l11'):
        return events.NetEvent(target = target,
                               variations = {'delay': "%sms" % delay})

    @classmethod
    def _resetEvent(cls, target = 'l11'):
        events.resetTarget(cls.net.get(target))


    @classmethod
    def _strToNodes(cls, pairs):
        return [(cls.net.getNodeByName(s1), cls.net.getNodeByName(s2)) for s1, s2 in pairs]

    class HostStats(object):
        """Storage for delay results"""

        def __init__(self, host, target, method = '', baseline = None, measures = []):
            self.host = host
            self.target = target
            self.measures = measures
            self.baseline = baseline
            self.method = method

        def subtrackBaseline(self, ping):
            return DelayStats(ping.timestamp,
                              ping.step,
                              ping.sent,
                              ping.received,
                              ping.rttmin - self.baseline.rttmin,
                              ping.rttavg - self.baseline.rttavg,
                              ping.rttmax - self.baseline.rttmax,
                              ping.rttdev)

        def getPair(self):
            return self.host.name, self.target.name

        def getStrPair(self):
            return "%s,%s" % self.getPair()

        def printAll(self):
            return "\n%s -> %s\nbaseline for %s: \n   %s\nmeasures : \n   %s\n" % (self.host.name,
                                                                                   self.target.name,
                                                                                   self.method,
                                                                                   self.baseline.printAll() if self.baseline is not None else '',
                                                                                   "\n   ".join([m.printAll() for m in self.measures]))

        def toDict(self):
            return {'host': self.host.name,
                    'target': self.target.name,
                    'baseline': self.baseline.toDict(),
                    'measures': [m.toDict() for m in self.measures]}

    @classmethod
    def saveResults(cls, methods):
        """Save results to json file
        :param methods: results to save"""
        info('Saving delay results\n')
        import json

        results = {}
        for name, method in methods.iteritems():
            results[name] = {}
            results[name]['pairs'] = []
            results[name]['real_steps'] = method['real_steps']
            for pair in method['pairs']:
                results[name]['pairs'].append(pair.toDict())
        fn = "checks/delay_%s.json" % datetime.datetime.now()
        json.dump(results, open(fn, 'w'))
        return fn

    @classmethod
    def loadResults(cls, jsonFile):
        """Load results from json file
        :param jsonFile : json format file to load from"""
        ms = json.load(open(jsonFile, 'r'))
        methods = {}
        class Host(object):
            def __init__(self, name):
                self.name = name
        for name, method_data in ms.iteritems():
            methods[name] = {}
            methods[name]['pairs'] = []
            methods[name]['real_steps'] = method_data['real_steps']
            for pairdata in method_data['pairs']:
                s = cls.HostStats(method = name,
                                  host = Host(pairdata['host']),
                                  target = Host(pairdata['target']),
                                  baseline = DelayStats(**pairdata['baseline']),
                                  measures = [DelayStats(**m) for m in pairdata['measures']])
                methods[name]['pairs'].append(s)
        info("Loading delay results done\n")
        return methods


    @classmethod
    def makeResults(cls, methods, saveResults = True):
        """Process results and produce graphics
        :param methods: results to save
        :param saveResults: save results to file ?"""
        if saveResults:
            try:
                fn = cls.saveResults(methods)
                info("Saved delay results to file %s\n" % fn)
            except Exception as e:
                error("Could not save delay results : %s\n" % e)

        import numpy as np
        from matplotlib.backends.backend_pdf import PdfPages

        nmethods = len(methods)
        line = 1
        ncols = 3 + len(cls.STEPS)
        nlines = nmethods
        nbins = 15
        #dict(step : dict(method, data))
        mboxes = {'baseline': {}}
        try:
            pdf = PdfPages('checks/delay.pdf')
            for name, method in methods.iteritems():
                info("Result of measures for method %s:\n" % name)
                for pair in method['pairs']:
                    info(pair.printAll())
                avgs = {}
                rdiffs = {}
                adiffs = {}
                devs = {}
                ts = {}
                steps = zip(*map(lambda x: (2 * x[0], x[1]), method['real_steps']))
                step_time = np.array((0,) + steps[1])
                step_values = np.array((0,) + steps[0])

                for pair in method['pairs']:
                    avg = map(lambda measure: measure.rttavg, pair.measures)
                    adiff = map(lambda measure: pair.subtrackBaseline(measure).rttavg - 2 * measure.step, pair.measures)
                    rdiff = map(lambda measure: abs(pair.subtrackBaseline(measure).rttavg) / (2 * measure.step + 1), pair.measures)
                    dev = map(lambda measure: measure.rttdev, pair.measures)
                    for measure in pair.measures:
                        if not mboxes.has_key(measure.step):
                            mboxes[measure.step] = {}
                        if not mboxes[measure.step].has_key(name):
                            mboxes[measure.step][name] = []
                        # mboxes[measure.step][name].append(measure.rttavg)
                        mboxes[measure.step][name].append(pair.subtrackBaseline(measure).rttavg)
                    if not mboxes['baseline'].has_key(pair.getPair()):
                        mboxes['baseline'][pair.getPair()] = {}
                    mboxes['baseline'][pair.getPair()][name] = (pair.baseline.rttavg, pair.baseline.rttdev)

                    t = map(lambda measure: measure.timestamp, pair.measures)
                    avgs[pair.getPair()] = np.array(avg)
                    rdiffs[pair.getPair()] = np.log10(np.array(rdiff))
                    # rdiffs[pair.getPair()] = np.array(rdiff)
                    adiffs[pair.getPair()] = np.array(adiff)
                    devs[pair.getPair()] = np.array(dev)
                    ts[pair.getPair()] = np.array(t)


                # plot the data
                Graph.subplot(nlines, ncols, line)
                for pair in method['pairs']:
                    Graph.errorbar(ts[pair.getPair()], avgs[pair.getPair()],
                                 yerr = devs[pair.getPair()],
                                 fmt = '.',
                                 color = Graph.getColor(pair.getPair()),
                                 label = "%s,%s" % pair.getPair())
                    Graph.hold(True)
                Graph.step(step_time, step_values, 'r', where = 'post',
                         g_xlabel = 'Time (s)',
                         g_ylabel = 'RTT time for %s' % name)
                Graph.legend(loc = 2)
                Graph.subplot(nlines, ncols, line + 1)
                n, bins, patches = Graph.hist(rdiffs.values(), nbins,
                                            normed = 1,
                                            label = ["%s,%s" % x for x in rdiffs.keys()],
                                            g_xlabel = 'Logarithmic Relative error')
                Graph.legend(loc = 'upper left', bbox_to_anchor = (0.9, 1.0), ncol = 1)
                # ax.set_xticklabels([lab.get_text() for lab in ax.get_xaxis().get_ticklabels()])
                Graph.subplot(nlines, ncols, line + 2)
                Graph.hist(adiffs.values(), nbins,
                           label = ["%s,%s" % x for x in adiffs.keys()],
                           g_xlabel = 'Absolute error')
                Graph.legend(loc = 'upper left', bbox_to_anchor = (0.9, 1.0), ncol = 1)
                # plt.hist(diffs.values(), stacked = True)
                # plt.xticks(bins, ["2^%s" % i for i in bins])
                # plt.hold(True)
                # plt.plot(steps_time, steps_val, 'r,-')
                #         plt.axis([0, 60, 0, 2000])
                # ax = plt.gca()
                # ax.yaxis.grid(True, linestyle = '-', which = 'major', color = 'lightgrey',
                #                alpha = 0.5)
                Graph.draw()
                line += ncols
            fig = Graph.gcf()
            fig.set_size_inches(50, 20)
            pdf.savefig(bbox_inches = 'tight')  #'checks/delay.pdf', format = 'pdf', )
            Graph.close()

            #compare methods to each other
            ncols = nmethods + 1
            Graph.clf()

            Graph.subplot(1, ncols, 1)
            bl = mboxes.pop('baseline')
            d = 1.0 / len(methods)
            #make sure we iterate with the same order over methods
            mets = sorted(methods.keys())
            for pair, data in bl.iteritems():
                vals = [[], []]
                for me in mets:
                    vals[0].append(data[me][0])
                    vals[1].append(data[me][1])
                Graph.errorbar([d + i for i in range(1, len(mets) + 1)],
                             vals[0],
                             yerr = vals[1],
                             fmt = '.',
                             color = Graph.getColor(pair),
                             label = '%s,%s' % pair)
                Graph.hold = True
                d += 1.0 / len(methods)
            for l in range(1, len(mets) + 1):
                Graph.axvspan(l, l + 1, facecolor = Graph.getColor(mets[l - 1]), alpha = 0.1, hold = True)  #, linestyle = '--')
            Graph.legend(loc = 2)
            Graph.decorate(g_xtickslab = ['', ] + mets, g_xticks = [0.5 + i for i in range(0, len(methods) + 1)],
                           g_grid = True,
                           g_xlabel = 'Measurement method', g_ylabel = 'Measured delays with stddev (ms)',
                           g_title = 'Baseline for all methods')

            nstep = 2
            for step in sorted(mboxes.keys()):
                m_datas = mboxes[step]
                Graph.subplot(1, ncols, nstep)
                Graph.boxplot([m_datas[met] for met in mets], sym = '^',
                              g_xtickslab = m_datas.keys(), g_grid = True,
                              g_xlabel = 'Measurement method', g_ylabel = 'Measured delays - baseline (ms)',
                              g_title = 'Measures for step 2x%sms' % step)

                Graph.axhline(2 * step, color = 'r')
                nstep += 1
            fig = Graph.gcf()
            fig.set_size_inches(35, 14)
            pdf.savefig(bbox_inches = 'tight')  #'checks/boxdelay.pdf', format = 'pdf', )
            Graph.close()

            d = pdf.infodict()
            d['Title'] = 'Delays measurement'
            d['Author'] = u'Francois Espinet'
            d['Subject'] = 'Delay measurement'
            d['Keywords'] = 'measurement delays'
            d['ModDate'] = datetime.datetime.today()
        finally:
            pdf.close()
        Graph.show()


class _PyplotGraph(type):
    """Interface with the pyplot object"""

    def __new__(mcs, *args, **kwargs):
        #import pyplot and register  it
        import matplotlib.pyplot as plt
        mcs.plt = plt
        return type.__new__(mcs, *args, **kwargs)

    def __getattr__(cls, item):
        def decorate( *args, **kwargs):
            o = getattr(cls.plt, item)(*args, **cls.decorate(g_filter = True,**kwargs))
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


if __name__ == "__main__":
    print("Making results from check/s/delay.json and check/s/bw.json")
    Delay.makeResults(Delay.loadResults('checks/s/delay.json'), saveResults = False)
    Bandwidth.makeResults(Bandwidth.loadResults('checks/s/bw.json'), saveResults = False)
