"""Module for checking properties on Mininet virtual network
Currently, bandwidth and delays can be check with the methods provided by
the measures module"""

import time
import datetime
import random
import os
import json
import traceback

from mininet.log import info, error
import events
from measures import Traceroute, Ping, DelayStats, IPerf, Spruce, IGI, Assolo, Abing, BwStats
import vars


def check(net, level, netchecks = None):
    """Check virtual network
    :param net : the network to test
    :param level : level of check, the higher the thorougher
    :param netchecks : check to perform
    """
    if level >= 1:
        net.pingAll()
    if level >= 2:
        if netchecks is not None:
            for ck in netchecks:
                checker = ck.pop('checker')
                if checker == 'delay':
                    checkDelay(net, ck)
                elif checker == 'bw':
                    checkBw(net, ck)
                else:
                    error("Unknown checker provided\n")


def checkDelay(net, checkParams):
    """Check delays on net
    :param net : net to check
    :param checkParams: parameters for check
    """
    info("Checking delays consistency.\n")
    try:
        Delay(net = net, **checkParams).check()
    except Exception as e:
        error('Could not check delays %s\n' % e)
        traceback.print_exc()


def checkBw(net, checkParams):
    """Check Bandwidth on net
    :param net : net to check
    :param checkParams: parameter for check
    """
    info("Checking bandwidth consistency.\n")
    try:
        Bandwidth(net = net, **checkParams).check()
    except Exception as e:
        error('Could not check bandwidth %s\n' % e)
        traceback.print_exc()


class Bandwidth(object):
    """Set of bandwidth check to perform"""
    SAMPLE_NUMBER = 'sample_number'

    _START_WAIT = 0.1
    _STOP_WAIT = 0.2
    _SAMPLE_WAIT = 0.01

    DEFAULT_OPTIONS = {
        'sample_number': 20
    }

    def __init__(self, net = None, affected_check = None, unaffected_check = None,
                 name = None, targets = None,
                 options = None):
        self.net = net
        self.checkName = name
        self.PAIRS = map(tuple, affected_check)
        #number of packets per sample/baseline
        self.PACKET_NUMBER = 4
        #bitrate in Mbps
        self.STEPS = {}
        for target, tops in targets.iteritems():
            self.STEPS[target] = tops['steps']

        self.steps_len = min([len(steps) for steps in self.STEPS.values()])
        self.options = self.DEFAULT_OPTIONS.copy()
        self.options.update(options)
        self.methods = {
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
            # 'yaz': {
            #     'method': Yaz.bw,
            #     'options': {'binDir': os.path.join(vars.testBinPath)}
            # }
        }
        # for val in self.methods.values():
        #     opt = val['options']
        #     opt.update(val['blOptions'])
        #     val['blOptions'] = opt
        self.time_start = None

    def _getTimeStamp(self):
        return time.time() - self.time_start


    def check(self):
        """Check the net"""
        info("Starting check %s\n" % self.checkName)
        # get a baseline
        pairs = self._strToNodes(self.PAIRS)
        for name, method in self.methods.iteritems():
            method['pairs'] = [self.HostStats(pair[0], pair[1], name) for pair in pairs]
        info("&&& Selected pairs : %s\n" % ", ".join(["(%s, %s)" % pair for pair in self.PAIRS]))
        info("&&& Selected methods : %s\n" % ", ".join(self.methods.keys()))
        self._getBaselines(self.methods)
        info("&&& Steps for this run \n%s" % "\n ".join(
            ["\t%s : %s" % (link,
                            " ".join(
                                ["%sMbps" % step for step in steps]))
             for link, steps in self.STEPS.iteritems()]))
        info("\n")
        try:
            info("&&& Running tests\n")
            for name, method in self.methods.iteritems():
                info("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
                info("&& Running test for method %s\n" % name)
                self._runSteps(method)
            info("&&& All tests are done\n")
        finally:
            self.makeResults(self.methods, checkName = self.checkName, saveResults = True)

    def _runSteps(self, method):
        self.time_start = time.time()
        method['real_steps'] = {target: [] for target in self.STEPS.keys()}
        for stepNum in range(0, self.steps_len):
            self.__runStep(stepNum, method)
        #for step graph
        for target in self.STEPS.keys():
            method['real_steps'][target].append((self.STEPS[target][-1], self._getTimeStamp()))
        for target in self.STEPS.keys():
            self._resetEvent(target)

    def __runStep(self, stepNum, method):
        info("&& Testing next step : \n")
        step = 0
        for target, steps in self.STEPS.iteritems():
            info("\tlink : %s, bw %sMbps " % (target, steps[stepNum]))
            method['real_steps'][target].append((steps[stepNum], self._getTimeStamp()))
            self._makeEvent(steps[stepNum], target)
            step = min(steps[stepNum], step) if step > 0 else steps[stepNum]
        time.sleep(self._START_WAIT)
        self.__getSamples(step, method['pairs'], method['method'], **method['options'])
        time.sleep(self._STOP_WAIT)


    def __getSamples(self, step, pairs, method, **options):
        info("& Getting samples : ")
        options['bw'] = "%sM" % step
        for i in range(1, self.options[self.SAMPLE_NUMBER] + 1):
            info("%s " % i)
            for pair in pairs:
                bw = method((pair.host, pair.target), **options)
                bw.step = step
                bw.timestamp = self._getTimeStamp()
                # if ping.sent > 0:
                pair.measures.append(bw)
                info("({:.2f}M) ".format(bw.bw / (1000.0 ** 2)))
                time.sleep(self._SAMPLE_WAIT)
        info("\n")


    def _getBaselines(self, methods):
        info("&& Getting baselines\n")
        for name, method in methods.iteritems():
            info("& Getting baseline for method %s : " % name)
            self.__setBaseline(method['pairs'], method['method'], **method['options'])
        info("&& Baselines done\n")

    @staticmethod
    def __setBaseline(pairs, method, **options):
        for pair in pairs:
            pair.baseline = method((pair.host, pair.target), **options)
            info(' %s : %s' % ("(%s,%s)" % pair.getPair(), "{:.2f}".format(pair.baseline.bw / (1000.0 ** 2))))
        info("\n")


    def _makeEvent(self, delay, target):
        events.runEvent(self.__getEvent(delay, target), self.net)

    @staticmethod
    def __getEvent(bw, target = 'l11'):
        return events.NetEvent(target = target,
                               variations = {'bw': bw})

    def _resetEvent(self, target = 'l11'):
        events.resetTarget(self.net.get(target))


    def _strToNodes(self, pairs):
        return [(self.net.getNodeByName(s1), self.net.getNodeByName(s2)) for s1, s2 in pairs]

    class HostStats(object):
        """Storage for results"""

        def __init__(self, host, target, method = '', baseline = None, measures = None):
            self.host = host
            self.target = target
            self.measures = measures if measures is not None else []
            self.baseline = baseline
            self.method = method

        def subtrackBaseline(self, sample):
            """Remove baseline from sample
            :param sample : sample to correct
            """
            return DelayStats(sample.timestamp,
                              sample.step,
                              sample.sent,
                              sample.received,
                              sample.bw - self.baseline.bw)

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
    def saveResults(cls, methods, checkName):
        """Save results to json file
        :param methods: results to save
        :param checkName : name of the check to save"""
        info('Saving bandwidth results\n')
        import json

        results = {}
        for name, method in methods.iteritems():
            results[name] = {}
            results[name]['pairs'] = []
            results[name]['real_steps'] = method['real_steps']
            for pair in method['pairs']:
                results[name]['pairs'].append(pair.toDict())
        fn = "checks/%s_%s.json" % (checkName, datetime.datetime.now())
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
                                  baseline = BwStats(**pairdata['baseline']),
                                  measures = [BwStats(**m) for m in pairdata['measures']])
                methods[name]['pairs'].append(s)
        info("Loading bw results done.\n")
        return methods

    @classmethod
    def makeResults(cls, methods, checkName = 'bw', saveResults = True):
        """Make result to graphics
        :param methods : results to process
        :param checkName: name of the current check being processed
        :param saveResults : save results to json file ?"""
        if saveResults:
            try:
                fn = cls.saveResults(methods, checkName)
                info("Saved bandwidth results to file %s\n" % fn)
            except Exception as e:
                error("Could not save bandwidth results : %s\n" % e)

        import numpy as np
        from matplotlib.backends.backend_pdf import PdfPages

        try:
            for name, method in methods.iteritems():
                info("Result of measures for method %s:\n" % name)
                for pair in method['pairs']:
                    info(pair.printAll())
        except Exception as e:
            error("Could not print results %s\n" % e)

        nmethods = len(methods)
        line = 1
        ncols = 1
        nlines = nmethods
        # nbins = 15
        #dict(step : dict(method, data))
        # mboxes = {}
        try:
            fn = 'checks/%s.pdf' % checkName
            pdf = PdfPages(fn)
            for name, method in methods.iteritems():
                info("Result of measures for method %s:\n" % name)
                avgs = {}
                ts = {}
                steps = {'total': None}
                for target, tsteps in method['real_steps'].iteritems():
                    st = zip(*tsteps)
                    step_time = np.array((0,) + st[1])
                    step_values = np.array((0,) + st[0])
                    steps[target] = (step_time, step_values)
                    steps['total'] = (step_time, np.minimum(steps['total'][1], step_values)) if steps['total'] is not None else (step_time, step_values)

                for pair in method['pairs']:
                    avg = map(lambda measure: measure.bw / (1000 ** 2), pair.measures)

                    t = map(lambda measure: measure.timestamp, pair.measures)
                    avgs[pair.getPair()] = np.array(avg)
                    ts[pair.getPair()] = np.array(t)


                # plot the data
                Graph.subplot(nlines, ncols, line)
                for pair in method['pairs']:
                    Graph.scatter(ts[pair.getPair()], avgs[pair.getPair()],
                                  color = Graph.getColor(pair.getPair()),
                                  label = "%s,%s" % pair.getPair())
                    Graph.hold = True
                for target, tsteps in steps.iteritems():
                    Graph.step(tsteps[0], tsteps[1], 'r', where = 'post', label = target, color = Graph.getColor(target))
                    Graph.hold = True
                Graph.decorate(g_xlabel = 'Time (s)',
                               g_ylabel = 'BW estimation with %s (Mbps)' % name)
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
            info("Saved graphics to %s\n" % fn)
        Graph.show()


class Delay(object):
    _FUDGE_FACTOR = 0.8

    SAMPLE_NUMBER = 'sample_number'
    PACKET_NUMBER = 'packet_number'
    #deadline after which the command terminates
    SAMPLE_DEADLINE = 5

    BL_PACKET_NUMBER = 40
    #rate at which to send the packets
    SEND_SPEED = 0.3
    # consider packet lost if it arrives after WAIT_FACTOR * step * 2
    WAIT_FACTOR = 3

    #account for linux ICMP port unreachable message per second limit (in ms if > 10)
    UDP_SEND_WAIT = 900

    _START_WAIT = 0.1
    _STOP_WAIT = 0.2
    _SAMPLE_WAIT = 0.01

    DEFAULT_OPTIONS = {
        'sample_number': 50,
        #number of packets per sample
        'packet_number': 4
    }

    def __init__(self, net = None, affected_check = None, unaffected_check = None,
                 name = None, targets = None,
                 options = None):
        self.net = net
        self.checkName = name
        self.PAIRS = map(tuple, affected_check)
        #delays (in ms) to test
        self.STEPS = {}
        for target, tops in targets.iteritems():
            self.STEPS[target] = tops['steps']
        self.steps_len = min([len(steps) for steps in self.STEPS.values()])
        self.options = self.DEFAULT_OPTIONS.copy()
        self.options.update(options)
        self.methods = {
            'udp-trrt': {'method': Traceroute.ping,
                         'options': {'npackets': self.options[self.PACKET_NUMBER],
                                     'sendwait': self.UDP_SEND_WAIT,
                                     'binDir': vars.testBinPath,
                                     'proto': Traceroute.P_UDP},
                         'blOptions': {'npackets': self.BL_PACKET_NUMBER}
            },
            'udplite-trrt': {'method': Traceroute.ping,
                             'options': {'npackets': self.options[self.PACKET_NUMBER],
                                         'binDir': vars.testBinPath,
                                         'proto': Traceroute.P_UDPLITE},
                             'blOptions': {'npackets': self.BL_PACKET_NUMBER, }
            },
            'icmp-trrt': {'method': Traceroute.ping,
                          'options': {'npackets': self.options[self.PACKET_NUMBER],
                                      'proto': Traceroute.P_ICMP,
                                      'binDir': vars.testBinPath},
                          'blOptions': {'npackets': self.BL_PACKET_NUMBER}
            },
            'icmp-ping': {'method': Ping.ping,
                          'options': {'deadline': self.SAMPLE_DEADLINE,
                                      'npackets': self.options[self.PACKET_NUMBER],
                                      'sendspeed': self.SEND_SPEED},
                          'blOptions': {'npackets': self.BL_PACKET_NUMBER}
            },
            'tcp-trrt': {'method': Traceroute.ping,
                         'options': {'npackets': self.options[self.PACKET_NUMBER],
                                     'proto': Traceroute.P_TCP,
                                     'binDir': vars.testBinPath},
                         'blOptions': {'npackets': self.BL_PACKET_NUMBER}
            }
        }
        for val in self.methods.values():
            opt = val['options'].copy()
            opt.update(val['blOptions'])
            val['blOptions'] = opt
        self.time_start = None

    def _getTimeStamp(self):
        return time.time() - self.time_start


    def check(self):
        """Check the network"""
        info('&& Starting check %s\n' % self.checkName)
        # get a baseline
        pairs = self._strToNodes(self.PAIRS)
        for name, method in self.methods.iteritems():
            method['pairs'] = [self.HostStats(pair[0], pair[1], name) for pair in pairs]
        info("&&& Selected pairs : %s\n" % ", ".join(["(%s, %s)" % pair for pair in self.PAIRS]))
        info("&&& Selected methods : %s\n" % ", ".join(self.methods.keys()))
        self._getBaselines(self.methods)
        info("&&& Steps for this run \n%s" % "\n ".join(
            ["\t%s : %s" % (link,
                            " ".join(
                                ["%sms" % step for step in steps]))
             for link, steps in self.STEPS.iteritems()]))
        info("\n")
        try:
            info("&&& Running tests\n")
            for name, method in self.methods.iteritems():
                info("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
                info("&& Running test for method %s\n" % name)
                self._runSteps(method)
            info("&&& All tests are done\n")
        finally:
            info("&&& Making results\n")
            self.makeResults(self.methods, checkName = self.checkName, saveResults = True)


    def _runSteps(self, method):
        self.time_start = time.time()
        method['real_steps'] = {target: [] for target in self.STEPS.keys()}
        for stepNum in range(0, self.steps_len):
            self.__runStep(stepNum, method)
        #for step graph
        for target in self.STEPS.keys():
            method['real_steps'][target].append((self.STEPS[target][-1], self._getTimeStamp()))
        for target in self.STEPS.keys():
            self._resetEvent(target)

    def getWaitTime(self, step):
        return 5.0  #float(2 * step * self.WAIT_FACTOR) / 1000.0

    def __runStep(self, stepNum, method):
        info("&& Testing next step : \n")
        step = 0
        for target, steps in self.STEPS.iteritems():
            info("\tlink : %s, delay %sms " % (target, steps[stepNum]))
            method['real_steps'][target].append((steps[stepNum], self._getTimeStamp()))
            self._makeEvent(steps[stepNum], target)
            step += steps[stepNum]
        time.sleep(self._START_WAIT)
        method['options']['wait'] = self.getWaitTime(step)
        self._getSamples(step, method['pairs'], method['method'], **method['options'])
        time.sleep(self._STOP_WAIT)


    def _getSamples(self, step, pairs, method, **options):
        info("& Getting samples : ")
        for i in range(1, self.options[self.SAMPLE_NUMBER] + 1):
            info("%s " % i)
            for pair in pairs:
                ping = method((pair.host, pair.target), **options)
                ping.step = step
                ping.timestamp = self._getTimeStamp()
                # if ping.sent > 0:
                pair.measures.append(ping)
                info("({:.2f}) ".format(ping.rttavg))
                time.sleep(self._SAMPLE_WAIT)
        info("\n")

    def _getBaselines(self, methods):
        info("&& Getting baselines\n")
        for name, method in methods.iteritems():
            info("& Getting baseline for method %s :" % name)
            self.__setBaseline(method['pairs'], method['method'], **method['blOptions'])
            info("\n")
        info("&& Baselines done\n")

    @staticmethod
    def __setBaseline(pairs, method, **options):
        for pair in pairs:
            pair.baseline = method((pair.host, pair.target), **options)
            info(' %s : %s' % ("(%s,%s)" % pair.getPair(), "{:.2f}".format(pair.baseline.rttavg)))

    def _makeEvent(self, delay, target):
        events.runEvent(self.__getEvent(delay, target), self.net)

    @staticmethod
    def __getEvent(delay, target = 'l11'):
        return events.NetEvent(target = target,
                               variations = {'delay': "%sms" % delay})

    def _resetEvent(self, target = 'l11'):
        events.resetTarget(self.net.get(target))

    def _strToNodes(self, pairs):
        return [(self.net.getNodeByName(s1), self.net.getNodeByName(s2)) for s1, s2 in pairs]

    class HostStats(object):
        """Storage for delay results"""

        def __init__(self, host, target, method = '', baseline = None, measures = None):
            self.host = host
            self.target = target
            if measures is not None:
                self.measures = measures
            else:
                self.measures = []
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
    def saveResults(cls, methods, checkName = 'delay'):
        """Save results to json file
        :param methods: results to save
        :param checkName : name of the check to save"""
        info('Saving delay results\n')
        import json

        results = {}
        for name, method in methods.iteritems():
            results[name] = {}
            results[name]['pairs'] = []
            results[name]['real_steps'] = method['real_steps']
            for pair in method['pairs']:
                results[name]['pairs'].append(pair.toDict())
        fn = "checks/%s_%s.json" % (checkName, datetime.datetime.now())
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
    def makeResults(cls, methods, checkName = 'delay', saveResults = True):
        """Process results and produce graphics
        :param methods: results to save
        :param checkName: name of the check
        :param saveResults: save results to file ?"""
        if saveResults:
            try:
                fn = cls.saveResults(methods, checkName)
                info("Saved delay results to file %s\n" % fn)
            except Exception as e:
                error("Could not save delay results : %s\n" % e)

        try:
            for name, method in methods.iteritems():
                info("Result of measures for method %s : " % name)
                for pair in method['pairs']:
                    info(pair.printAll())
                info("\n")
        except Exception as e:
            error("Could not print results : %s.\n" % e)

        import numpy as np
        from matplotlib.backends.backend_pdf import PdfPages

        nSteps = max([
            max([len(steps) for steps in method['real_steps'].values()])
            for method in methods.values()])
        nmethods = len(methods)
        line = 1
        ncols = 3 + nSteps
        nlines = nmethods
        nbins = 15
        #dict(step : dict(method, data))
        mboxes = {'baseline': {}}
        try:
            fn = 'checks/%s.pdf' % checkName
            pdf = PdfPages(fn)
            for name, method in methods.iteritems():
                avgs = {}
                rdiffs = {}
                adiffs = {}
                devs = {}
                ts = {}
                steps = {'total': None}
                for target, tsteps in method['real_steps'].iteritems():
                    st = zip(*map(lambda x: (2 * x[0], x[1]), tsteps))
                    step_time = np.array((0,) + st[1])
                    step_values = np.array((0,) + st[0])
                    steps[target] = (step_time, step_values)
                    steps['total'] = (step_time, np.add(steps['total'][1], step_values)) if steps['total'] is not None else (step_time, step_values)

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
                    Graph.hold = True
                for target, tsteps in steps.iteritems():
                    Graph.step(tsteps[0], tsteps[1], 'r', where = 'post', label = target, color = Graph.getColor(target))
                    Graph.hold = True
                Graph.decorate(g_xlabel = 'Time (s)',
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
            d = 1.0 / (2 * len(bl))
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
                d += 1.0 / len(bl)
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
                Graph.boxplot([m_datas[met] for met in mets], sym = '^')
                Graph.axhline(2 * step, color = 'r')
                Graph.decorate(g_xtickslab = mets, g_grid = True,
                               g_xlabel = 'Measurement method', g_ylabel = 'Measured delays - baseline (ms)',
                               g_title = 'Measures for step 2x%sms' % step)
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
            info("Saved graphics to %s\n" % fn)
        Graph.show()


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


if __name__ == "__main__":
    print("Making results from check/s/delay.json and check/s/bw.json")
    Delay.makeResults(Delay.loadResults('checks/s/delay.json'), checkName = 'sdelay', saveResults = False)
    Bandwidth.makeResults(Bandwidth.loadResults('checks/s/bw.json'), checkName = 'sbw', saveResults = False)
