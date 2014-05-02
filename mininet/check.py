__author__ = 'francois'

import time
import datetime
from threading import Timer, Event, Condition, Thread
import random
import math

from mininet.log import info, debug
import events
from measures import Traceroute, Ping, DelayStats


colors = 'bgcmyk'


def check(net, level):
    if level >= 1:
        net.pingAll()
    if level >= 2:
        checkDelay(net)
        # checkLoss(net)
        checkBw(net)


def getColor(item = None):
    if item is None:
        return colors[random.randint(0, len(colors) - 1)]
    return colors[hash(item) % len(colors)]


def checkDelay(net):
    info("Checking delays consistency.\n")
    Delay.check(net)


def checkLoss(net):
    info("Checking loss consistency.\n")
    # Loss.check(net)


def checkBw(net):
    pass


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
    BL_PACKET_NUMBER = 10
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
        'udp': {'method': Traceroute.ping,
                'options': {'npackets': PACKET_NUMBER,
                            'sendwait': UDP_SEND_WAIT},
                'blOptions': {'npackets': BL_PACKET_NUMBER,
                              'sendwait': UDP_SEND_WAIT}
        },
        'udplite': {'method': Traceroute.ping,
                    'options': {'npackets': PACKET_NUMBER,
                                'proto': Traceroute.P_UDPLITE,
                                'sendwait': UDP_SEND_WAIT},
                    'blOptions': {'npackets': BL_PACKET_NUMBER,
                                  'proto': Traceroute.P_UDPLITE,
                                  'sendwait': UDP_SEND_WAIT}
        },
        'traceping': {'method': Traceroute.ping,
                      'options': {'npackets': PACKET_NUMBER,
                                  'proto': Traceroute.P_ICMP},
                      'blOptions': {'npackets': BL_PACKET_NUMBER,
                                    'proto': Traceroute.P_ICMP}
        },
        'ping': {'method': Ping.ping,
                 'options': {'deadline': SAMPLE_DEADLINE,
                             'npackets': PACKET_NUMBER,
                             'sendspeed': SEND_SPEED},
                 'blOptions': {'npackets': BL_PACKET_NUMBER,
                               'sendspeed': SEND_SPEED}
        },
        'tcp': {'method': Traceroute.ping,
                'options': {'npackets': PACKET_NUMBER,
                            'proto': Traceroute.P_TCP},
                'blOptions': {'npackets': BL_PACKET_NUMBER,
                              'proto': Traceroute.P_TCP}
        }
    }

    @classmethod
    def getTimeStamp(cls):
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
        cls.getBaselines(cls.methods)
        info("&&& Steps for this run %s, time to complete : %s \n" % (", ".join(["%sms" % step for step in cls.STEPS]),
                                                                      "%s" % datetime.timedelta(seconds = sum(cls.times) * len(cls.methods)) ))
        info("&&& Running tests\n")
        for name, method in cls.methods.iteritems():
            info("&& Running test for method %s\n" % name)
            cls.runSteps(method)
        info("&&& All tests are done\n")
        cls.makeResults(cls.methods, saveResults = True)


    @classmethod
    def runSteps(cls, method):
        cls.time_start = time.time()
        method['real_steps'] = []
        for step in cls.STEPS:
            cls.runStep(step, method)
        #for step graph
        method['real_steps'].append((cls.STEPS[-1], cls.getTimeStamp()))
        cls.resetEvent()

    @classmethod
    def getWaitTime(cls, step):
        return 5.0  #float(2 * step * cls.WAIT_FACTOR) / 1000.0

    @classmethod
    def runStep(cls, step, method):
        info("&& Testing next step %sms\n" % step)
        method['real_steps'].append((step, cls.getTimeStamp()))
        cls.makeEvent("%sms" % step)
        time.sleep(cls._START_WAIT)
        method['options']['wait'] = cls.getWaitTime(step)
        cls.getSamples(step, method['pairs'], method['method'], method['options'])
        time.sleep(cls._STOP_WAIT)


    @classmethod
    def getSamples(cls, step, pairs, method, options = {}):
        info("& Getting samples : ")
        for i in range(1, cls.SAMPLE_NUMBER + 1):
            info("%s " % i)
            for pair in pairs:
                ping = method(pair.host, pair.target, **options)
                ping.step = step
                ping.timestamp = cls.getTimeStamp()
                # if ping.sent > 0:
                pair.measures.append(ping)
                debug(":{:.2f}, ".format(ping.rttavg))
                time.sleep(cls._SAMPLE_WAIT)
        info("\n")

    @classmethod
    def getBaselines(cls, methods):
        info("&& Getting baselines\n")
        for name, method in methods.iteritems():
            info("& Getting baseline for method %s\n" % name)
            cls.setBaseline(method['pairs'], method['method'], method['blOptions'])
        info("&& Baselines done\n")

    @classmethod
    def setBaseline(cls, pairs, method, options = {}):
        for pair in pairs:
            pair.baseline = method(pair.host, pair.target, **options)


    @classmethod
    def makeEvent(cls, delay):
        events.runEvent(cls._getEvent(delay), cls.net)

    @classmethod
    def _getEvent(cls, delay, target = 'l11'):
        return events.NetEvent(target = target,
                               variations = {'delay': delay})

    @classmethod
    def resetEvent(cls, target = 'l11'):
        events.resetTarget(cls.net.get(target))


    @classmethod
    def _strToNodes(cls, pairs):
        return [(cls.net.getNodeByName(s1), cls.net.getNodeByName(s2)) for s1, s2 in pairs]

    class HostStats(object):
        def __init__(self, host, target, method = ''):
            self.host = host
            self.target = target
            self.measures = []
            self.baseline = None
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
                                                                                   self.baseline.printAll(),
                                                                                   "\n   ".join([m.printAll() for m in self.measures]))

        def toDict(self):
            return {'host': self.host.name,
                    'target': self.target.name,
                    'baseline': self.baseline.toDict(),
                    'measures': [m.toDict() for m in self.measures]}

    @classmethod
    def saveResults(cls, methods):
        info('Saving results\n')
        import json

        results = {}
        for name, method in methods.iteritems():
            results[name] = {}
            results[name]['pairs'] = []
            results[name]['real_steps'] = method['real_steps']
            for pair in method['pairs']:
                results[name]['pairs'].append(pair.toDict())
        json.dump(results, open('checks/results.json', 'w'))

    @classmethod
    def makeResults(cls, methods, saveResults = False):
        if saveResults:
            cls.saveResults(methods)

        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        nmethods = len(methods)
        line = 1
        ncols = 3 + len(cls.STEPS)
        nlines = nmethods
        nbins = 15
        #dict(step : dict(method, data))
        mboxes = {}
        try :
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

                    t = map(lambda measure: measure.timestamp, pair.measures)
                    avgs[pair.getPair()] = np.array(avg)
                    rdiffs[pair.getPair()] = np.array(rdiff)
                    adiffs[pair.getPair()] = np.array(adiff)
                    devs[pair.getPair()] = np.array(dev)
                    ts[pair.getPair()] = np.array(t)


                # plot the data
                plt.subplot(nlines, ncols, line)
                for pair in method['pairs']:
                    plt.errorbar(ts[pair.getPair()], avgs[pair.getPair()], yerr = devs[pair.getPair()], fmt = getColor(pair.getPair()) + '.',
                                 label = "%s,%s" % pair.getPair())
                    plt.hold(True)
                plt.step(step_time, step_values, 'r', where = 'post')
                plt.xlabel('Time (s)', fontsize = 10)
                plt.ylabel('RTT time for %s' % name, fontsize = 10)
                plt.legend(loc = 2)
                plt.subplot(nlines, ncols, line + 1)
                plt.hist(rdiffs.values(), nbins, normed = 1,
                         label = ["%s,%s" % x for x in rdiffs.keys()])
                plt.legend(loc = 'upper left', bbox_to_anchor = (0.9, 1.0), ncol = 1)
                plt.xlabel('Relative error')
                plt.subplot(nlines, ncols, line + 2)
                plt.hist(adiffs.values(), nbins,
                         label = ["%s,%s" % x for x in adiffs.keys()])
                plt.legend(loc = 'upper left', bbox_to_anchor = (0.9, 1.0), ncol = 1)
                plt.xlabel('Absolute error')
                # plt.hist(diffs.values(), stacked = True)
                # plt.xticks(bins, ["2^%s" % i for i in bins])
                # plt.hold(True)
                # plt.plot(steps_time, steps_val, 'r,-')
                #         plt.axis([0, 60, 0, 2000])
                # ax = plt.gca()
                # ax.yaxis.grid(True, linestyle = '-', which = 'major', color = 'lightgrey',
                #                alpha = 0.5)
                plt.draw()
                line += ncols
            fig = plt.gcf()
            fig.set_size_inches(50, 20)
            pdf.savefig(bbox_inches = 'tight')#'checks/delay.pdf', format = 'pdf', )
            plt.close()

            #compare methods to each other
            plt.clf()
            nstep = 1
            for step in sorted(mboxes.keys()):
                m_datas = mboxes[step]
                plt.subplot(1, nmethods, nstep)
                plt.boxplot(m_datas.values(), sym = '^')
                ax = plt.gca()
                ax.set_xticklabels(m_datas.keys())
                ax.yaxis.grid(True, linestyle = '-', which = 'major', color = 'lightgrey',alpha = 0.5)
                plt.xlabel('Measurement method')
                plt.ylabel('Measured delays - baseline')
                plt.title('Measures for step 2x%sms'%step)
                nstep += 1
            fig = plt.gcf()
            fig.set_size_inches(30, 14)
            pdf.savefig(bbox_inches = 'tight')#'checks/boxdelay.pdf', format = 'pdf', )
            plt.close()

            d = pdf.infodict()
            d['Title'] = 'Delays measurement'
            d['Author'] = u'Francois Espinet'
            d['Subject'] = 'Delay measurement'
            d['Keywords'] = 'measurement delays'
            d['ModDate'] = datetime.datetime.today()
        finally:
            pdf.close()
        plt.show()


class Loss(object):
    net = None
    #(delay (in ms), duration)
    # STEPS = [(0, 10), (0.01, 10), (0.1, 10)]#, (1, 30), (5, 40), (0, 0)]
    STEPS = [(0, 10), (0.01, 10), (0.1, 10), (1, 10), (5, 10)]
    #wait at most WAIT_TIME second before considering packet lost
    WAIT_TIME = 0.001
    #wait no more than 60 seconds for each test
    DEADLINE = 60
    PACKET_NUMBER = 1000
    SEND_SPEED = 0.21

    real_steps = []
    time_start = None
    step_counter = 0
    test_done = None
    step = None
    step_up_done = Event()
    tests_lock = Condition()
    tests = 0

    methods = {
        'udp': {'method': Traceroute.loss,
                'options': {'npackets': '100',
                            'proto': Traceroute.P_UDP}
        },
        'ping': {'method': Ping.ping,
                 'options': {'deadline': DEADLINE,
                             'npackets': PACKET_NUMBER,
                             'wait': WAIT_TIME,
                             'sendspeed': SEND_SPEED}
        },
        'tcp': {'method': Traceroute.loss,
                'options': {'npackets': '100',
                            'proto': Traceroute.P_TCP}
        },
        'traceping': {'method': Traceroute.loss,
                      'options': {'npackets': '100',
                                  'proto': Traceroute.P_ICMP}
        },
    }

    @classmethod
    def getTimeStamp(cls):
        return time.time() - cls.time_start


    @classmethod
    def getStep(cls):
        return cls.step[0]


    @classmethod
    def check(cls, net):
        cls.net = net
        # get a baseline
        p = [('h1', 'h7'), ('h2', 'h6')]
        p = cls._strToNodes(p)
        for name, method in cls.methods.iteritems():
            method['pairs'] = [cls.HostStats(pair[0], pair[1], name) for pair in p]

        info("Steps for this run %s\n" % repr(cls.STEPS))
        info("Running test\n")
        cls.runStepChain()
        for method in cls.methods.viewvalues():
            t = Thread(target = cls.measureLoss, args = [method['pairs'], method['method'], method['options']])
            t.daemon = True
            t.start()
        cls.test_done.wait()
        info("Tests are done\n")
        cls.makeResults(cls.methods)

    @classmethod
    def runStepChain(cls):
        cls.tests = 0
        cls.time_start = time.time()
        cls.step_counter = 0
        cls.test_done = Event()
        cls.stepUp()

    @classmethod
    def stepUp(cls):
        info("Waiting for next step\n")
        cls.step_up_done.clear()
        cls._waitTests()
        if cls.step_counter >= len(cls.STEPS):
            cls.test_done.set()
            #for step graph
            cls.real_steps.append((cls.getStep(), cls.getTimeStamp()))
            return
        step = cls.STEPS[cls.step_counter]
        info("Testing next step %s\n" % repr(step))
        cls.step = step
        cls.real_steps.append((step[0], cls.getTimeStamp()))
        cls.makeEvent(step[0])
        cls.step_counter += 1
        cls.step_up_done.set()
        Timer(step[1], cls.stepUp).start()

    @classmethod
    def _waitTests(cls):
        with cls.tests_lock:
            while cls.tests > 0:
                cls.tests_lock.wait()

    @classmethod
    def _addTest(cls):
        with cls.tests_lock:
            cls.tests += 1
            cls.tests_lock.notifyAll()

    @classmethod
    def _rmTest(cls):
        with cls.tests_lock:
            cls.tests -= 1
            cls.tests_lock.notifyAll()

    @classmethod
    def _strToNodes(cls, pairs):
        return [(cls.net.getNodeByName(s1), cls.net.getNodeByName(s2)) for s1, s2 in pairs]

    @classmethod
    def measureLoss(cls, pairs, method, options = {}):
        for pair in pairs:
            t = Thread(target = cls.measureLossPair, args = [pair, method, options])
            t.daemon = True
            t.start()

    @classmethod
    def measureLossPair(cls, pair, method, options = {}):
        while not cls.test_done.is_set():
            info("Starting a loss test\n")
            try:
                cls.step_up_done.wait()
                cls._addTest()
                step = cls.getStep()
                ping = method(pair.host, pair.target, **options)
                ping.step = step
                ping.timestamp = cls.getTimeStamp()
                if ping.sent > 0:
                    pair.measures.append(ping)
            finally:
                cls._rmTest()


    @classmethod
    def makeEvent(cls, loss):
        events.runEvent(cls._getEvent(loss), cls.net)

    @classmethod
    def _getEvent(cls, loss, target = 'l11'):
        return events.NetEvent(target = target,
                               variations = {'loss': loss})


    @classmethod
    def makeResults(cls, methods):
        import matplotlib.pyplot as plt
        import numpy as np

        line = 1
        ncols = 2
        nmethods = len(methods)
        nbins = 15
        st = zip(*map(lambda x: (2 * x[0], x[1]), cls.real_steps))
        step_time = np.array((0,) + st[1])
        step_values = np.array((0,) + st[0])
        for name, method in methods.iteritems():
            losses = {}
            diffs = {}
            ts = {}
            info("Result of measures :\n")
            for pair in method['pairs']:
                info(pair.printAll())

            for pair in method['pairs']:
                loss = map(lambda measure: float(measure.sent - measure.received) / float(measure.sent) * 100, pair.measures)
                diff = map(lambda measure: float(measure.sent - measure.received) / float(measure.sent) * 100 - 2 * measure.step, pair.measures)
                t = map(lambda measure: measure.timestamp, pair.measures)
                losses[pair.getPair()] = np.array(loss)
                diffs[pair.getPair()] = np.array(diff)
                ts[pair.getPair()] = np.array(t)


            # plot the data
            plt.subplot(nmethods, ncols, line)
            for pair in method['pairs']:
                plt.scatter(ts[pair.getPair()], losses[pair.getPair()], color = getColor(pair.getPair()), label = "%s,%s" % pair.getPair())
                plt.hold(True)
            plt.step(step_time, step_values, 'r', where = 'post')
            plt.xlabel('Time (s)', fontsize = 10)
            plt.ylabel('Loss %s' % name, fontsize = 10)
            plt.legend(loc = 2)
            plt.subplot(nmethods, ncols, line + 1)
            plt.hist(diffs.values(), nbins, normed = 1,
                     label = ["%s,%s" % x for x in diffs.keys()])
            plt.legend(loc = 'upper left', bbox_to_anchor = (1.0, 1.0), ncol = 1)

            # plt.hist(diffs.values(), stacked = True)
            # plt.xticks(bins, ["2^%s" % i for i in bins])
            # plt.hold(True)
            # plt.plot(steps_time, steps_val, 'r,-')
            #         plt.axis([0, 60, 0, 2000])
            plt.draw()
            line += ncols
        fig = plt.gcf()
        fig.set_size_inches(10, 14)
        plt.savefig('checks/loss.pdf', format = 'pdf', bbox_inches = 'tight')
        plt.show()


    class HostStats(object):
        def __init__(self, host, target, method = ''):
            self.host = host
            self.target = target
            self.measures = []
            self.baseline = None
            self.method = method

        def getPair(self):
            return self.host.name, self.target.name

        def printAll(self):
            return "\n%s -> %s\nbaseline for %s: \n   %s\nmeasures : \n   %s\n" % (self.host.name,
                                                                                   self.target.name,
                                                                                   self.method,
                                                                                   "None",
                                                                                   "\n   ".join([m.printAll() for m in self.measures]))


class Plotter(object):
    pass


if __name__ == "__main__":
    import json

    class HStats(object):
        def __init__(self, method, host, target, baseline, measures):
            self.host = host
            self.target = target
            self.baseline = baseline
            self.measures = measures
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
            return self.host, self.target

        def printAll(self):
            return "\n%s -> %s\nbaseline for %s: \n   %s\nmeasures : \n   %s\n" % (self.host,
                                                                                   self.target,
                                                                                   self.method,
                                                                                   self.baseline.printAll(),
                                                                                   "\n   ".join([m.printAll() for m in self.measures]))


    ms = json.load(open('checks/s/results.json', 'r'))
    methods = {}
    for name, method_data in ms.iteritems():
        methods[name] = {}
        methods[name]['pairs'] = []
        methods[name]['real_steps'] = method_data['real_steps']
        for pairdata in method_data['pairs']:
            s = HStats(name, pairdata['host'], pairdata['target'], DelayStats(**pairdata['baseline']),
                       [DelayStats(**m) for m in pairdata['measures']])
            methods[name]['pairs'].append(s)
    print("Loading done, making results")
    Delay.makeResults(methods)
