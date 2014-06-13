from threading import Timer
from string import Template

from mininet.log import output, error


class EventsManager(object):
    start_time = None
    t_start = None
    timers = []
    network = None
    events = []

    @classmethod
    def setNet(cls, net):
        cls.network = net

    @classmethod
    def startClock(cls, net):
        cls.setNet(net)
        if cls.start_time is not None:
            output('* Events starting in %s seconds\n' % cls.start_time)
            cls.startTimers(delay = cls.start_time)

    @classmethod
    def startTimers(cls, delay = None):
        if delay:
            cls.t_start = Timer(delay, cls.startTimers)
            cls.t_start.start()
            return
        for event in cls.events:
            event.timerRun.start()

    @classmethod
    def stopClock(cls):
        for event in cls.events:
            try:
                event.timerRun.cancel()
                event.timerReset.cancel()
            except:
                pass

    @classmethod
    def sheduleEvent(cls, event):
        event.id = len(cls.events)
        # global timers
        if event.repeat is not None:
            event.timerRun = Timer(0.0, runPeriodicEvent, args = [event, cls.d_network])
            cls.events.append(event)
            output('* Event %s : Scheduled periodic event on equipment %s:\n > duration %s\n > period %s \n > modifying parameters : %s\n-------\n'
                   % (event.id, event.target, event.duration, event.repeat, ", ".join("%s:%s" % (k, v) for k, v in event.variations.iteritems())))
        else:
            event.timerRun = Timer(0.0, runEvent, args = [event, cls.d_network])
            cls.events.append(event)
            output('* Event %s : Scheduled event on equipment %s\n > duration %s\n > modifying parameters : %s\n-------\n'
                   % (event.id, event.target, event.duration, ", ".join(event.variations.keys())))

    @classmethod
    def d_network(cls):
        return cls.network


def runEvent(event, net):
    if callable(net):
        net = net()
    output('* Event %s : Running event on %s\n' % (event.id, event.target))
    targ = net.get(event.target)
    try:
        # supports links only
        targ.set(event.variations)
    except Exception as e:
        error("Error while changing event parameters for event %s : %s\n" % (event.id, e))
        targ.set(event.variations)
    if event.duration is not None:
        event.timerReset = Timer(event.duration, stopEvent, args = [targ, event])
        event.timerReset.start()


def stopEvent(target, event):
    output('* Event %s : Stopping event on %s\n' % (event.id, event.target))
    resetTarget(target)


def resetTarget(target):
    try:
        target.reset()
    except Exception as e:
        error("Error while resetting event on %s : %s" % (target.name, e))


def runPeriodicEvent(event, net):
    event.timerRun = Timer(event.repeat, runPeriodicEvent, args = [event, net])
    event.timerRun.start()
    runEvent(event, net)
    if event.nrepeat is not None:
        event.nrepeat -= 1


def replaceParams(event, params):
    # TODO : flatten dict
    for key, val in event.viewitems():
        if type(val) is dict:
            replaceParams(val, params)
            continue
        event[key] = Template(event[key]).substitute(**params)


class NetEvent(object):
    def __init__(self, target = None, repeat = None, duration = None, variations = None, nrepeat = None):
        self.target = target
        self.repeat = repeat
        self.variations = variations if variations is not None else {}
        self.duration = duration
        self.nrepeat = nrepeat
        self.timerReset = None
        self.timerRun = None
        self.id = None

