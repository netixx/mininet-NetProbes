from threading import Timer
from mininet.log import info
from string import Template

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
            info('* Events starting in %s seconds\n' % cls.start_time)
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
        #     global timers
        if event.repeat is not None:
            event.timerRun = Timer(0.0, runPeriodicEvent, args = [event, cls.d_network])
            cls.events.append(event)
            info('* Event %s : Scheduled periodic event on equipment %s:\n > duration %s\n > period %s \n > modifying parameters : %s\n-------\n'
                 % (event.id, event.target, event.duration, event.repeat, ", ".join(event.variations.keys())))
        else:
            event.timerRun = Timer(0.0, runEvent, args = [event, cls.d_network])
            cls.events.append(event)
            info('* Event %s : Scheduled event on equipment %s\n > duration %s\n > modifying parameters : %s\n-------\n'
                 % (event.id, event.target, event.duration, ", ".join(event.variations.keys())))

    @classmethod
    def d_network(cls):
        return cls.network

def runEvent(event, net):
    if callable(net) : net = net()
    info('* Event %s : Running event on %s\n' % (event.id, event.target))
    targ = net.get(event.target)
    # supports links only
    targ.set(event.variations)
    if event.duration is not None:
        event.timerReset = Timer(event.duration, stopEvent, args = [targ, event])
        event.timerReset.start()

def stopEvent(target, event):
    info('* Event %s : Stopping event on %s\n' % (event.id, event.target))
    target.reset()

def resetTarget(target):
    target.reset()

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

