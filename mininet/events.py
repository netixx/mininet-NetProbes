from threading import Timer

start_time = None
t_start = None
timers = []
network = None
events = []

from mininet.log import info


def startClock(net):
    info('* Events starting in %s seconds\n' % start_time)
    global network
    network = net
    global t_start
    t_start = Timer(start_time, _startTimers)
    t_start.start()


def _startTimers():
    for event in events:
        event.timerRun.start()


def stopClock():
    for event in events:
        try:
            event.timerRun.cancel()
            event.timerReset.cancel()
        except:
            pass


def sheduleEvent(event):
    event.id = len(events)
    #     global timers
    if event.repeat is not None:
        event.timerRun = Timer(start_time, runPeriodicEvent, args = [event])
        events.append(event)
        info('* Event %s : Scheduled periodic event on equipment %s:\n > duration %s\n > period %s \n > modifying parameters : %s\n-------\n'
             % (event.id, event.target, event.duration, event.repeat, ", ".join(event.variations.keys())))
    else:
        event.timerRun = Timer(start_time, runEvent, args = [event])
        events.append(event)
        info('* Event %s : Scheduled event on equipment %s\n > duration %s\n > modifying parameters : %s\n-------\n'
             % (event.id, event.target, event.duration, ", ".join(event.variations.keys())))


def runEvent(event, net = network):
    info('* Event %s : Running event on %s\n' % (event.id, event.target))
    targ = net.get(event.target)  # network.get(event.target)
    # supports links only
    targ.set(event.variations)
    if event.duration is not None:
        event.timerReset = Timer(event.duration, stopEvent, args = [targ, event])
        event.timerReset.start()


def stopEvent(target, event):
    info('* Event %s : Stopping event on %s\n' % (event.id, event.target))
    target.reset()


def runPeriodicEvent(event):
    event.timerRun = Timer(event.repeat, runPeriodicEvent, args = [event])
    event.timerRun.start()
    runEvent(event)
    if event.nrepeat is not None:
        event.nrepeat -= 1


def replaceParams(event, params):
    # TODO : flatten dict
    for key, val in event.viewitems():
        if type(val) is dict:
            replaceParams(val, params)
            continue
        if params.has_key(val):
            event[key] = params[val]


class NetEvent(object):
    def __init__(self, target = None, repeat = None, duration = None, variations = {}, nrepeat = None):
        self.target = target
        self.repeat = repeat
        self.variations = variations
        self.duration = duration
        self.nrepeat = nrepeat
        self.timerReset = None
        self.timerRun = None
        self.id = None

