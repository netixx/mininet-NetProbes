__author__ = 'francois'
import re
import math


class DelayStats(object):
    """Generic class for measuring delay"""
    def __init__(self, timestamp = -1.0, step = -1.0, sent = -1.0, received = -1.0, rttmin = -1.0, rttavg = -1.0, rttmax = -1.0, rttdev = -1.0):
        self.timestamp = timestamp
        self.step = step
        self.sent = sent
        self.received = received
        self.rttmin = rttmin
        self.rttavg = rttavg
        self.rttmax = rttmax
        self.rttdev = rttdev

    def printAll(self):
        return "t: {:5.1f}, step: {:>3}, sent: {:>2}, received: {:>2}, " \
               "rttmin: {:6.2f}, rttavg: {:6.2f}, rttmax: {:6.2f}, rttdev: {:5.3f}".format(self.timestamp,
                                                                                           self.step,
                                                                                           self.sent,
                                                                                           self.received,
                                                                                           self.rttmin,
                                                                                           self.rttavg,
                                                                                           self.rttmax,
                                                                                           self.rttdev
        )


class Traceroute(object):
    """Adapter for the traceroute command"""
    P_ICMP = '-I'
    P_UDP = ''
    P_UDPLITE = '-UL'
    P_TCP = '-T'

    DEFAULT_OPTIONS = {'proto' : P_UDP,
                       'wait' : '1',
                       'npackets' : '3',
                       'nqueries' : '16'}


    _TRACEROUTE = "traceroute {proto} -n -q {npackets} -N {nqueries} -w {wait} {target}"

    _TR_HEADER = r'traceroute to (\S+) \((\d+\.\d+\.\d+\.\d+)\)'
    _TR_HOP_FAIL = r'\s*(?P<hopnum>\d+)(?:\s+\*)+'
    _TR_HOP_OK = r'\s*(?P<hopnum>\d+)\s+(?P<address>([a-zA-Z0-9]+(?:\.|:)?)+)\s+(?P<times>.*)'

    @classmethod
    def traceroute(cls, node, target, **options):
        out, err, exitcode = node.pexec(cls._getTracerouteCmd(target.IP(), options))
        return cls._parseTrace(out.decode())

    @classmethod
    def ping(cls, node, target, **options):
        """Use traceroute as ping"""
        out, err, exitcode = node.pexec(cls._getTracerouteCmd(target.IP(), options))
        hops = cls._parseTrace(out.decode())
        return hops[-1]

    @classmethod
    def _getTracerouteCmd(cls, target, options = {}):
        opts = {"target": target}
        opts.update(cls.DEFAULT_OPTIONS)
        opts.update(options)
        return cls._TRACEROUTE.format(**opts)

    @classmethod
    def _parseTrace(cls, output):
        hops = []
        header = re.search(cls._TR_HEADER, output)
        if header is None:
            return [cls.HopResult(0)]
        for line in output.splitlines():
            hop = re.match(cls._TR_HOP_FAIL, line)
            if hop is not None:
                hops.append(cls.HopResult(int(hop.group('hopnum'))))
            hop = re.match(cls._TR_HOP_OK, line)
            if hop is not None:
                hops.append(cls.HopResult(int(hop.group('hopnum')), hop.group('address'), cls._parseTimes(hop.group('times'))))
        return hops

    @classmethod
    def _parseTimes(cls, times):
        parts = times.split()
        times = []
        for part in parts:
            if part == '*':
                times.append(None)
            num = re.match(r'\d+\.\d+', part)
            if num is not None:
                times.append(float(part))

        return times

    class HopResult(DelayStats):
        def __init__(self, hopnum, address = "",  times = []):
            self.address = address
            self.hopnum = hopnum
            sent = len(times) if len(times) > 0 else -1.0
            rtimes = filter(None, times)
            received = len(rtimes) if len(rtimes) > 0 else -1.0
            if received > 0:
                rttavg = sum(rtimes) / received
                rttdev = math.sqrt(sum(map(lambda x: (x-rttavg)**2, rtimes))/received)
                rttmin = min(rtimes)
                rttmax = max(rtimes)
            else:
                rttavg = -1.0
                rttdev = -1.0
                rttmin = -1.0
                rttmax = -1.0
            DelayStats.__init__(self,
                                sent = sent,
                                received = received,
                                rttmin = rttmin,
                                rttavg = rttavg,
                                rttmax = rttmax,
                                rttdev = rttdev)


class Ping(object):
    """Iterface for the ping command"""
    _PING = "ping -c {npackets} -W {wait} -w {deadline} {target}"
    DEFAULT_OPTIONS = {'npackets' : '2',
                       'wait' : '1000',
                       'deadline' : '1000'}

    @classmethod
    def ping(cls, node, target, **options):
        out, err, exitcode = node.pexec(cls._getPingCmd(target.IP(), options))
        # if exitcode != 0:
        # if isSweep:
        #     return cls._parseSweepPing(out.decode())
        return cls._parsePing(out.decode())


    @classmethod
    def _getPingCmd(cls, target, options = {} ):
        opts = {"target": target}
        opts.update(cls.DEFAULT_OPTIONS)
        opts.update(options)
        return cls._PING.format(**opts)


    @classmethod
    def _parsePing(cls, pingOutput):
        # Parse ping output and return all data.
        # Check for downed link
        r = r'[uU]nreachable'
        m = re.search(r, pingOutput)
        if m is not None:
            return DelayStats(sent = 0, received = 0)
        r = r'(\d+) (?:packets? )?transmitted, (\d+) (?:packets? )?received'
        m = re.search(r, pingOutput)
        if m is None:
            return DelayStats()
        sent, received = int(m.group(1)), int(m.group(2))
        r = r'(?:(?:rtt)|(?:round-trip)) min/avg/max/(?:m|(?:std))dev = '
        r += r'(\d+\.\d+)/(\d+\.\d+)/(\d+\.\d+)/(\d+\.\d+) ms'
        m = re.search(r, pingOutput)
        if m is None:
            return DelayStats(sent = sent, received = received)
        try:
            rttmin = float(m.group(1))
            rttavg = float(m.group(2))
            rttmax = float(m.group(3))
            rttdev = float(m.group(4))
        except:
            return DelayStats(sent = sent, received = received)
        return DelayStats(sent = sent, received = received, rttmin = rttmin, rttavg = rttavg, rttmax = rttmax, rttdev = rttdev)
