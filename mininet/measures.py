__author__ = 'francois'
import re
import math
import time
import signal
import shlex
import os

binDir = "."


def setBinDir(dir):
    global binDir
    binDir = dir


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

    def printPackets(self):
        return "t: {:5.1f}, step: {:>3}, sent: {:>2}, received: {:>2}".format(self.timestamp,
                                                                              self.step,
                                                                              self.sent,
                                                                              self.received)

    def toDict(self):
        return {'timestamp': self.timestamp,
                'step': self.step,
                'sent': self.sent,
                'received': self.received,
                'rttmin': self.rttmin,
                'rttavg': self.rttavg,
                'rttmax': self.rttmax,
                'rttdev': self.rttdev}


class PerfStats(DelayStats):
    def __init__(self, timestamp = -1.0, step = -1.0, sent = -1.0, received = -1.0, transfer = -1.0, bw = -1.0, jitter = -1.0, errors = -1.0,
                 loss = -1.0, outoforder = -1.0):
        self.timestamp = timestamp
        self.step = step
        self.sent = sent
        self.received = received
        self.transfer = transfer
        self.bw = bw
        self.jitter = jitter
        self.errors = errors
        self.loss = loss
        self.outoforder = outoforder


    def printAll(self):
        return "t: {:5.1f}, step: {:>3}, sent: {:10.0f}, received: {:10.0f}, " \
               "transfer: {:10.0f}, bw: {:10.0f}, jitter: {:5.3f}, errors: {:10.0f}, loss: {:5.3f}, outoforder {:5.0f}".format(self.timestamp,
                                                                                                                               self.step,
                                                                                                                               self.sent,
                                                                                                                               self.received,
                                                                                                                               self.transfer,
                                                                                                                               self.bw,
                                                                                                                               self.jitter,
                                                                                                                               self.errors,
                                                                                                                               self.loss,
                                                                                                                               self.outoforder
        )


# class LossStats(object):
#     def __init__(self

class Traceroute(object):
    """Adapter for the traceroute command"""
    P_ICMP = '-I'
    P_UDP = ''
    P_UDPLITE = '-UL'
    P_TCP = '-T'

    DEFAULT_OPTIONS = {'proto': P_UDP,
                       'wait': '5',
                       'npackets': '3',
                       'nqueries': '16',
                       'sendwait': '0'}

    _TRACEROUTE_PACKET_LIMIT = 10
    _TRACEROUTE = "traceroute {proto} -n -q {npackets} -N {nqueries} -w {wait} -z {sendwait} {target}"

    _TR_HEADER = r'traceroute to (\S+) \((\d+\.\d+\.\d+\.\d+)\)'
    _TR_HOP_FAIL = r'\s*(?P<hopnum>\d+)(?:\s+\*)+'
    _TR_HOP_OK = r'\s*(?P<hopnum>\d+)\s+(?P<address>([a-zA-Z0-9]+(?:\.|:)?)+)\s+(?P<times>.*)'

    @classmethod
    def traceroute(cls, node, target, **options):
        out, err, exitcode = node.pexec(cls._getTracerouteCmd(target.IP(), options))
        return cls._parseTrace(out.decode())

    @classmethod
    def loss(cls, node, target, **options):
        if options.has_key('npackets') and int(options['npackets']) > cls._TRACEROUTE_PACKET_LIMIT:
            res = DelayStats(sent = 0.0, received = 0.0)
            pk = int(options['npackets'])
            while pk > 0:
                options['npackets'] = pk
                out, err, exitCode = node.pexec(cls._getTracerouteCmd(target.IP(), options))
                r = cls.sumTraceDelay(cls._parseTrace(out.decode()))
                res = DelayStats(sent = r.sent + res.sent,
                                 received = r.received + res.received)
                pk -= cls._TRACEROUTE_PACKET_LIMIT
            return res
        out, err, exitCode = node.pexec(cls._getTracerouteCmd(target.IP(), options))
        return cls.sumTraceDelay(cls._parseTrace(out.decode()))


    @classmethod
    def ping(cls, node, target, **options):
        """Use traceroute as ping"""
        out, err, exitcode = node.pexec(cls._getTracerouteCmd(target.IP(), options))
        hops = cls._parseTrace(out.decode())
        return cls.sumTraceDelay(hops)

    @staticmethod
    def sumTraceDelay(hops):
        if len(hops) == 0:
            return DelayStats()
        sent = hops[0].sent
        received = hops[0].sent
        rttmin = 0
        rttmax = 0
        rttdevs = []
        rttavgs = []
        for hop in hops:
            #if the chain is broken by a node not answering
            if hop.sent <= 0 or hop.received <= 0:
                return DelayStats(sent = sent, received = 0)
            sent = max(hop.sent, sent)
            received = min(hop.received, received)
            rttavgs.append(hop.rttavg)
            rttdevs.append(hop.rttdev)
            rttmin += hop.rttmin
            rttmax += hop.rttmax

        rttavg = sum(rttavgs) / len(rttavgs)
        rttdev = math.sqrt(sum(rttdevs) / len(rttdevs))
        return DelayStats(sent = sent,
                          received = received,
                          rttmin = rttmin,
                          rttmax = rttmax,
                          rttdev = rttdev,
                          rttavg = rttavg)

    @classmethod
    def _getTracerouteCmd(cls, target, options = {}):
        opts = {"target": target}
        opts.update(cls.DEFAULT_OPTIONS)
        opts.update(options)
        if int(opts['npackets']) > cls._TRACEROUTE_PACKET_LIMIT:
            opts['npackets'] = cls._TRACEROUTE_PACKET_LIMIT
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
        def __init__(self, hopnum, address = "", times = []):
            self.address = address
            self.hopnum = hopnum
            sent = len(times) if len(times) > 0 else -1.0
            rtimes = filter(None, times)
            received = len(rtimes) if len(rtimes) > 0 else -1.0
            if received > 0:
                rttavg = sum(rtimes) / received
                rttdev = math.sqrt(sum(map(lambda x: (x - rttavg) ** 2, rtimes)) / received)
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
    _PING = "ping -c {npackets} -W {wait} -w {deadline} -i {sendspeed} {target}"
    DEFAULT_OPTIONS = {'npackets': '2',
                       'wait': '1000',
                       'deadline': '1000',
                       'sendspeed': '1'}

    @classmethod
    def ping(cls, node, target, **options):
        out, err, exitcode = node.pexec(cls._getPingCmd(target.IP(), options))
        # if exitcode != 0:
        # if isSweep:
        #     return cls._parseSweepPing(out.decode())
        return cls._parsePing(out.decode())

    @classmethod
    def loss(cls, node, target, **options):
        out, err, exitcode = node.pexec(cls._getPingCmd(target.IP(), options))
        return cls._parsePing(out.decode())


    @classmethod
    def _getPingCmd(cls, target, options = {}):
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

        # @classmethod
        # def _parseSweepPing(cls, pingOutput):
        #     errorTuple = (-1.0, -1.0, [])
        #     # Parse ping output and return all data.
        #     #         errorTuple = (1, 0, 0, 0, 0, 0)
        #     # Check for downed link
        #     r = r'[uU]nreachable'
        #     m = re.search(r, pingOutput)
        #     if m is not None:
        #         return errorTuple
        #     r = r'(\d+) packets transmitted, (\d+) (?:packets)? received'
        #     m = re.search(r, pingOutput)
        #     if m is None:
        #         return errorTuple
        #     sent, received = int(m.group(1)), int(m.group(2))
        #     r = r'(\d+) bytes from .*: icmp_seq=(\d+)'
        #     m = re.findall(r, pingOutput)
        #     sweep = []
        #     if m is not None:
        #         for line in m:
        #             sweep.append((int(line[1]), int(line[0])))
        #     return sent, received, sweep


class IPerf(object):
    P_TCP = ''
    P_UDP = '-u'
    UDP_BW = '10M'

    _CLIENT_MODE = '-c'
    _SERVER_MODE = '-s'
    MAX_ATTEMPTS = 5
    DEFAULT_OPTIONS = {'proto': P_TCP,
                       'udpBw': UDP_BW}

    _CLI_TPL = r'(?P<timestamp>[^,]+),,,,,(?P<id>[^,]+),(?P<interval>[^,]+),(?P<transfer>\d+),(?P<bw>\d+)'
    _SRV_TPL = r'(?P<timestamp>[^,]+),,,,,(?P<id>[^,]+),(?P<interval>[^,]+),(?P<transfer>\d+),(?P<bw>\d+)(?P<udp>.*)?'
    _UDP_SRV_TPL = r',(?P<jitter>\d+\.\d+),(?P<errors>\d+),(?P<datagrams>\d+),(?P<loss>\d+\.\d+),(?P<outoforder>\d+)?'


    @classmethod
    def loss(cls, nodes, npackets = 100, **options):
        options['proto'] = cls.P_UDP
        options['udpBw'] = '%sK' % npackets
        return cls.iperf(nodes, **options)


    @classmethod
    def bw(cls, nodes, **options):
        """Run iperf between two hosts."""
        assert len(nodes) == 2
        client, server = nodes
        opts = {"ip": server.IP()}
        opts.update(cls.DEFAULT_OPTIONS)
        opts.update(options)
        server.pexec('killall -9 iperf')
        iperf = 'iperf -y c --reportexclude CMSV {proto} {mode}'
        iperfCli = '%s {ip} {udpBw}' % iperf
        iperfSrv = '%s' % iperf

        if opts['proto'] == cls.P_UDP:
            opts['udpBw'] = '-b %s' % opts['udpBw']
        else:
            opts['udpBw'] = ''
        srv = server.popen(shlex.split(iperfSrv.format(mode = cls._SERVER_MODE, **opts)))
        # servout = ''
        # while server.lastPid is None:
        #     servout += server.monitor()
        if opts['proto'] == cls.P_TCP:
            attempts = 0
            telnetCli = 'sh -c "echo A | telnet -e A {ip} 5001"'
            while ('Connected' not in client.pexec(shlex.split(telnetCli.format(**opts)))[0]
                   and attempts < cls.MAX_ATTEMPTS):
                attempts += 1
                time.sleep(.5)
        cliout, clierr, exitcode = client.pexec(shlex.split(iperfCli.format(mode = cls._CLIENT_MODE, **opts)))
        srv.send_signal(signal.SIGINT)
        srvout, srverr = srv.communicate()
        srv.wait()
        return cls._parseIperf(srvout.decode(), cliout.decode())
        # servout += server.waitOutput()
        # result = [ self._parseIperf( servout ), self._parseIperf( cliout ) ]
        # if protocol == cls.P_UDP:
        #     result.insert( 0, udpBw )
        # output( '*** Results: %s\n' % result )
        # return result

    @classmethod
    def _parseIperf(cls, srvOut, cliOut):
        #TODO: check for more lines
        #TODO: cross check with client output
        for line in srvOut.splitlines():
            m = re.match(cls._SRV_TPL, line)
            if m is not None:
                bw = float(m.group('bw'))
                transfer = float(m.group('transfer'))
                if m.groupdict().has_key('udp'):
                    u = re.match(cls._UDP_SRV_TPL, m.group('udp'))
                    if u is not None:
                        sent = float(u.group('datagrams'))
                        errors = float(u.group('errors'))
                        received = sent - errors
                        return PerfStats(sent = sent, received = received,
                                         loss = float(u.group('loss')),
                                         outoforder = float(u.group('outoforder')),
                                         bw = bw,
                                         jitter = float(u.group('jitter')),
                                         transfer = transfer,
                                         errors = errors)

                return PerfStats(bw = bw, transfer = transfer)


class PathChirp(object):

    JUMBO_OPT = '-J'

    _SND_EXEC = os.path.join(binDir, 'pathchirp_snd')
    _RCV_EXEC = os.path.join(binDir, 'pathchirp_rcv')
    _RUN_EXEC = os.path.join(binDir, 'pathchirp_run')

    _SND_CMD = '%s {senderPort}' % _SND_EXEC
    _RCV_CMD = '%s' % _RCV_EXEC
    _RUN_CMD = '%s -S {sender} -R {receiver} -t {duration} {jumbo} -l {lowRate} -u {highRate} -d {duration} -p {packetSize} -s {spreadFactor} -d {decreaseFactor} -b {busyPeriod} -n {nEstimates} -a {averageProbingRate}' % _RUN_EXEC

    DEFAULT_OPTIONS = {'senderPort': 8365,
                       'jumbo' : 1,
                       'lowRate' : 10,
                       'highRate' : 200,
                       'duration' : 600,
                       'packetSize' : 1000,
                       'spreadFactor' : 1.2,
                       'decreaseFactor' : 1.5,
                       'busyPeriod' : 5,
                       'nEstimates' : 11,
                       'averageProbingRate' : 0.3}

    @classmethod
    def bw(cls, nodes, master = None, **options):
        sender, receiver = nodes
        if master is None:
            master = sender

        opts = {'sender': sender.IP(),
                'receiver' : receiver.IP()}
        opts.update(cls.DEFAULT_OPTIONS)
        opts.update(options)

        rcv = receiver.popen(shlex.split(cls._RCV_CMD.format(opts)))
        snd = sender.popen(shlex.split(cls._SND_CMD.format(opts)))
        #make test happen and wait for completion
        out, err, exitcode = master.pexec(shlex.split(cls._RUN_CMD.format(opts)))

        #shutdown receiver and sender
        snd.send_signal(signal.SIGINT)
        rcv.send_signal(signal.SIGINT)

        return cls._parsePathChirp(out)

    @classmethod
    def _parsePathChirp(cls, out):
        print out


class PathLoad(object):
    pass


class IGI(object):
    pass


class Spruce(object):
    pass
