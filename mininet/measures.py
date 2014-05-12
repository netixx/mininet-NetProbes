__author__ = 'francois'
import re
import math
import time
import signal
import shlex
import os

_float_format = r'\d+\.\d+'
_int_format = r'\d+'
_ip_format = r'([a-zA-Z0-9]+(?:\.|:)?)+'


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

    def printAvg(self):
        return "{:6.2f}".format(self.rttavg)

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


class BwStats(object):
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
        return "t: {:5.1f}, step: {:>3}, bw: {:10.0f}".format(self.timestamp,
                                                              self.step,
                                                              self.bw)

    def printAllExtended(self):
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

    def toDict(self):
        return {'timestamp': self.timestamp,
                'step': self.step,
                'sent': self.sent,
                'received': self.received,
                'bw': self.bw}


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
                       'sendwait': '0',
                       'firstTTL': 1,
                       'maxTTL': 30}

    _TRACEROUTE_PACKET_LIMIT = 1000
    _TRACEROUTE = "{exec} {proto} -n -q {npackets} -N {nqueries} -w {wait} -z {sendwait} -f {firstTTL} -m {maxTTL} {target}"

    _TR_HEADER = r'traceroute to (\S+) \((\d+\.\d+\.\d+\.\d+)\)'
    _TR_HOP_FAIL = r'\s*(?P<hopnum>\d+)(?:\s+\*)+'
    _TR_HOP_OK = r'\s*(?P<hopnum>\d+)\s+(?P<address>([a-zA-Z0-9]+(?:\.|:)?)+)\s+(?P<times>.*)'

    @classmethod
    def traceroute(cls, nodes, binDir = None, **options):
        """Perform a traceroute to target from node"""
        node, target = nodes
        out, err, exitcode = node.pexec(cls._getTracerouteCmd(binDir = binDir, target = target.IP(), **options))
        return cls._parseTrace(out.decode())

    @classmethod
    def loss(cls, nodes, binDir = None, **options):
        node, target = nodes
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
        out, err, exitCode = node.pexec(cls._getTracerouteCmd(binDir = binDir, target = target.IP(), **options))
        return cls.sumTraceDelay(cls._parseTrace(out.decode()))


    @classmethod
    def ping(cls, nodes, binDir = None, **options):
        """Use traceroute as ping from node to target"""
        node, target = nodes
        out, err, exitcode = node.pexec(cls._getTracerouteCmd(binDir = binDir, target = target.IP(), firstTTL = 60, maxTTL = 60, **options))
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
    def _getTracerouteCmd(cls, binDir = None, **options):
        if binDir is None:
            opts = {'exec': 'traceroute'}
        else:
            opts = {'exec': os.path.join(binDir, 'traceroute')}
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
    _PING = "{exec} -c {npackets} -W {wait} -w {deadline} -i {sendspeed} {target}"
    DEFAULT_OPTIONS = {'npackets': '2',
                       'wait': '1000',
                       'deadline': '1000',
                       'sendspeed': '1'}

    @classmethod
    def ping(cls, nodes, binDir = None, **options):
        """Perform ping from node to target"""
        node, target = nodes
        out, err, exitcode = node.pexec(cls._getPingCmd(target = target.IP(), **options))
        # if exitcode != 0:
        # if isSweep:
        #     return cls._parseSweepPing(out.decode())
        return cls._parsePing(out.decode())

    @classmethod
    def loss(cls, nodes, binDir = None, **options):
        node, target = nodes
        out, err, exitcode = node.pexec(cls._getPingCmd(target = target.IP(), **options))
        return cls._parsePing(out.decode())


    @classmethod
    def _getPingCmd(cls, binDir = None, **options):
        if binDir is None:
            opts = {'exec': 'ping'}
        else:
            opts = {'exec': os.path.join(binDir, 'ping')}
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
    """Interface for the IPerf program"""

    _EXEC = 'iperf'

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
    def bw(cls, nodes, binDir = None, **options):
        """Run iperf between two hosts.
        param nodes: tuple of nodes to measure the bandwidth between
        param binDir: optional bin directory to launch executables from
        param options: options for this run (merged with DEFAULT_OPTIONS)
        """
        assert len(nodes) == 2
        client, server = nodes
        opts = {"ip": server.IP()}
        if binDir is not None:
            opts['exec'] = os.path.join(binDir, cls._EXEC)
        else:
            opts['exec'] = cls._EXEC

        opts.update(cls.DEFAULT_OPTIONS)
        opts.update(options)
        server.pexec('killall -9 iperf')
        iperf = '{exec} -y c --reportexclude CMSV {proto} {mode}'
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
        r = BwStats()
        #TODO: check for more lines
        #TODO: cross check with client output
        for line in srvOut.splitlines():
            m = re.match(cls._SRV_TPL, line)
            if m is not None:
                bw = float(m.group('bw'))
                transfer = float(m.group('transfer'))
                if m.groupdict().has_key('udp') and m.group('udp') != '':
                    u = re.match(cls._UDP_SRV_TPL, m.group('udp'))
                    if u is not None:
                        sent = float(u.group('datagrams'))
                        errors = float(u.group('errors'))
                        received = sent - errors
                        r = BwStats(sent = sent, received = received,
                                    loss = float(u.group('loss')),
                                    outoforder = float(u.group('outoforder')),
                                    bw = bw,
                                    jitter = float(u.group('jitter')),
                                    transfer = transfer,
                                    errors = errors)
                    else:
                        r = BwStats(bw = bw, transfer = transfer)
                else:
                    r = BwStats(bw = bw, transfer = transfer)
        return r


class PathLoad(object):
    """Interface for the pathload program
    !!! Does NOT work as of yet for unknown reasons"""
    _SND_EXEC = 'pathload_snd'
    _RCV_EXEC = 'pathload_rcv'

    _SND_CMD = '{sndExec}'
    _RCV_CMD = '{rcvExec} -t {timeout} -s {senderIp} {bw-resol}'

    DEFAULT_OPTIONS = {
        'timeout': 1000,
        'resolution': None
    }

    @classmethod
    def bw(cls, nodes, binDir = None, **options):
        sender, receiver = nodes
        opts = {'senderIp': sender.IP(),
                'outfile': '',
                'logfile': ''
        }

        if binDir is not None:
            opts['sndExec'] = os.path.join(binDir, cls._SND_EXEC)
            opts['rcvExec'] = os.path.join(binDir, cls._RCV_EXEC)
        else:
            opts['sndExec'] = cls._SND_EXEC
            opts['rcvExec'] = cls._RCV_EXEC

        opts.update(cls.DEFAULT_OPTIONS)
        opts.update(options)
        if opts['resolution'] is None:
            opts['bw-resol'] = ''
        else:
            opts['bw-resol'] = '-w %s' % opts['resolution']
        #start sender first for pathload
        snd = sender.popen(shlex.split(cls._SND_CMD.format(**opts)))
        # nbsr = NonBlockingStreamReader(snd.stdout)
        # output = nbsr.readline(0.1)
        time.sleep(1)
        out, err, exitcode = receiver.pexec(shlex.split(cls._RCV_CMD.format(**opts)))
        snd.send_signal(signal.SIGINT)
        return cls._parsePathLoad(out, err)

    @classmethod
    def _parsePathLoad(cls, out, err):
        print "out :", out
        print "err :", err
        return BwStats()


class IGI(object):
    """Interface to the IGI program"""
    _CLI_EXEC = 'ptr-client'
    _SRV_EXEC = 'ptr-server'

    _CLI_CMD = '{cliExec} -n {nprobes} -s {packetSize} -k {ntrains} {serverIp}'
    _SRV_CMD = '{srvExec}'

    _PTR_TPL = r'PTR:\s+(?P<bw>\d+\.\d+)\s+Mpbs(?: (suggested))?'
    _IGI_TPL = r'IGI:\s+(?P<bw>\d+\.\d+)\s+Mpbs'

    DEFAULT_OPTIONS = {
        'nprobes': 60,
        'packetSize': '500B',
        'ntrains': 3
    }

    @classmethod
    def bw(cls, nodes, binDir = None, **options):
        """Measure bandwidth with igi
        param nodes: tuple of nodes to measure the bandwidth between
        param binDir: optional bin directory to launch executables from
        param options: options for this run (merged with DEFAULT_OPTIONS)
        """
        client, server = nodes
        opts = {'serverIp': server.IP(),
                'npairs': 100
        }

        if binDir is not None:
            opts['cliExec'] = os.path.join(binDir, cls._CLI_EXEC)
            opts['srvExec'] = os.path.join(binDir, cls._SRV_EXEC)
        else:
            opts['cliExec'] = cls._CLI_EXEC
            opts['srvExec'] = cls._SRV_EXEC

        opts.update(cls.DEFAULT_OPTIONS)
        opts.update(options)
        #start server first for IGI
        srv = server.popen(shlex.split(cls._SRV_CMD.format(**opts)))
        # nbsr = NonBlockingStreamReader(rcv.stdout)
        # output = nbsr.readline(0.1)

        out, err, exitcode = client.pexec(shlex.split(cls._CLI_CMD.format(**opts)))
        srv.send_signal(signal.SIGINT)
        return cls._parseIgi(out)


    @classmethod
    def _parseIgi(cls, out):
        r = BwStats()
        ptr = re.search(cls._PTR_TPL, out)
        if ptr is not None:
            r = BwStats(bw = float(ptr.group('bw')) * (1000 ** 2))
        igi = re.search(cls._IGI_TPL, out)
        if igi is not None:
            r = BwStats(bw = float(igi.group('bw')) * (1000 ** 2))
        return r


class Spruce(object):
    """Interface for the spruce program"""
    _SND_EXEC = 'spruce_snd'
    _RCV_EXEC = 'spruce_rcv'

    _SND_CMD = '{sndExec} -h {receiverIp} -n {npairs} -c {bw}'
    _RCV_CMD = '{rcvExec}'

    _BW_TPL = r'A|available B|bandwidth(?: estimate)?: (?P<bw>\d+)\sKbps'
    DEFAULT_OPTIONS = {
        'npairs': 100,
        'bw': '100M'
    }

    @classmethod
    def bw(cls, nodes, binDir = None, **options):
        """Measure bandwidth with spruce
        param nodes: tuple of nodes to measure the bandwidth between
        param binDir: optional bin directory to launch executables from
        param options: options for this run (merged with DEFAULT_OPTIONS)
        """
        sender, receiver = nodes
        opts = {'receiverIp': receiver.IP(),
                'npairs': 100
        }

        if binDir is not None:
            opts['sndExec'] = os.path.join(binDir, cls._SND_EXEC)
            opts['rcvExec'] = os.path.join(binDir, cls._RCV_EXEC)
        else:
            opts['sndExec'] = cls._SND_EXEC
            opts['rcvExec'] = cls._RCV_EXEC

        opts.update(cls.DEFAULT_OPTIONS)
        opts.update(options)
        #start receiver first for spruce
        rcv = receiver.popen(shlex.split(cls._RCV_CMD.format(**opts)))
        # nbsr = NonBlockingStreamReader(rcv.stdout)
        # output = nbsr.readline(0.1)

        out, err, exitcode = sender.pexec(shlex.split(cls._SND_CMD.format(**opts)))
        rcv.send_signal(signal.SIGINT)
        return cls._parseSpruce(err)

    @classmethod
    def _parseSpruce(cls, out):
        r = BwStats()
        m = re.search(cls._BW_TPL, out)
        if m is not None:
            #convert to bits per seconds
            r = BwStats(bw = float(m.group('bw')) * 1000)
        return r


class Assolo(object):
    """Interface for the assolo program,
    master = sender in current implementation"""
    JUMBO_OPT = '-J'

    _SND_EXEC = 'assolo_snd'
    _RCV_EXEC = 'assolo_rcv'
    _RUN_EXEC = 'assolo_run'

    _SND_CMD = '{sndExec} -U {senderPort}'
    _RCV_CMD = '{rcvExec}'
    _RUN_CMD = '{runExec} -S {sender} -R {receiver} -t {duration} -J {jumbo} -U {senderPort}  -l {lowRate} -u {highRate} -p {packetSize} -s {' \
               'spreadFactor} -d {decreaseFactor} -b {busyPeriod} -n {nEstimates} -a {averageProbingRate}'

    DEFAULT_OPTIONS = {'senderPort': 8365,
                       'jumbo': 1,
                       'lowRate': 10,
                       'highRate': 200,
                       'duration': 600,
                       'packetSize': 1000,
                       'spreadFactor': 1.2,
                       'decreaseFactor': 1.5,
                       'busyPeriod': 5,
                       'nEstimates': 11,
                       'averageProbingRate': 0.3}

    _INST_FILE_TPL = r'Opening file: (?P<filename>.*)'
    _TMSTP_LINE_TPL = r'\s*(?P<timestamp>\d+\.\d+)\s+(?P<bw>\d+\.\d+)'
    _COALESCING = r'Interrupt Coalescence detected'

    @classmethod
    def bw(cls, nodes, master = None, binDir = None, **options):
        """Measure bandwidth with assolo
        param nodes: tuple of nodes to measure the bandwidth between
        param binDir: optional bin directory to launch executables from
        param options: options for this run (merged with DEFAULT_OPTIONS)
        """
        sender, receiver = nodes
        if master is None:
            master = sender

        opts = {'sender': sender.IP(),
                'receiver': receiver.IP()}
        if binDir is not None:
            opts['sndExec'] = os.path.join(binDir, cls._SND_EXEC)
            opts['rcvExec'] = os.path.join(binDir, cls._RCV_EXEC)
            opts['runExec'] = os.path.join(binDir, cls._RUN_EXEC)
        else:
            opts['sndExec'] = cls._SND_EXEC
            opts['rcvExec'] = cls._RCV_EXEC
            opts['runExec'] = cls._RUN_EXEC

        opts.update(cls.DEFAULT_OPTIONS)
        opts.update(options)
        rcv = receiver.popen(shlex.split(cls._RCV_CMD.format(**opts)))
        snd = sender.popen(shlex.split(cls._SND_CMD.format(**opts)))
        #make test happen and wait for completion
        out, err, exitcode = master.pexec(shlex.split(cls._RUN_CMD.format(**opts)))
        #shutdown receiver and sender
        snd.send_signal(signal.SIGINT)
        rcv.send_signal(signal.SIGINT)
        return cls._parseAssolo(err)

    @classmethod
    def _parseAssolo(cls, out):
        r = BwStats()

        res = re.search(cls._INST_FILE_TPL, out.decode())
        try:
            if res is None:
                return r
            fn = res.group('filename')
            coal = re.search(cls._COALESCING, out.decode())
            #interrupt coalescing detected, throw result away!
            if coal is not None:
                print 'Interrupt coalescing detected, throwing result'
                return r

            values = []
            with open(fn, 'r') as f:
                r.bw = 0
                for line in f:
                    m = re.match(cls._TMSTP_LINE_TPL, line.rstrip('\n'))
                    if m is not None:
                        #get values for this timestamp
                        ctm = float(m.group('timestamp'))
                        cbw = float(m.group('bw'))
                        values.append((ctm, cbw))
            ptm = 0
            ttot = 0
            pbw = 0
            #running avg
            rbw = 0
            n = 0
            #file in not sorted!
            for ctm, cbw in sorted(values):
                n += 1
                #first step : average = cbw, set initial timestamp
                if n == 1:
                    r.bw = cbw * (1000 ** 2)
                    ptm = ctm
                    pbw = cbw
                    continue
                #calculate duration from last timestamp
                dt = ctm - ptm
                #update last timestamp
                ptm = ctm
                #update total run time
                ttot += dt
                #weigh measure with duration of measure, averaging value
                rbw += ((cbw + pbw) / 2 ) * dt
                pbw = cbw
            if n > 1:
                #divide by total to get average if more than one measure
                r.bw = (rbw / ttot) * (1000 ** 2)
            return r
        finally:
            os.remove(fn)


class Abing(object):
    """Interface for the abing bw measurement program"""
    _SND_EXEC = 'abing'
    _RFL_EXEC = 'abw_rfl'

    _SND_CMD = '{sndExec} -m {portNumber} -n {nSamples} {period} -d {rflIp} -p {nPackets} -a {meanAlpha}'
    _RFL_CMD = '{rflExec} -m {portNumber} -s {nSamples}'
    _RFL_CALIBRATE_CMD = '{rflExec} -c'
    _SND_CALIBRATE_CMD = '{sndExec} -c'

    _TIMESTAMP_TPL = r'\s*(?P<timestamp>\d+)\s+(?P<direction>T|F):\s+(?P<address>{ip})\s+'
    _ABW_XTR_DBC_TPL = r'ABw-Xtr-DBC:\s+(?P<Abw>{float})\s+(?P<Xtr>{float})\s+(?P<DBC>{float})\s+'
    _ABW_TPL = r'ABW:\s+(?P<ABW>{float})\s+Mbps\s+'
    _RTT_TPL = r'RTT:\s+(?P<rttavg>{float})\s+(?P<rttmin>{float})\s+(?P<rttmax>{float})\s+ms\s+'
    _PP_TPL = r'(?P<sent>{int})\s+(?P<received>{int})'

    _LINE_TPL = (_TIMESTAMP_TPL + _ABW_XTR_DBC_TPL + _ABW_TPL + _RTT_TPL + _PP_TPL).format(float = _float_format,
                                                                                           int = _int_format,
                                                                                           ip = _ip_format)

    DEFAULT_OPTIONS = {
        'portNumber': 8176,
        'nSamples': 2,
        'samplePeriod': 1,
        'period': '',
        'nPackets': 20,
        'meanAlpha': 0.75
    }

    @classmethod
    def bw(cls, nodes, binDir = None, **options):
        """Measure bandwidth with abing
        param nodes: tuple of nodes to measure the bandwidth between
        param binDir: optional bin directory to launch executables from
        param options: options for this run (merged with DEFAULT_OPTIONS)
        """
        sender, reflector = nodes
        opts = {'rflIp': reflector.IP()}

        if binDir is not None:
            opts['sndExec'] = os.path.join(binDir, cls._SND_EXEC)
            opts['rflExec'] = os.path.join(binDir, cls._RFL_EXEC)
        else:
            opts['sndExec'] = cls._SND_EXEC
            opts['rflExec'] = cls._RFL_EXEC

        opts.update(cls.DEFAULT_OPTIONS)
        opts.update(options)
        if opts['nSamples'] > 1:
            opts['period'] = '-t %s' % opts['samplePeriod']

        cls._calibrate(reflector, **opts)
        #start reflector first for abing
        rfl = reflector.popen(shlex.split(cls._RFL_CMD.format(**opts)))
        # time.sleep(2)
        # nbsr = NonBlockingStreamReader(rcv.stdout)
        # output = nbsr.readline(0.1)
        out, err, exitcode = sender.pexec(shlex.split(cls._SND_CMD.format(**opts)))
        rfl.send_signal(signal.SIGINT)
        return cls._parseAbing(out)

    @classmethod
    def _calibrate(cls, reflector, **options):
        rfl = reflector.popen(shlex.split(cls._RFL_CALIBRATE_CMD.format(**options)))
        reflector.pexec(shlex.split(cls._SND_CALIBRATE_CMD.format(**options)))
        rfl.wait()


    @classmethod
    def _parseAbing(cls, out):
        r = BwStats(sent = 0, received = 0)
        ptm = 0
        ttot = 0
        pbw = 0
        #running avg
        rbw = 0
        n = 0
        rttavg = 0
        prttavg = 0
        for line in out.splitlines():
            m = re.match(cls._LINE_TPL, line)
            if m is not None:
                if m.group('direction') == 'T':
                    r.sent += int(m.group('sent'))
                    r.received += int(m.group('received'))
                    # r.rttmin = min(r.rttmin, float(m.group('rttmin'))) if r.rttmin > 0 else float(m.group('rttmin'))
                    # r.rttmax = max(r.rttmax, float(m.group('rttmax'))) if r.rttmax > 0 else float(m.group('rttmax'))
                    cbw = float(m.group('Abw'))
                    ctm = float(m.group('timestamp'))
                    crttavg = float(m.group('rttavg'))
                    n += 1
                    if n == 1:
                        r.bw = cbw * (1000 ** 2)
                        r.rttavg = crttavg
                        ptm = ctm
                        pbw = cbw
                        prttavg = crttavg
                        continue

                    #calculate duration from last timestamp
                    dt = ctm - ptm
                    #update last timestamp
                    ptm = ctm
                    #update total run time
                    ttot += dt
                    #weigh measure with duration of measure, averaging value
                    rbw += ((cbw + pbw) / 2 ) * dt
                    pbw = cbw
                    rttavg += ((crttavg + prttavg) / 2 ) * dt
                    prttavg = crttavg

                elif m.group('direction') == 'F':
                    pass
        if n > 1:
            #divide by total to get average if more than one measure
            r.bw = (rbw / ttot) * (1000 ** 2)
            # r.rttavg = rttavg/ttot

        return r


class Yaz(object):
    """Interface for the yaz bandwidth measurement program
    !!! Does NOT work in mininet as of yet for unknown reasons..."""
    _EXEC = 'yaz'

    _SND_CMD = '{exec} -v -S {receiverIp} -p {controlPort} -P {port} -l {minPacketSize} -c {initPacketSize} -i {initPacketSpacing} -n {' \
               'streamLength} -m {nStreams} -r {resolution} -s {streamSpacing}'
    _RCV_CMD = '{exec} -v -R -p {controlPort} -P {port}'

    DEFAULT_OPTIONS = {
        'minPacketSize': 200,
        'initPacketSize': 1500,
        'initPacketSpacing': 40,
        'streamLength': 50,
        'nStreams': 1,
        'resolution': 500.0,
        'streamSpacing': 50,
        'controlPort': 13979,
        'port': 13989,
        'duration': 10
    }

    @classmethod
    def bw(cls, nodes, binDir = None, **options):
        """Measure available bandwidth between given nodes
        param nodes: tuple of nodes to measure the bandwidth between
        param binDir: optional bin directory to launch executables from
        param options: options for this run (merged with DEFAULT_OPTIONS)
        """
        sender, receiver = nodes
        opts = {'receiverIp': receiver.IP(),
        }

        if binDir is not None:
            opts['exec'] = os.path.join(binDir, cls._EXEC)
        else:
            opts['exec'] = cls._EXEC

        opts.update(cls.DEFAULT_OPTIONS)
        opts.update(options)
        #start receiver first for yaz
        rcv = receiver.popen(shlex.split(cls._RCV_CMD.format(**opts)))
        # nbsr = NonBlockingStreamReader(rcv.stdout)
        # output = nbsr.readline(0.1)
        snd = sender.popen(shlex.split(cls._SND_CMD.format(**opts)))
        time.sleep(opts['duration'])
        snd.send_signal(signal.SIGINT)
        out, err = snd.communicate()
        rcv.send_signal(signal.SIGINT)
        return cls._parseYaz(out, err)

    @classmethod
    def _parseYaz(cls, out, err):
        print 'out', out
        print'err', err
        r = BwStats()
        return r


# from threading import Thread
# from Queue import Queue, Empty
#
# class NonBlockingStreamReader(object):
#     def __init__(self, stream):
#         '''
#         stream: the stream to read from.
#                 Usually a process' stdout or stderr.
#         '''
#
#         self._s = stream
#         self._q = Queue()
#
#         def _populateQueue(stream, queue):
#             '''
#             Collect lines from 'stream' and put them in 'quque'.
#             '''
#
#             while True:
#                 line = stream.readline()
#                 if line:
#                     queue.put(line)
#                 else:
#                     raise UnexpectedEndOfStream
#
#         self._t = Thread(target = _populateQueue,
#                          args = (self._s, self._q))
#         self._t.daemon = True
#         self._t.start()  #start collecting lines from the stream
#
#     def readline(self, timeout = None):
#         try:
#             return self._q.get(block = timeout is not None,
#                                timeout = timeout)
#         except Empty:
#             return None
#
#
# class UnexpectedEndOfStream(Exception): pass
