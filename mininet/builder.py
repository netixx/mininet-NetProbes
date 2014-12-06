# !/usr/bin/env python
"""
    Run Mininet hifi emulation of a custom network topology defined by a .json file
    Start a NetProbes instance per host (can be turned off by options)
    Ability to schedule events to occur on the virtual network via the json file.
    Example are given in the data folder.
"""

import signal
from os import getuid
import os
from argparse import ArgumentParser
import re
import time
import shlex
import collections
from string import Template

from mininet.net import Mininet
from mininet.topolib import Topo
import mininet.log as lg
from custom_mininet import TCLink, CPULimitedHost, Host, CLI
from mininet.node import RemoteController
import tools
from mininet.term import tunnelX11
from events import NetEvent, EventsManager
import events
import monitor


class CustomTopo(Topo):
    """Topology builder for any specified topology in json format"""

    def __init__(self, topoFilePath = None, simParams = None, hostOptions = None, **opts):
        if topoFilePath is None:
            raise RuntimeError("No topology file given")
        super(CustomTopo, self).__init__(**opts)
        self.netOptions = {"host": Host}
        reader = TopologyLoader(topoObj = self, topoFile = topoFilePath, cliParams = simParams)
        reader.loadTopoFromFile()
        self.defaultHostOptions = hostOptions if hostOptions is not None else {}

    def getDefaultHostParams(self):
        return self.defaultHostOptions.copy()

    def getNetOptions(self):
        return self.netOptions

    def setNetOption(self, option, value):
        self.netOptions[option] = value


class TopologyLoader(object):
    """Loads all the information in the json file"""

    # keywords for reading type of equipments
    KW_HOSTS = "hosts"
    KW_LINKS = "links"
    KW_SWITCHES = "switches"
    KW_ROUTERS = "routers"
    KW_EVENTS = "events"
    KW_CHECKS = "checks"

    def __init__(self, topoObj, topoFile, cliParams):
        self.cliParams = cliParams
        self.container = topoObj
        self.fileName = topoFile
        self.nodes = {}
        self.events = []
        self.hosts = []
        self.routers = []
        self.switches = []
        self.checks = []
        self.links = []

    def setOption(self, option, value):
        self.container.setNetOption(option, value)

    def loadTopoFromFile(self):
        from json import load as jload

        graph = None
        with open(self.fileName, 'r') as f:
            graph = jload(f)
        if graph is None:
            raise RuntimeError('Topology could not be read')
        # read all types of expected equipments
        for typeOfEquipment, listOfEquipment in graph.iteritems():
            # switch on equipment type
            if typeOfEquipment == self.KW_HOSTS:
                self.hosts = listOfEquipment
            elif typeOfEquipment == self.KW_LINKS:
                self.links = listOfEquipment
            elif typeOfEquipment == self.KW_ROUTERS:
                self.routers = listOfEquipment
            elif typeOfEquipment == self.KW_SWITCHES:
                self.switches = listOfEquipment
            elif typeOfEquipment == self.KW_EVENTS:
                self.events = listOfEquipment
            elif typeOfEquipment == self.KW_CHECKS:
                self.checks = listOfEquipment
            else:
                raise RuntimeError('Unknown equipment type or keyword')
        # load links last as they require other elements
        self.loadHosts()
        # self.loadRouters()
        self.loadSwitches()
        self.loadLinks()
        self.loadEvents()
        self.loadChecks()

    def loadChecks(self):
        checks = []
        for check in self.checks:
            # events.replaceParams(event, self.cliParams)
            checkParams = {'name': check['name'],
                           'affected_check': check['affected_check'],
                           'unaffected_check': check['unaffected_check']}

            # set checker class to use
            if check['variations'].has_key('delay'):
                kw = 'delay'
            elif check['variations'].has_key('bw'):
                kw = 'bw'
            checkParams['checker'] = kw
            if check['variations'][kw].has_key('options'):
                checkParams['options'] = check['variations'][kw]['options']
            else:
                checkParams['options'] = None
            checkParams['targets'] = dict(check['variations'][kw]['targets'])
            checks.append(checkParams)
        if len(checks) > 0:
            global netchecks
            netchecks = checks
            # self.setOptions('netchecks', checks)


    def registerNode(self, name, node):
        self.nodes[name] = node

    def loadSwitches(self):
        for switch in self.switches:
            name = str(switch["name"])
            # TODO : fill with options
            if switch.has_key("options"):
                o = self.container.addSwitch(name, switch.get("options"))
            else:
                o = self.container.addSwitch(name)
            self.registerNode(name, o)

    def loadRouters(self):
        if len(self.routers) > 0:
            self.setOption('router', RemoteController)
        for router in self.routers:
            # load router
            pass

    def loadHosts(self):
        for host in self.hosts:
            name = str(host["name"])
            if host.has_key("options"):
                opts = host["options"]
                if opts.has_key("cpu"):
                    self.setOption("host", CPULimitedHost)
                if opts.has_key("commandOpts"):
                    cmd = opts['commandOpts']
                    opts['commandOpts'] = Template(cmd).substitute(**self.cliParams)
                o = self.container.addHost(name, **host["options"])
            else:
                o = self.container.addHost(name)
            self.registerNode(name = name, node = o)

    def loadLinks(self):
        for link in self.links:
            opts = None
            hosts = link["hosts"]
            if link.has_key('name'):
                opts = {'name': link['name']}
            if link.has_key("options"):
                if opts is not None:
                    opts.update(link["options"])
                else:
                    opts = link['options']
                if hasTCLinkProperties(opts):
                    self.setOption("link", TCLink)
            if opts is not None:
                self.addLink(hosts[0], hosts[1], opts)
            else:
                self.addLink(hosts[0], hosts[1])


    def addLink(self, host1, host2, options = None):
        host1 = str(host1)
        host2 = str(host2)
        port1 = None
        port2 = None
        p = host1.rpartition(":")
        # if split successful
        if p[1] == ':':
            port1 = p[2]
            host1 = p[0]
        p = host2.rpartition(":")
        if p[1] == ':':
            port2 = p[2]
            host2 = p[0]
        if options is None:
            options = {}
        return self.container.addLink(self.nodes[host1], self.nodes[host2], port1 = port1, port2 = port2, **options)

    def loadEvents(self):
        for event in self.events:
            events.replaceParams(event, self.cliParams)
            e = NetEvent()
            e.target = str(event["target"])
            # variations are handed directly and use the same syntax as the TCLink option
            e.variations = event['variations']
            if hasTCLinkProperties(e.variations):
                self.setOption('link', TCLink)
            expectedTimeFormat = "(\d+)ms"
            # parse durations
            if event.has_key("repeat") and event["repeat"] != "none":
                rm = re.match(expectedTimeFormat, event['repeat'])
                if rm is not None:
                    e.repeat = float(rm.group(1)) / 1000.0
            if event.has_key("duration"):
                rm = re.match(expectedTimeFormat, event["duration"])
                if rm is not None:
                    e.duration = float(rm.group(1)) / 1000.0
                    if e.repeat is not None and e.duration > e.repeat:
                        raise RuntimeError("Duration of the event is greater that the repeat time")
            if event.has_key('nrepeat'):
                e.nrepeat = int(event['nrepeat'])
            # register event for scheduling
            EventsManager.sheduleEvent(e)


def hasTCLinkProperties(opts):
    return (opts.has_key("bw")
            or opts.has_key("delay")
            or opts.has_key("loss")
            or opts.has_key("max_queue_size")
            or opts.has_key("use_htb"))


class CustomMininet(Mininet):
    """Customized mininet network class"""

    def addLink(self, node1, node2, port1 = None, port2 = None,
                cls = None, **params):
        name = None
        if params.has_key('name'):
            name = params['name']
            del params['name']
        if name is not None:
            lg.info('\nLink %s : ' % name)
        l = super(CustomMininet, self).addLink(node1, node2, port1, port2, cls, **params)
        if name is not None:
            self.nameToNode[name] = l
        return l

    def addHost(self, name, cls = None, **params):
        """Add host.
           name: name of host to add
           cls: custom host class/constructor (optional)
           params: parameters for host
           returns: added host"""
        defaults = self.topo.getDefaultHostParams()
        defaults.update(params)
        return super(CustomMininet, self).addHost(name, cls, **defaults)


# Use for routers
from mininet.node import Controller

POXDIR = os.environ['HOME'] + '/pox'
BEACON_EXEC = os.environ['HOME'] + '/beacon-1.0.4/beacon'
FLOODLIGHT_EXEC = os.environ['HOME'] + '/floodlight/floodlight.sh'


class KillController(Controller):
    def terminate(self):
        # print "shell", self.shell
        # lg.error(self.shell)
        # Controller.terminate(self)
        os.killpg(self.pid, signal.SIGKILL)
        self.cleanup()

class POX(Controller):
    def __init__(self, name, cdir = POXDIR,
                 command = 'python pox-debug.py',
                 # cargs = ('openflow.of_01 --port=%s forwarding.l2_learning'),
                 cargs = 'forwarding.l2_learning openflow.of_01 --port=%s',
                 **kwargs):
        Controller.__init__(self, name, cdir = cdir,
                            command = command,
                            cargs = cargs, **kwargs)

class POXRouter(Controller):
    def __init__(self, name, cdir = POXDIR,
                 command = 'python pox-debug.py',
                 # cargs = ('openflow.of_01 --port=%s forwarding.l2_learning'),
                 cargs = 'forwarding.l3_learning openflow.of_01 --port=%s',
                 **kwargs):

        Controller.__init__(self, name, cdir = cdir,
                            command = command,
                            cargs = cargs, **kwargs)


class Beacon(KillController):
    def __init__(self, name, cdir = None,
                 command = BEACON_EXEC,
                 # cargs = ('openflow.of_01 --port=%s forwarding.l2_learning'),
                 cargs = "--port %s",
                 **kwargs):
        Controller.__init__(self, name, cdir = cdir,
                            command = command,
                            cargs = cargs, **kwargs)
        # self.shell = True


class Floodlight(KillController):
    def __init__(self, name, cdir = None,
                 command = FLOODLIGHT_EXEC,
                 # cargs = ('openflow.of_01 --port=%s forwarding.l2_learning'),
                 cargs = "--port %s",
                 **kwargs):
        self.shell = True
        Controller.__init__(self, name, cdir = cdir,
                            command = command,
                            cargs = cargs, **kwargs)
        self.shell = True


controllers = {
    'nox': Controller,
    'beacon': Beacon,
    'pox': POX,
    'floodlight': Floodlight,
    'poxroute' : POXRouter,
}



from mininet.node import OVSKernelSwitch
# from mininet.node import OVSBridge, DefaultController, OVSSwitch
#from mininet.nodelib import LinuxBridge

switches = {
    # 'ovs-kernel-switch' : OVSKernelSwitch,#same as ovs switch
    # 'ovs-switch' : OVSSwitch,
    # 'ovs-bridge' : OVSBridge,
    # 'linux-bridge' : LinuxBridge,
    'default' : OVSKernelSwitch
}


# class MultiSwitch(OVSSwitch):
# "Custom Switch() subclass that connects to different controllers"
# def start(self, controllers):
# return OVSSwitch.start(self, [ cmap[ self.name ] ])
#
# topo = TreeTopo(depth = 2, fanout = 2)
# net = Mininet(topo = topo, switch = MultiSwitch, build = False)
# for c in [ c0, c1 ]:
# net.addController(c)
# net.build()
# net.start()
# CLI(net)
# net.stop()

def popen(host, cmd):
    if args.show_command_output:
        stdout = None
        stderr = None
    else:
        import os

        stdout = open(os.devnull, 'wb')
        stderr = open(os.devnull, 'wb')
    # devnull = open(os.devnull, 'wb')
    # h = host.popen(cmd, stdin = open(os.devnull, 'wb'), stdout = open(os.devnull, 'wb'), stderr = open(os.devnull, 'wb'))
    h = host.popen(cmd, stdin = None, stdout = stdout, stderr = stderr)
    return h


def runCommand(host):
    return popen(host, shlex.split("%s" % host.command))


def interract_once(net):
    import interract

    if args.watcher_start_event or args.watcher_post_event or args.watcher_probe:
        interract.wait_start_events(args.watcher_start_event)

        interract.post_events(net, netprobes, args.watcher_post_event)

        if args.watcher_probe is not None:
            # interract.wait_process(netprobes[args.watcher_probe])
            interract.wait_reset(args.watcher_reset_event)
            interract.make_watcher_results(args.watcher_log, args.watcher_output, vars.topoFile, pparams, watcher_type = args.watcher_type)


def interract_mul(net):
    import interract

    if args.watcher_wait_up is not None:
        interract.wait_file(args.watcher_wait_up)
    try:
        for simargs in args.simulations:
            # parse parameters for this simulation
            if args.sim_prepend is not None:
                pre = args.sim_prepend.format(sim = "'%s'" % simargs)
            else:
                pre = simargs
            interract.pre_events(net, netprobes, pre)

            interract.wait_start_events(args.watcher_start_event)

            interract.post_events(net, netprobes, args.watcher_post_event)

            interract.wait_reset(args.watcher_reset_event)
            interract.make_watcher_results(args.watcher_log, args.watcher_output, vars.topoFile, pparams, watcher_type = args.watcher_type)

    except (Exception, SystemExit) as e:
        lg.error(e)


def interract(net):
    if len(args.simulations) > 0:
        lg.output("Start automatic multiple interaction process\n")
        interract_mul(net)
        return
    if args.watcher_start_event or args.watcher_post_event or args.watcher_probe:
        lg.output('Started automatic interaction process\n')
        interract_once(net)
        return

    return CLI(net)


def runTopo(topoFile, simParams, hostOptions, checkLevel, controller, switch):
    topo = CustomTopo(topoFilePath = topoFile, simParams = simParams, hostOptions = hostOptions)
    if checkLevel > 1:
        topo.setNetOption('link', TCLink)
    # net = CustomMininet(topo = topo, controller = Beacon, autoSetMacs = True, **topo.getNetOptions())
    # net = CustomMininet(topo = topo, controller = Beacon, **topo.getNetOptions())
    net = CustomMininet(topo = topo, controller = controller, switch = switch, **topo.getNetOptions())
    global netprobes
    netprobes = collections.OrderedDict()
    try:
        lg.output('Constructing virtual network..\n')
        start(net)
        check(net, checkLevel)
        lg.output("Starting hosts")
        lg.info(": ")
        for host in net.hosts:
            lg.info("%s " % host.name)
            if host.monitor_rules is not None:
                monitor.start(host, host.monitor_rules)
            if host.command is not None:
                lg.info("cmd ")
                host.command = host.command.format(commandOpts = host.commandOpts, name = host.name).format(name = host.name)
                if host.isXHost:
                    t = makeTerm(host, cmd = host.command)
                    if len(t) < 1:
                        lg.error("Error while starting terminal for host %s\n" % host.name)
                        continue
                    if len(t) == 2:
                        tunnel, term = t
                    else:
                        term = t
                    try:
                        if term.poll() is not None:
                            lg.error(
                                "Terminal with command %s ended early for host %s : %s\n" % (host.command, host.name, repr(term.communicate())))
                    except:
                        pass
                    netprobes[host.name] = term
                else:
                    netprobes[host.name] = runCommand(host)
                    # print(netprobes[host.name].communicate())
            else:
                if host.isXHost:
                    makeTerm(host)
                    lg.info("term ")
            lg.info("done ")
        lg.output("\n")
        EventsManager.startClock(net)
        interract(net)
        mon = False
        counter = monitor.Counter()
        for host in net.hosts:
            if host.monitor_rules is not None:
                monitor.collect(host, monitor_file, counter)
                monitor.stop(host, host.monitor_rules)
                mon = True
        for name, probe in netprobes.iteritems():
            lg.info("Send sigint to probe %s\n" % name)
            import signal

            try:
                probe.send_signal(signal.SIGINT)
                time.sleep(0.05)
            except OSError as e:
                lg.error("Failed to send SIGINT to %s : %s\n" % ( name, e))
        if mon:
            monitor.writeSummary(monitor_file, counter)
    finally:
        stop(net)
        # cleanup !
        lg.info("Stopping remaining processes...\n")
        kill = 0
        for name, probe in netprobes.iteritems():
            if probe.poll() is None:
                kill += 1
        if kill > 0:
            lg.info("Found %s process(es) to kill\n" % kill)
            time.sleep(3)
            for name, probe in netprobes.iteritems():
                if probe.poll() is None:
                    try:
                        lg.info("Send terminate signal to %s\n" % name)
                        probe.terminate()
                        time.sleep(0.001)
                    except OSError as e:
                        lg.error("Failed to terminate %s : %s\n" % (name, e))
            time.sleep(3)
            for name, probe in netprobes.iteritems():
                if probe.poll() is None:
                    try:
                        lg.info("Send kill signal to %s\n" % name)
                        probe.kill()
                    except OSError as e:
                        lg.error("Failed to kill %s : %s\n" % (name, e))
        lg.output("\nAll done\n")


def check(net, level):
    if level > 0:
        import check

        check.check(net, level, netchecks = netchecks)


def startXterm(net):
    for host in net.hosts:
        makeTerm(host)


def start(net):
    # net.start()
    # does start for us
    rootnode = tools.connectToInternet(net)
    lg.info("Ips are as follows :\n")
    for host in net.hosts:
        lg.info("%s:%s\n" % (host.name, host.IP()))
    return rootnode


def stop(net):
    net.stop()
    if EventsManager.t_start is not None:
        EventsManager.t_start.cancel()
    EventsManager.stopClock()


def makeTerm(node, title = 'Node', term = 'xterm', display = None, cmd = 'bash'):
    """Create an X11 tunnel to the node and start up a terminal.
       node: Node object
       title: base title
       term: 'xterm' or 'gterm'
       returns: two Popen objects, tunnel and terminal"""
    title += ': ' + node.name
    if not node.inNamespace:
        title += ' (root)'
    cmds = {
        'xterm': ['xterm', '-title', title, '-geometry', '150x30', '-display'],
        'gterm': ['gnome-terminal', '--title', title, '--display']
    }
    if term not in cmds:
        lg.error('invalid terminal type: %s' % term)
        return
    display, tunnel = tunnelX11(node, display)
    if display is None:
        return []
    term = popen(node, cmds[term] + [display, '-e', 'env TERM=ansi ' + cmd])
    return [tunnel, term] if tunnel else [term]


monitor_file = ""
binDir = None
netchecks = None
if __name__ == '__main__':
    DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(DIR, '..'))

    if getuid() != 0:
        print "Please run this script as root / use sudo."
        exit(-1)

    CustomMininet.init()
    parser = ArgumentParser(description = "Options for starting the custom Mininet network builder")

    parser.add_argument("--show-command-output",
                        dest = "show_command_output",
                        action = 'store_true',
                        default = False,
                        help = "Pipes output of all commands to stdout of this program")
    # emulation options
    parser.add_argument("--topo",
                        dest = 'tfile',
                        help = 'Topology to load for this simulation',
                        default = os.path.join(DIR, 'data/flat.json'))

    parser.add_argument("--vars",
                        dest = 'vars',
                        default = [],
                        action = 'append',
                        help = 'Pass variables to the emulation')

    parser.add_argument("--auto-start-events",
                        dest = 'start_time',
                        default = None,
                        help = 'Time to pass before network is modified')

    #network construction options
    parser.add_argument('-c', '--check',
                        dest = 'net_check',
                        action = 'count',
                        default = 0,
                        help = 'Set level of checks to perform after network construction.')

    #command options
    ncmds = parser.add_mutually_exclusive_group()

    ncmds.add_argument('--no-command',
                       dest = 'no_command',
                       action = 'store_true',
                       default = False,
                       help = 'Do not execute command on the hosts')

    cmds = ncmds.add_argument_group()
    cmds.add_argument('--command',
                      dest = 'command',
                      help = 'Command to run on nodes',
                      default = '$HOME/netprobes/start.sh {commandOpts}')

    cmds.add_argument('--strace',
                      dest = 'strace',
                      action = 'store_true',
                      default = False,
                      help = 'Run strace on command inside nodes')

    #monitoring options
    parser.add_argument('--monitor',
                        dest = 'monitor_file',
                        default = False,
                        help = 'Monitor each host for network usage')

    parser.add_argument('--force-x',
                        dest = 'force_x',
                        action = 'store_true',
                        default = False,
                        help = 'Force start XTerm terminal for each host')

    parser.add_argument('--bin-dir',
                        dest = 'bin_dir',
                        default = os.path.join(ROOT_DIR, 'bin'),
                        help = 'Path to the bin directory')

    parser.add_argument('--watcher-output',
                        dest = 'watcher_output',
                        default = None,
                        help = "Path to the output file")

    parser.add_argument('--watcher-log',
                        dest = 'watcher_log',
                        default = None,
                        help = "Path to the watcher log file")

    parser.add_argument('--watcher-probe',
                        dest = 'watcher_probe',
                        default = None,
                        help = "Name of the watcher probe (hxxx)")

    parser.add_argument('--watcher-wait-up',
                        dest = 'watcher_wait_up',
                        default = None,
                        help = "Location of file to wait before NetProbes overlay is complete")

    parser.add_argument('--watcher-start-event',
                        dest = 'watcher_start_event',
                        default = None,
                        help = "Location of file to wait for starting events")

    parser.add_argument('--watcher-post-event',
                        dest = "watcher_post_event",
                        default = None,
                        help = "Action to perform after events have been applied.")

    parser.add_argument('--watcher-reset-event',
                        dest = "watcher_reset_event",
                        default = None,
                        help = "Location of file to wait for the reset events.")

    parser.add_argument('--watcher-type',
                        dest = "watcher_type",
                        choices = ['delay', 'bw'],
                        default = "delay",
                        help = "Type of the watcher to launch")

    parser.add_argument('-q', '--quiet',
                        dest = 'quiet',
                        action = 'count',
                        default = 0,
                        help = "Set verbosity.")

    parser.add_argument('--sim-prepend',
                        dest = 'sim_prepend',
                        help = "Command to prepend before running each simulation")

    parser.add_argument('--sim',
                        dest = 'simulations',
                        action = 'append',
                        default = [],
                        help = 'Simulation to perform without restarting network.')

    parser.add_argument('--controller',
                        choices = controllers.keys(),
                        default = 'nox',
                        help = "Specify controller to use (see mininet doc)")

    parser.add_argument('--switch',
                        choices = switches.keys(),
                        default = 'default',
                        help = "Specify switch to use (see mininet doc)")

    # import sys
    #
    # print  sys.argv

    args = parser.parse_args()

    #increase file descriptor limits
    from mininet.util import fixLimits

    fixLimits()

    if args.quiet >= 1:
        lg.setLogLevel('output')
    else:
        lg.setLogLevel('info')

    # topoFile = os.path.join(DIR, "data", args.tfile + ".json")
    topoFile = args.tfile
    if args.start_time is not None:
        EventsManager.start_time = int(args.start_time)
    # monitorUsage = args.monitorUsage
    import vars

    vars.watcher_log = args.watcher_log
    vars.testBinPath = args.bin_dir
    hOpts = {'commandOpts': "-id {name}"}
    if args.monitor_file:
        monitor_file = args.monitor_file
        monitor.prepareFile(monitor_file)
        hOpts['monitor_rules'] = monitor.rules

    if args.no_command:
        hOpts['command'] = None
    elif args.command:
        hOpts['command'] = args.command
        if args.strace:
            hOpts['command'] = "strace {command}".format(command = hOpts['command'])
    if args.force_x:
        hOpts['isXHost'] = True

    pparams = {}
    # record parameters from the cli
    for param in args.vars:
        p = param.partition("=")
        pparams[p[0]] = p[2]

    vars.topoFile = topoFile
    vars.watcher_output = args.watcher_output
    runTopo(topoFile = topoFile,
            simParams = pparams,
            hostOptions = hOpts,
            checkLevel = args.net_check,
            controller = controllers[args.controller],
            switch = switches[args.switch])
