#!/usr/bin/env python
"""
    Run Mininet hifi emulation of a custom network topology defined by a .json file
    Start a NetProbes instance per host (can be turned off by options)
    Ability to schedule events to occur on the virtual network via the json file.
    Example are given in the data folder.
"""

from os import getuid
import os
from argparse import ArgumentParser
import re

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

    def __init__(self, topoFilePath = None, simParams = {}, hostOptions = None, **opts):
        pparams = {}
        # record parameters from the cli
        for param in simParams:
            p = param.partition("=")
            pparams[p[0]] = p[2]

        if topoFilePath is None:
            raise RuntimeError("No topology file given")
        super(CustomTopo, self).__init__(**opts)
        self.netOptions = {"host": Host}
        reader = TopologyLoader(topoObj = self, topoFile = topoFilePath, cliParams = pparams)
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
            #switch on equipment type
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
        #         self.loadRouters()
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

            #set checker class to use
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
            #load router
            pass

    def loadHosts(self):
        for host in self.hosts:
            name = str(host["name"])
            if host.has_key("options"):
                opts = host["options"]
                if opts.has_key("cpu"):
                    self.setOption("host", CPULimitedHost)
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
        if options is None : options = {}
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
# class POX(Controller):
#     def __init__(self, name, cdir = None,
#                   command = 'python pox.py',
#                   cargs = ('openflow.of_01 --port=%s '
#                           'forwarding.l2_learning'),
#                   **kwargs):
#         Controller.__init__(self, name, cdir = cdir,
#                              command = command,
#                              cargs = cargs, **kwargs)
#
# class MultiSwitch(OVSSwitch):
#     "Custom Switch() subclass that connects to different controllers"
#     def start(self, controllers):
#         return OVSSwitch.start(self, [ cmap[ self.name ] ])
#
# topo = TreeTopo(depth = 2, fanout = 2)
# net = Mininet(topo = topo, switch = MultiSwitch, build = False)
# for c in [ c0, c1 ]:
#     net.addController(c)
# net.build()
# net.start()
# CLI(net)
# net.stop()

def runTopo(topoFile, simParams, hostOptions, checkLevel):
    topo = CustomTopo(topoFilePath = topoFile, simParams = simParams, hostOptions = hostOptions)
    if checkLevel > 1:
        topo.setNetOption('link', TCLink)
    net = CustomMininet(topo = topo, **topo.getNetOptions())
    netprobes = []
    try:
        start(net)
        EventsManager.startClock(net)
        for host in net.hosts:
            if host.monitor_rules is not None:
                monitor.start(host, host.monitor_rules)
            if host.command is not None:
                host.command = host.command.format(commandOpts = host.commandOpts, name = host.name).format(name = host.name)
                if host.isXHost:
                    makeTerm(host, cmd = host.command)
                else:
                    import shlex
                    netprobes.append(host.popen(shlex.split("bash -c '%s' &" % host.command)))
            else:
                if host.isXHost:
                    makeTerm(host)
        check(net, checkLevel)
        CLI(net)
    finally:
        mon = False
        counter = monitor.Counter()
        for host in net.hosts:
            if host.monitor_rules is not None:
                monitor.collect(host, monitor_file, counter)
                monitor.stop(host, host.monitor_rules)
                mon = True
            if host.command is not None:
                # print("%s %s"%(host.name,host.lastPid))
                # import subprocess
                # subprocess.call(["/bin/kill", "-s INT", str(host.lastPid)])
                host.cmd('kill -s INT %')
                # host.sendInt()
        for probe in netprobes:
            import signal
            probe.send_signal(signal.SIGINT)
        if mon:
            monitor.writeSummary(monitor_file, counter)
        stop(net)


def check(net, level):
    if level > 0:
        import check

        check.check(net, level, netchecks = netchecks)


def startXterm(net):
    for host in net.hosts:
        makeTerm(host)


def start(net):
    net.start()
    rootnode = tools.connectToInternet(net)
    print "Ips are as follows :"
    for host in net.hosts:
        print host.name, host.IP()
    return rootnode


def stop(net):
    net.stop()
    if EventsManager.t_start is not None:
        EventsManager.t_start.cancel()
    EventsManager.stopClock()


def makeTerm(node, title = 'Node', term = 'xterm', display = None, cmd = ''):
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
    term = node.popen(cmds[term] + [display, '-e', 'env TERM=ansi bash ' + cmd])
    return [tunnel, term] if tunnel else [term]


monitor_file = ""
binDir = None
netchecks = None
if __name__ == '__main__':
    lg.setLogLevel('info')

    DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(DIR, '..'))

    if getuid() != 0:
        print "Please run this script as root / use sudo."
        exit(-1)

    CustomMininet.init()
    parser = ArgumentParser(description = "Options for starting the custom Mininet network builder")

    #emulation options
    parser.add_argument("--topo",
                        dest = 'tfile',
                        help = 'Topology to load for this simulation',
                        default = 'flat')

    parser.add_argument("--vars",
                        dest = 'vars',
                        nargs = '+',
                        default = [],
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

    args = parser.parse_args()
    topoFile = os.path.join(DIR, "data", args.tfile + ".json")
    if args.start_time is not None:
        EventsManager.start_time = int(args.start_time)
    # monitorUsage = args.monitorUsage
    import vars

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
            hOpts['command'] = "-c 'strace {command} '".format(command = hOpts['command'])
    if args.force_x:
        hOpts['isXHost'] = True

    runTopo(topoFile = topoFile,
            simParams = args.vars,
            hostOptions = hOpts,
            checkLevel = args.net_check)
