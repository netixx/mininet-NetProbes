#!/usr/bin/env python

'''
    Copyright (C) 2012  Stanford University

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
    Description: Load topology in Mininet
    Author: James Hongyi Zeng (hyzeng_at_stanford.edu)
'''

from os import getuid
import os
from mininet.cli import CLI
from mininet.net import Mininet
from mininet.topolib import Topo
import mininet.log as lg
from custom_mininet import TCLink
from mininet.node import CPULimitedHost, RemoteController

import tools, events
from mininet.term import tunnelX11

from argparse import ArgumentParser
import re
from events import NetEvent

class CustomTopo(Topo):
    """Topology builder for any specified topology in json format"""
    
    def __init__(self, topoFilePath = None, params = {}, **opts):
        pparams = {}
        # record parameters from the cli
        for param in params :
            p = param.partition("=")
            pparams[p[0]] = p[2]

        if topoFilePath is None:
            raise RuntimeError("No topology file given")
        super(CustomTopo, self).__init__(**opts)
        self.netOptions = {}
        reader = TopologyLoader(topoObj = self, topoFile = topoFilePath, cliParams = pparams)
        reader.loadTopoFromFile()

    def getNetOptions(self):
        return self.netOptions

    def setNetOption(self, option, value):
        self.netOptions[option] = value

class TopologyLoader(object):
    # keywords for reading type of equipments
    KW_HOSTS = "hosts"
    KW_LINKS = "links"
    KW_SWITCHES = "switches"
    KW_ROUTERS = "routers"
    KW_EVENTS = "events"

    def __init__(self, topoObj, topoFile, cliParams):
        self.cliParams = cliParams
        self.container = topoObj
        self.fileName = topoFile
        self.nodes = {}
        self.events = []
        self.hosts = []
        self.routers = []
        self.switches = []
        
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
            else :
                raise RuntimeError('Unknown equipment type or keyword')
        # load links last as they require other elements
        self.loadHosts()
#         self.loadRouters()
        self.loadSwitches()
        self.loadLinks()
        self.loadEvents()

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
            self.setOptions('router', RemoteController)
        for router in self.routers:
            #load router
            pass
    
    def loadHosts(self):
        for host in self.hosts:
            name = str(host["name"])
            if host.has_key("options"):
                opts = host["options"]
                if opts.has_key("cpu"):
                    self.setOptions("host", CPULimitedHost)
                o = self.container.addHost(name, **host["options"])
            else :
                o = self.container.addHost(name)
            self.registerNode(name = name, node = o)

    
    def loadLinks(self):
        for link in self.links:
            opts = None
            hosts = link["hosts"]
            if link.has_key('name'):
                opts = {}
                opts['name'] = link['name']
            if link.has_key("options"):
                if opts is not None:
                    opts.update(link["options"])
                else :
                    opts = link['options']
                if hasTCLinkProperties(opts):
                    self.setOption("link", TCLink)
            if opts is not None:
                self.addLink(hosts[0], hosts[1], opts)
            else:
                self.addLink(hosts[0], hosts[1])

    
    def addLink(self, host1, host2, options = {}):
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
                    if e.repeat is not None and e.duration > e.repeat :
                        raise RuntimeError("Duration of the event is greater that the repeat time")
            if event.has_key('nrepeat'):
                e.nrepeat = int(event['nrepeat'])
            # register event for scheduling
            events.sheduleEvent(e)

def hasTCLinkProperties(opts):
    return (opts.has_key("bw")
                or opts.has_key("delay")
                or opts.has_key("loss")
                or opts.has_key("max_queue_size")
                or opts.has_key("use_htb"))


class CustomMininet(Mininet):

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
        return l;

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

def runTopo(topoFile, params):
    topo = CustomTopo(topoFilePath = topoFile, params = params)
    net = CustomMininet(topo = topo, **topo.getNetOptions())
    try :
        start(net)
        if xterm:
            startXterm(net)
        events.startClock(net)
        CLI(net)
    finally:
        stop(net)

def runTopoWithNetProbes(topoFile, netProbePath, params):
    topo = CustomTopo(topoFilePath = topoFile, params = params)
    net = CustomMininet(topo = topo, **topo.getNetOptions())
    try :
        start(net)

        cmd = netProbePath + ' -id {name}'
        if strace:
            cmd = "-c 'strace " + cmd + " '"
        for host in net.hosts:
            acmd = cmd.format(name = host.name)
            if xterm:
                makeTerm(host, cmd = acmd)
            else:
                host.cmd(acmd + ' &')

        events.startClock(net)
        CLI(net)
    finally:
        for host in net.hosts:
            host.cmd('kill %' + cmd)
    
        stop(net)
 
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
    if events.t_start is not None:
        events.t_start.cancel()
    events.stopClock()

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
        'xterm': [ 'xterm', '-title', title, '-display' ],
        'gterm': [ 'gnome-terminal', '--title', title, '--display' ]
    }
    if term not in cmds:
        lg.error('invalid terminal type: %s' % term)
        return
    display, tunnel = tunnelX11(node, display)
    if display is None:
        return []
    term = node.popen(cmds[ term ] + [ display, '-e', 'env TERM=ansi bash ' + cmd])
    return [ tunnel, term ] if tunnel else [ term ]


xterm = False
strace = False
if __name__ == '__main__':
    lg.setLogLevel('info')

    DIR = os.path.dirname(os.path.abspath(__file__))
    if getuid() != 0:
        print "Please run this script as root / use sudo."
        exit(-1)

    CustomMininet.init()
    parser = ArgumentParser(description = "Options for starting the custom Mininet network builder")
    parser.add_argument('--np-path',
                        dest = 'netProbesPath',
                        help = 'Absolute path to the NetProbes start script',
                        default = '$HOME/netprobes/start.sh')

    parser.add_argument("--topo",
                        dest = 'tfile',
                        help = 'Topology to load for this simulation',
                        default = 'flat')

    parser.add_argument('--no-netprobes',
                        dest = 'no_netprobes',
                        action = 'store_true',
                        help = 'Do not start NetProbe probes on the hosts')

    parser.add_argument("--vars",
                        dest = 'vars',
                        nargs = '+',
                        default = [],
                        help = 'Pass variables to the emulation')

    parser.add_argument("--start",
                       dest = 'start_time',
                       default = 5,
                       help = 'Time to pass before network is modified')

    parser.add_argument('-x', '--xterm',
                        dest = 'xterm',
                        action = 'store_true',
                        help = 'Start XTerm terminal for each host')

    parser.add_argument('--strace',
                        dest = 'strace',
                        action = 'store_true',
                        help = 'Run strace on command inside nodes')

    args = parser.parse_args()
    topoFile = os.path.join(DIR, "data", args.tfile + ".json")
    events.start_time = int(args.start_time)
    xterm = args.xterm
    strace = args.strace
    if (args.no_netprobes):
        runTopo(topoFile = topoFile, params = args.vars)
    else:
        runTopoWithNetProbes(topoFile = topoFile, netProbePath = args.netProbesPath, params = args.vars)
        
