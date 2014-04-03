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
from mininet.link import TCLink

import tools

class CustomTopo(Topo):
    """Topology builder for any specified topology in json format"""
    
    def __init__(self, topoFilePath = None, **opts):
        if topoFilePath is None:
            raise RuntimeError("No topology file given")
        super(CustomTopo, self).__init__(**opts)
        self.netOptions = {}
        reader = TopologyLoader(topoObj = self, topoFile = topoFilePath)
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

    def __init__(self, topoObj, topoFile):
        self.container = topoObj
        self.fileName = topoFile
        self.nodes = {}
        
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
            else :
                raise RuntimeError('Unknown equipment type')
        # load links last as they require other elements
        self.loadHosts()
#         self.loadRouters()
        self.loadSwitches()
        self.loadLinks()

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
        for router in self.routers:
            #load router
            pass
    
    def loadHosts(self):
        for host in self.hosts:
            name = str(host["name"])
            if host.has_key("options"):
                o = self.container.addHost(name, host["options"])
            else :
                o = self.container.addHost(name)
            self.registerNode(name = name, node = o)

    
    def loadLinks(self):
        for link in self.links:
            hosts = link["hosts"]
            if link.has_key("options"):
                opts = link["options"]
                if opts.has_key("bw") or opts.has_key("delay"):
                    self.setOption("link", TCLink)
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

class CustomMininet(Mininet):
    pass
#     def build(self):
#         Mininet.build(self)
        # do other stuff eventually


def runTopo(topoFile):
    topo = CustomTopo(topoFilePath = topoFile)
    net = CustomMininet(topo = CustomTopo(topoFilePath = topoFile), **topo.getNetOptions())
    try :
        start(net)

        CLI(net)
    finally:
        net.stop()

def runTopoWithNetProbes(topoFile, netProbePath):
    topo = CustomTopo(topoFilePath = topoFile)
    net = CustomMininet(topo = CustomTopo(topoFilePath = topoFile), **topo.getNetOptions())
    try :
        start(net)

        cmd = netProbePath
        for host in net.hosts:
            host.cmd(cmd + ' &')

        CLI(net)
    finally:
        for host in net.hosts:
            host.cmd('kill %' + cmd)
    
        net.stop()
 
def start(net):
    net.start()
    rootnode = tools.connectToInternet(net)
    print "Ips are as follows :"
    for host in net.hosts:
        print host.name, host.IP()
    return rootnode

if __name__ == '__main__':
    lg.setLogLevel('info')

    DIR = os.path.dirname(os.path.abspath(__file__))
    if getuid() != 0:
        print "Please run this script as root / use sudo."
        exit(-1)

    CustomMininet.init()
    tfile = "flat.json"
    topoFile = os.path.join(DIR, "data", tfile)
    netProbePath = "/home/mininet/netprobes/start.sh"
    #     runTopo(topoFile = topoFile)
    runTopoWithNetProbes(topoFile = topoFile, netProbePath = netProbePath)
        
