'''
Created on 8 avr. 2014

@author: francois
'''
import mininet
from mininet.link import TCIntf
import re

class TCLink(mininet.link.TCLink):
    def __init__(self, node1, node2, port1 = None, port2 = None,
                  intfName1 = None, intfName2 = None, **params):
        mininet.link.TCLink.__init__(self, node1, node2, port1 = port1, port2 = port2,
                       intfName1 = intfName1, intfName2 = intfName2, **params)

    def set(self, newparams):
        self.setIntf(self.intf1, newparams)
        self.setIntf(self.intf2, newparams)

    def setIntf(self, interface, newparams):
        params = interface.params.copy()
        params.update(newparams)
        config(interface, params)

    def reset(self):
        self.resetIntf(self.intf1)
        self.resetIntf(self.intf2)

    def resetIntf(self, interface):
        config(interface, interface.params)

def rawCmd(node, *args):
        cmd = ""
        # Allow sendCmd( [ list ] )
        if len(args) == 1 and type(args[ 0 ]) is list:
            cmd = args[ 0 ]
        # Allow sendCmd( cmd, arg1, arg2... )
        elif len(args) > 0:
            cmd = args
        # Convert to string
        if not isinstance(cmd, str):
            cmd = ' '.join([ str(c) for c in cmd ])
        if not re.search(r'\w', cmd):
            # Replace empty commands with something harmless
            cmd = 'echo -n'
        node.write(cmd + '\n')

def tc(interface, cmd, tc = 'tc'):
    "Execute tc command for our interface"
    c = cmd % (tc, interface)  # Add in tc command and our name
#     debug(" *** executing command: %s\n" % c)
    return rawCmd(interface.node, c)

def config(interface, params):
    cmds = [ '%s qdisc del dev %s root' ]
    cmd, parent = TCIntf.bwCmds(interface, **filterBwOpts(params))
    cmds += cmd
    cmd, parent = TCIntf.delayCmds(parent, **filterDelayOpts(params))
    cmds += cmd
    for cmd in cmds:
        tc(interface, cmd)

def filterBwOpts(options):
    return {k: v for k, v in options.iteritems() if isBwOption(k)}

def filterDelayOpts(options):
    return {k: v for k, v in options.iteritems() if isDelayOptions(k)}

def isBwOption(opt):
    return (opt == 'bw' or
            opt == 'speedup' or
            opt == 'use_hfsc' or
            opt == 'use_tbf' or
            opt == 'latency_ms' or
            opt == 'enable_ecn' or
            opt == 'enable_red')

def isDelayOptions(opt):
    return (opt == 'delay' or
            opt == 'jitter' or
            opt == 'loss' or
            opt == 'max_queue_size')
