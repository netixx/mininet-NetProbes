"""
Created on 8 avr. 2014

@author: francois
"""
import argparse
import re
import shlex

import mininet
from mininet.link import TCIntf
import monitor
from events import EventsManager


class CLI(mininet.cli.CLI):
    def do_monitor(self, line):
        """Start/stop or collect monitoring network usage of the nodes"""
        print("Not implemented")
        parser = argparse.ArgumentParser()
        # parser.add_argument('command',
        # metavar = 'COMMAND',
        # choices = ['start', 'stop', 'collect', 'reset'],
        # dest = 'command',
        #                     help = 'Monitor command to run')
        #
        # parser.add_argument('host',
        #                     # action = 'append',
        #                     # metavar = 'host',
        #                     nargs = '*',
        #                     help = 'Host to monitor')
        #
        subp = parser.add_subparsers(dest = 'subparser_name')
        # parser for the add command
        # # subp1 = subp.add_parser('start')
        #
        # # parser for the do command
        # # subp2 = subp.add_parser('stop')
        # # subp2.add_argument('test', metavar = 'test',
        # #                    help = 'The message you want to send to the probe')
        # # subp2.add_argument('options', nargs = argparse.REMAINDER)
        # # subp2.set_defaults(func = self.stop)
        #
        # # parse for the remove command
        subp3 = subp.add_parser('collect')
        subp3.add_argument('file',
                           help = "File to collect the data to")

        # # subp3.set_defaults(func = self.collect)
        # # subp4 = subp.add_parser('reset')
        # # subp4.set_defaults(func = self.reset)
        #
        args = parser.parse_args(shlex.split(line))
        for host in self.mn.hosts():
            monitor.collect(host, args.file)
            #
            # if len(args.host) == 0:
            #     hosts = self.mn.hosts()
            # else:
            #     hosts = [self.mn.getNde(h) for h in args.hosts]
            #
            # if args.command == 'start':
            #     func = monitor.start
            # elif args.command == 'stop':
            #     func = monitor.stop
            # elif args.command == 'collect':
            #     func = monitor.collect
            # elif args.command == 'reset':
            #     func = monitor.reset
            #
            # for host in hosts:
            #     func(monitor.getMonitor(host))

    def do_events(self, line):
        try:
            parser = argparse.ArgumentParser()
            subp = parser.add_subparsers(dest = 'command')
            subp1 = subp.add_parser('start')
            args = parser.parse_args(shlex.split(line))
            if args.command == 'start':
                EventsManager.startTimers()
        except (SystemExit, Exception) as e:
            mininet.log.error('A problem occurred : %s\n' % e)

    def do_watchers(self, line):
        import vars, os

        if vars.watcher_output is not None and os.path.exists(vars.watcher_output):
            import watcher_delay
            import datetime

            watcher_delay.appendResults(watcher_delay.makeResults(vars.watcher_output, vars.topoFile))
            # prevent results from being processed twice
            os.rename(vars.watcher_output, 'watchers/output/%s.json' % datetime.datetime.now())


class _Host(object):
    """A host with a command to run on startup"""

    def __init__(self, isXHost = False, command = None, commandOpts = None, monitor_rules = None):
        self.isXHost = isXHost
        self.command = command
        self.commandOpts = commandOpts
        self.monitor_rules = monitor_rules


class Host(mininet.node.Host, _Host):
    def __init__(self, name, inNamespace = True, isXHost = False, command = None, commandOpts = None, monitor_rules = None, **params):
        mininet.node.Host.__init__(self, name, inNamespace, **params)
        _Host.__init__(self, isXHost = isXHost, command = command, commandOpts = commandOpts, monitor_rules = monitor_rules)


class CPULimitedHost(mininet.node.CPULimitedHost, _Host):
    def __init__(self, name, sched = 'cfs', isXHost = False, command = None, commandOpts = None, monitor_rules = None, **kwargs):
        mininet.node.CPULimitedHost.__init__(self, name, sched = sched, **kwargs)
        _Host.__init__(self, isXHost = isXHost, command = command, commandOpts = commandOpts, monitor_rules = monitor_rules)


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
    if len(args) == 1 and type(args[0]) is list:
        cmd = args[0]
    # Allow sendCmd( cmd, arg1, arg2... )
    elif len(args) > 0:
        cmd = args
    # Convert to string
    if not isinstance(cmd, str):
        cmd = ' '.join([str(c) for c in cmd])
    if not re.search(r'\w', cmd):
        # Replace empty commands with something harmless
        cmd = 'echo -n'
    out, err, returncode = node.pexec(cmd)
    if returncode > 0:
        raise Exception('Command failed : %s' % err)


def tc(interface, cmd, tc = 'tc'):
    """Execute tc command for our interface"""
    c = cmd % (tc, interface)  # Add in tc command and our name
    # debug(" *** executing command: %s\n" % c)
    return rawCmd(interface.node, c)


def config(interface, params):
    try:
        tc(interface, '%s qdisc del dev %s root')
    except Exception:
        mininet.log.error("Nothing to reset\n")
    cmds = []
    cmd, parent = TCIntf.bwCmds(interface, **filterBwOpts(params))
    cmds += cmd
    cmd, parent = TCIntf.delayCmds(parent, **filterDelayOpts(params))
    cmds += cmd
    try:
        for cmd in cmds:
            tc(interface, cmd)
    except Exception as e:
        raise Exception("Could not configure interface with %s : %s\n" % (cmd, e))


def filterBwOpts(options):
    o = {k: v for k, v in options.iteritems() if isBwOption(k)}
    if o.has_key('bw'):
        o['bw'] = float(o['bw'])
    return o


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
