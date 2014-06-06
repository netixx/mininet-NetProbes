"""Monitor network usage of a node
"""
import re

from mininet.log import info

rules = [{
             'direction': "INPUT",
             'protocol': "tcp",
             'port': 5000
         },
         {
             'direction': "OUTPUT",
             'protocol': "tcp",
             'port': 5000
         }
]


# _MON_COLLECT_TPL = "sudo iptables -n -L {direction} -v | awk '/pkts/ || /dpt:/ {{print $$1,$$2,$$NF}}' | sort -nk1"
_MON_COLLECT_TPL = "sudo iptables -n -L {direction} -v -x"
_MON_START_TPL = "sudo iptables -A {direction} -p {protocol} --dport {port}"
_MON_STOP_TPL = "sudo iptables -D {direction} -p {protocol} --dport {port}"


_directions = ['INPUT', 'OUTPUT']
_CHAIN_TPL = r"^Chain (?P<direction>%s) \(policy\s+(?P<policy>\w+)\s+(?P<packets>\d+)\s+packets,\s+(?P<bytes>\d+)\s+bytes\s*\)" % r"|".join(_directions)
_PACKET_SPECS = r"|".join([r'{protocol} dpt:{port}'.format(**rule) for rule in rules])
_PACKETS_TPLS = r'^\s+(\d+)\s+(\d+[KMG]?).+?({specs})'.format(specs = _PACKET_SPECS)

def start(node, rules):
    """Start monitoring a node
    :param node: node to monitor
    :param rules: rules to monitor for this node
    """
    info("monitor ")
    for rule in rules:
        node.pexec(_formatCommand(_MON_START_TPL, rule))

def stop(node, rules):
    """Stop monitoring a node
    :param node: node to stop monitoring
    :param rules: rules to stop monitoring
    """
    for rule in rules:
        node.pexec(_formatCommand(_MON_STOP_TPL, rule))

def reset(node, rules):
    """Reset counters for this node"""
    node.stop(rules)
    node.start(rules)

def collect(node, file, counter = None):
    """Collect usage data for this node to file
    :param node: node to use
    :param file: file to write the output to
    :param counter: object to sum bytes and packets
    """
    with open(file, 'a') as f:
        info("Collecting usage data for host %s\n" % node.name)
        f.write("------ Usage statistics for host %s:\n" % node.name)
        for direction in _directions:
            rule = {'direction' : direction}
            f.write(_parseOutput(node.pexec(_formatCommand(_MON_COLLECT_TPL, rule))[0].decode(), counter) + "\n")



def prepareFile(file):
    """Prepare file for monitoring output
    :param file: file to use
    """
    with open(file, 'w') as f:
        f.write("Usage statistics\n")

def writeSummary(file, counter):
    """Write summary contained in counter object to file"""
    with open(file, 'a') as f:
        f.write(counter.printAll())

def _formatCommand(cmd, rule):
    return cmd.format(**rule)

def _parseOutput(output, counter):
    out = ""
    direction = None
    for line in output.splitlines():
        chain = re.match(_CHAIN_TPL, line)
        if chain is not None:
            out += line+"\n"
            out += "{:<12} {:10} {:10}\n".format('proto:port', 'packets', 'bytes')
            direction = chain.group("direction")
            if counter is not None:
                counter.counters[direction]['bytes'] += int(chain.group('bytes'))
                counter.counters[direction]['packets'] += int(chain.group('packets'))
        m = re.match(_PACKETS_TPLS, line)
        if m is not None and direction is not None:
            specs = m.group(3).replace(r' dpt', '')
            if counter is not None:
                # u = re.match(r"(\d+)(K|M|G)", m.group(2))
                # if u is not None:
                #     nbytes = int(u.group(1))
                #     if u.group(2) == 'K':
                #         nbytes *= 1024
                #     elif u.group(2) == 'M':
                #         nbytes *= 1024 ** 2
                #     elif u.group(2) == 'G':
                #         nbytes *= 1024 ** 3
                # else:
                nbytes = int(m.group(2))
                if not counter.counters[direction].has_key(specs):
                    counter.counters[direction][specs] = Counter.newCounter()
                counter.counters[direction][specs]['bytes'] += nbytes
                counter.counters[direction][specs]['packets'] += int(m.group(1))
            out += "{:<12} {:<10} {:<10}\n".format(specs, m.group(1), m.group(2))


    return out + "\n"

class Counter(object):
    """Counts total number of bytes"""

    def __init__(self):
        self.counters = {}
        for d in _directions:
            self.counters[d] = self.newCounter()

    @staticmethod
    def newCounter():
        return {'bytes': 0,
                'packets': 0}

    def printAll(self):
        o = "Packets/Bytes summary :\n"
        for d in _directions:
            o += "Total for direction {:<10}: packets : {packets:<10}, bytes : {bytes:<10}\n".format(d, **self.counters[d])
            for specs in [k for k in self.counters[d].iterkeys() if k not in self.newCounter().iterkeys()]:
                o += "   Rule {:<15}: packets : {packets:<10}, bytes : {bytes:<10}\n".format(specs, **self.counters[d][specs])
        return o
