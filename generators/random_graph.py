# !/usr/bin/python3
"""Generate tree topology"""
import struct
import socket


def ip2int(addr):
    # addr = ".".join(str(i) for i in addr)
    return struct.unpack("!I", socket.inet_aton(addr))[0]


def int2ip(addr):
    return socket.inet_ntoa(struct.pack("!I", addr))

# PARAMETERS
hostname = "h"
ipsep = "."
linkname = "l"
linknamesep = "-"
switchname = "s"

ipPrefix = "10."
# ipBase = ip2int(ipPrefix)

maxIpPart = 254

out = {'links': [],
       'switches': [],
       'hosts': []
}


def addLink(left, right):
    out['links'].append(
        {
            'hosts': [left, right],
            'name': linkname + linknamesep.join([left, right])
        })


def numToSw(num):
    return switchname + "%s" % num


def addSwitch(name):
    n = numToSw(name)
    out['switches'].append(
        {
            'name': n
        }
    )
    return n


def numToHost(num):
    return hostname + "%s" % num


def addHost(name, ip):
    n = numToHost(name)
    out['hosts'].append(
        {
            'name': n,
            'options': {
                'ip': ip
            }
        }
    )
    return n


def strIp(*args):
    return ipPrefix + ipsep.join(map(str, args))


import argparse
import networkx as nx


def getIp():
    ip1 = 0
    ip2 = 0
    ip3 = 0
    while ip1 <= maxIpPart:
        while ip2 <= maxIpPart:
            while ip3 <= maxIpPart:
                ip3 += 1
                yield [ip1, ip2, ip3]
            ip2 += 1
        ip1 += 1


generators = {
    'erdos_renyi': nx.generators.random_graphs.erdos_renyi_graph,
    'gnp': nx.generators.random_graphs.fast_gnp_random_graph,

}
parser = argparse.ArgumentParser()
parser.add_argument('--hosts',
                    type = int,
                    default = 20,
                    dest = 'nhosts')

parser.add_argument('--pedges',
                    type = float,
                    default = 0.20,
                    dest = 'pedges')

parser.add_argument('--generator',
                    dest = 'generator',
                    choices = generators.keys(),
                    default = 'gnp')

args = parser.parse_args()

g = generators[args.generator](args.nhosts, args.pedges)

for e in g.edges_iter():
    addLink(numToSw(e[0] + 1), numToSw(e[1] + 1))

for s in g:
    addSwitch(s + 1)

ip = getIp()
for i in range(1, args.nhosts + 1):
    addHost(i, strIp(*ip.next()))
    addLink(numToHost(i), numToSw(i))

import json

print(json.dumps(out, sort_keys = True, indent = 4, separators = (',', ': ')))
