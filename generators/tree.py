# !/usr/bin/python3
"""Generate tree topology"""
import struct
import socket
import json

def ip2int(addr):
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
    l = {
        'hosts': [left, right],
        'name': linkname + linknamesep.join([left, right])
    }
    if len(linkOptions) > 0:
        l['options'] = {}
        for op in linkOptions:
            l['options'].update(op)
    out['links'].append(l)


def addSwitch(name):
    n = switchname + "%s" % name
    out['switches'].append(
        {
            'name': n
        }
    )
    return n


def addHost(name, ip):
    n = hostname + "%s" % name
    o = {
        'name': n,
        'options': {
            'ip': ip
        }
    }
    for opt in hostOptions:
        o['options'].update(opt)
    out['hosts'].append(o)
    return n


def strIp(*args):
    return ipPrefix + ipsep.join(map(str, args))


def getIp(hostNum, treeId):

    #leafs
    if len(treeId) == dep:
        #look at the first treeIds
        ptid = treeId[:-pdep]
        p = prefixes[tuple(ptid)]
        iptid = treeId[-pdep:]
        ip = 1 + sum((fan ** i) * j for i, j in enumerate(reversed(iptid)))
        r = strIp("0", p, ip)
        return r


hostNum = 1
switchNum = 1


def addTree(depth, fanout, treeId):
    global hostNum
    global switchNum
    """Add a subtree starting with node n.
       returns: last node added"""
    isSwitch = depth > 0
    if isSwitch:
        node = addSwitch(switchNum)
        switchNum += 1
        if depth > 1:
            rang = range(fanout)
        else:
            rang = range(leafs)

        for i in rang:
            child = addTree(depth - 1, fanout, treeId + [i])
            addLink(node, child)

        if depth > 1:
            for i in range(swLeafs):
                child = addHost(hostNum, getIp(hostNum, treeId))
                hostNum += 1
                addLink(node, child)
    else:
        node = addHost(hostNum, getIp(hostNum, treeId))
        hostNum += 1
    return node

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--depth',
                    dest = 'depth',
                    type = int,
                    default = 3)
parser.add_argument('--fanout',
                    dest = 'fanout',
                    type = int,
                    default = 2)
parser.add_argument('--leafs',
                    dest = 'leafs',
                    type = int,
                    default = -1)
parser.add_argument('--extra-leafs',
                    dest = 'sw_leafs',
                    type = int,
                    default = 0)

parser.add_argument('--prefix-depth',
                    dest = 'pref_depth',
                    type = int,
                    default = 1,
                    help = "Each hop closer than prefix depth hop to another is in the same prefix")

parser.add_argument('--host-option',
                    dest = 'host_options',
                    action = 'append',
                    default = []
)

parser.add_argument('--link-options',
                    dest = 'link_options',
                    action = 'append',
                    default = []
)

args = parser.parse_args()

dep = args.depth
fan = args.fanout
leafs = args.leafs if args.leafs >= 0 else fan
swLeafs = args.sw_leafs
pdep = args.pref_depth
hostOptions = [json.loads(opt) for opt in args.host_options]
linkOptions = [json.loads(opt) for opt in args.link_options]

nhosts = fan ** (dep - 1) * leafs + swLeafs * fan ** (dep)
availableIps = []

a = dep-pdep
prefixes = {}

import itertools
comb = itertools.product((i for i in range(fan)), repeat = a)
for i, t in enumerate(comb):
    prefixes[t] = i+1


addTree(dep, fan, [])

import sys
print >>sys.stderr, nhosts

print(json.dumps(out, sort_keys = True, indent = 4, separators = (',', ': ')))
