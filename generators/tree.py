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
    # ip = 1 + sum((fan ** (dep-i)) * j for i, j in enumerate(reversed(treeId)))
    # if len(treeId) < 3:
    #     print treeId, list(reversed(treeId)),ip
    # # for i, j in enumerate(reversed(treeId)):
    # #     print i,j,(fan ** i) * j
    # r = int2ip(ipBase + ip)
    # print r
    # return r
    # depth = len(treeId)
    # # a leaf
    # # if depth - dep == 0:
    # ip1 = 0
    # if depth == 0:
    # ip3 = hostNum % maxIpPart
    # ip2 = hostNum / maxIpPart
    # else:
    # ip3 = treeId[-1]
    # sig = treeId[:-1]
    # n = -sum([fan ** (i + 1) for i in range(len(sig))])
    # for i, id in enumerate(sig):
    # n += id * fan ** (len(sig) - i)
    #     ip2 = n / fan + 1  # sum([fan**i for i in treeId[:-1]])
    #     if ip2 > maxIpPart:
    #         ip1 = ip2 / maxIpPart
    #         ip2 = ip2 % maxIpPart
    #
    # print hostNum, treeId, [ip1, ip2, ip3]
    # return availableIps[hostNum - 1]


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

# def assignIps(out):
#     import networkx as nx
#     g = nx.Graph()
#     g.add_nodes_from([n['name'] for n in out['hosts'] + out['switches']])
#     g.add_edges_from([e['hosts'] for e in out['links']])
#     source = out['hosts'][0]['name']
#     for host in out['hosts'][1:]:
#         d = len(nx.shortest_path(g, source = source, target = host['name']))
#         host['options']['ip'] = int2ip(ipBase + d)

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

args = parser.parse_args()

dep = args.depth
fan = args.fanout
leafs = args.leafs if args.leafs >= 0 else fan
swLeafs = args.sw_leafs
pdep = args.pref_depth

nhosts = fan ** (dep - 1) * leafs + swLeafs * fan ** (dep)
availableIps = []

a = dep-pdep
prefixes = {}

import itertools
comb = itertools.product((i for i in range(fan)), repeat = a)
for i, t in enumerate(comb):
    prefixes[t] = i+1

# def setIps(availableIps):
# ip1 = 0
#     ip2 = 0
#     ip3 = 0
#     while ip1 <= maxIpPart:
#         while ip2 <= maxIpPart:
#             while ip3 <= maxIpPart:
#                 if len(availableIps) >= nhosts:
#                     return
#                 ip3 += 1
#                 availableIps.append([ip1, ip2, ip3])
#                 p = len(availableIps) % leafs
#                 if p == 0:
#                     ip3 += leafs
#             ip2 += 1
#             # if 0 == len(availableIps) % fan:
#             # ip2 += fan
#             ip3 = 0
#         ip1 += 1
#         # if len(availableIps) % fan == 0:
#         # ip1 += fan
#         ip2 = 0


# setIps(availableIps)
# # print availableIps
# for ip in availableIps:
#     print ip, ip2int(ip)
#     print "  ", [abs(ip2int(ip) - ip2int(ip2)) / fan + 1 for ip2 in availableIps]

addTree(dep, fan, [])

import sys
print >>sys.stderr, nhosts
# assignIps(out)
import json
print(json.dumps(out, sort_keys = True, indent = 4, separators = (',', ': ')))
