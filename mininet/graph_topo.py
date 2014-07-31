# import networkx as nx
# from graphs import Graph
import pygraphviz as pg

# def buildGraph(topo):
#     g = nx.DiGraph()
#     g.add_nodes_from(n['name'] for n in topo['hosts'] + topo['switches'])
#     for e in topo['links']:
#         g.add_edge(*e['hosts'], key = e['name'])#, name = e['name'])
#     # g.add_edges_from(e['hosts'] for e in topo['links'] if e['name'] not in tLinks)
#     return g


def buildGraph(topo):
    g = pg.AGraph()
    g.add_nodes_from(n['name'] for n in topo['hosts'] + topo['switches'])
    for e in topo['links']:
        g.add_edge(*e['hosts'], label = e['name'] if e.has_key('name') and not args.no_link_names else "")#, name = e['name'])
    # g.add_edges_from(e['hosts'] for e in topo['links'] if e['name'] not in tLinks)
    return g


def graphTopo(graph, output, prog, options):
    from matplotlib.backends.backend_pdf import PdfPages

    graph.layout(prog = prog)
    graph.graph_attr.update(**options)
    graph.draw(output)

if __name__ == "__main__":
    from argparse import ArgumentParser
    import json
    parser = ArgumentParser()
    parser.add_argument('--topo',
                        dest = 'topoFile')
    parser.add_argument('--output',
                        dest = 'outFile')

    parser.add_argument('--prog',
                        choices = ['twopi', 'gvcolor', 'wc', 'ccomps', 'tred', 'sccmap', 'fdp', 'circo', 'neato', 'acyclic', 'nop', 'gvpr', 'dot', 'sfdp'],
                        dest = 'prog',
                        default = 'dot')

    parser.add_argument('--options',
                        dest = 'options',
                        action = 'append',
                        default = ["ranksep='0.3'", "nodesep='30'", "fontsize='70'"])
    parser.add_argument('--no-link-names',
                        dest = 'no_link_names',
                        action = 'store_true',
                        default = False)

    args = parser.parse_args()
    options = {}
    for o in args.options:
        key, sep, value = o.partition('=')
        options[key] = value
    graph = buildGraph(json.load(open(args.topoFile)))
    graphTopo(graph, args.outFile, args.prog, options)
