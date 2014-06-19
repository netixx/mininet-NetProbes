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
        g.add_edge(*e['hosts'], label = e['name'])#, name = e['name'])
    # g.add_edges_from(e['hosts'] for e in topo['links'] if e['name'] not in tLinks)
    return g


def graphTopo(graph, output):
    from matplotlib.backends.backend_pdf import PdfPages

    # pdf = PdfPages(output)
    graph.layout(prog = 'dot')
    graph.graph_attr.update(ranksep = '0.1', nodesep = '20', fontsize = '70')
    graph.draw(output)
    # nx.draw(graph, pos = nx.graphviz_layout(graph, prog = 'neato'))
    # # pdf.draw()
    # fig = Graph.gcf()
    # fig.set_size_inches(20, 20)
    # pdf.savefig(bbox_inches = 'tight')  # 'checks/delay.pdf', format = 'pdf', )
    # Graph.close()
    # pos = nx.layout.spring_layout(graph)
    # pos = nx.graphviz_layout(graph, prog = 'dot', args='-Gnodesep=20 -Granksep=10 -Eminlen=5')#, args = '-repulsiveforce="1.0"')
    # # pos = nx.graphviz_layout(graph, prog = 'sfdp', args = '-NK=2.0')
    # nx.draw(graph, pos, font_size = 7, arrows = False)
    # edge_labels = dict([((u, v,), d['key'])
    #                     for u, v, d in graph.edges(data = True)])
    #
    #
    # nx.draw_networkx_edge_labels(graph, pos = pos,edge_labels = edge_labels, font_size = 6)
    # # pdf.draw()
    # fig = Graph.gcf()
    # # fig.set_size_inches(30, 15)
    # pdf.savefig(bbox_inches = 'tight')  # 'checks/delay.pdf', format = 'pdf', )
    # Graph.close()

    # import datetime
    #
    # d = pdf.infodict()
    # d['Title'] = 'Delays measurement'
    # d['Author'] = u'Francois Espinet'
    # d['Subject'] = 'Delay measurement'
    # d['Keywords'] = 'measurement delays'
    # d['ModDate'] = datetime.datetime.today()
    # pdf.close()

if __name__ == "__main__":
    from argparse import ArgumentParser
    import json
    parser = ArgumentParser()
    parser.add_argument('--topo',
                        dest = 'topoFile')
    parser.add_argument('--output',
                        dest = 'outFile')

    args = parser.parse_args()
    graph = buildGraph(json.load(open(args.topoFile)))
    graphTopo(graph, args.outFile)
