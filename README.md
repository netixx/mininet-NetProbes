mininet-NetProbes : NetProbes integration into Mininet network emulator
=================

For more on Mininet : [http://mininet.org](http://mininet.org)

*This README is based on the assumption that Mininet is installed and working correctly.*

You can find NetProbes at [https://github.com/netixx/NetProbes](https://github.com/netixx/NetProbes).

**The aim of this project is to provide some level of integration between Mininet and NetProbes**, but although the given code is geared towards NetProbes,
it is easily possible to execute another program of your choosing (see start-probe.sh script)


The code provided here has been tested on ubuntu 10.04 (using bash as interpretor), it should run on other linux distributions with minor adaptations. Root access is required to run most of the scripts.

General directory structure
-----------------


    +-- generators              **contains scripts to generate network topologies, see script for options
    +-- mininet                 **contains code to customize Mininet and control NetProbes
    |   +-- builder.py          **main program for launching emualtions
    +-- start.sh                **shortcut (with pythonpath resolution) to builder.py
    +-- start-probe.sh          **script to launch NetProbes (assumes NetProbes root to be at $HOME/netprobes)
    +-- examples                **contains script to launch examples simulations
    |   +-- benchmarks          **scripts and topology to benchmarks measurement tools
    |   +-- trees               **script and topologies to make watcher simulations
    +-- topos                   **contains json encoded topology description
    +-- install.sh              **ALPHA: script to install libraries required to do test
    +-- lib                     **directory where libraries are placed (created if needed)


Network specification format (json style, UTF-8 encoding):
-----------------

Network topology is specified in a json format containing a main dictionnary with 3 types of keys : hosts, links and switches.This allows definition of any topologies by specifying nodes and edges of the graph.

It is possible to add options in order to use Mininet functionnalities 
such as bandwidth or delay restriction on edges. Options on the hosts are mostly NetProbes/program related options.

    Hosts:
        * name                  **required**    *name of the host (use for reference in configuration)*
        * options
            * ip                optional        *"10.0.0.2"*
            * command           optional        *command to launch with host*
            * commandOpts       optional        *options to the command*
            * isXHost           optional        *should we open a window for visualisation*
     
    Links:
        * hosts                             **required**    *array of connected hosts (e.g ["h1", "h2"])*
        * options                                           *dict of link options*
            * bw                            optional        *10 < bw < 1000 in MB*
            * delay                         optional        *e.g. "100ms"*
            * use_hfsc,use_tbf,use_htb      optionnal       *bucket to use (relavant with bw option)
    
    
    Switches:
        * name      **required**    *name of the switch*


Starting pointers
-----------------

Example topologies can be found in topos folder.

To start an emulation scenarios (e.g. flat.json):

    $ start.sh --topo topos/flat.json

   
To benchmark bandwidth and delay measurement tools on Mininet

    $ cd examples/benchmarks/
    # ./run.sh 


More advanced scenarios are available in the examples directory. They contain example of usage of NetProbes watchers to detect changes in link properties (bandwidth and delay). Multiple topologies (trees of different fanout and depth) as well as bandwidth and delay metrics are available.


To launch a scenario (e.g. delay variation, 512 hosts, depth 9, fanout 2):

    $ cd examples/trees/delay/512
    # bash run.sh 9x2-operator


To produce associatede graphs (watcher-set.pdf and watcher-link.pdf):

    $ cd -
    $ python mininet/watcher_delay.py --output watcher/watchers.json --graphs watcher/watchers.pdf




Remarks and future work
-------------------

There is early code snippets to support routers instead of switch, but this functionnality is not implemented yet.
