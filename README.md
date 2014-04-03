mininet-NetProbes
=================

NetProbe testbed using mininet emulation

Network specification format (json style, UTF-8 encoding):

Hosts:

* name
 
Links:

* hosts : array of connected hosts
* (optional) options : dict of link options
	* (optional) bw : 10 < bw < 1000 in MB
	* (optional) delay : 100ms

Switches:

* name