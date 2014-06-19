import time
import os
import shlex

from events import EventsManager
import mininet.log as lg


def wait_start_events(watcher_start_event = None):
    if watcher_start_event is not None:
        wait_file(watcher_start_event)
        EventsManager.startTimers()
        time.sleep(5)


def wait_file(file):
    lg.output("Waiting for signal on %s to start events\n" % file)
    while not os.path.exists(file):
        time.sleep(10)
    os.remove(file)


def post_events(net, netprobes, watcher_post_event = None):
    if watcher_post_event is not None:
        lg.output("Executing post event actions '%s'\n" % watcher_post_event)
        import subprocess

        cmd = watcher_post_event.split(" ")
        rcmd = []
        # replace name with ips
        for c in cmd:
            if netprobes.has_key(c):
                rcmd.append(net.nameToNode[c].IP())
            else:
                rcmd.append(c)
        lg.output("Running post command (%s)\n" % " ".join(rcmd))
        p = subprocess.Popen(shlex.split(" ".join(rcmd)))
        p.communicate()


def wait_process(process):
    lg.output("Waiting for process %s to terminate\n" % process.pid)
    # while process is running
    while process.poll() is None:
        time.sleep(10)


def make_watcher_results(watcher_log, watcher_output, topoFile):
    import datetime

    now = datetime.datetime.now()
    if watcher_log is not None:
        if os.path.exists(watcher_log):
            import shutil

            shutil.copyfile(watcher_log, 'watchers/logs/%s.log' % now)
        else:
            lg.error("No log file was found in %s\n" % watcher_log)
    if watcher_output is not None:
        if os.path.exists(watcher_output):
            import watcher_delay

            watcher_delay.appendResults(watcher_delay.makeResults(watcher_output, topoFile))
            # prevent results from being processed twice
            os.rename(watcher_output, 'watchers/output/%s.json' % now)
        else:
            lg.error('No file watcher output was found in %s\n' % watcher_output)
