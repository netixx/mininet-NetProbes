import time
import os
import shlex
import traceback

from events import EventsManager
import mininet.log as lg


def wait_start_events(watcher_start_event = None):
    if watcher_start_event is not None:
        wait_file(watcher_start_event)
        EventsManager.startTimers()
        time.sleep(2)


def wait_file(file):
    lg.output("Waiting for signal on %s to proceed\n" % file)
    while not os.path.exists(file):
        time.sleep(5)
    os.remove(file)


def wait_reset(watcher_reset_event = None):
    if watcher_reset_event is not None:
        wait_file(watcher_reset_event)
        EventsManager.stopEvents()
        time.sleep(2)


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


def pre_events(net, netprobes, watcher_pre_event = None):
    if watcher_pre_event is not None:
        lg.output("Executing pre event actions '%s'\n" % watcher_pre_event)
        import subprocess

        cmd = watcher_pre_event.split(" ")
        rcmd = []
        # replace name with ips
        for c in cmd:
            if netprobes.has_key(c):
                rcmd.append(net.nameToNode[c].IP())
            else:
                rcmd.append(c)
        lg.output("Running pre command (%s)\n" % " ".join(rcmd))
        p = subprocess.Popen(shlex.split(" ".join(rcmd)))
        p.communicate()
        p.wait()
        if p.returncode != 0:
            time.sleep(5)
            lg.error("Command returned %s\n" % p.returncode)
            p = subprocess.Popen(shlex.split(" ".join(rcmd)))
            p.communicate()
            p.wait()


def wait_process(process):
    lg.output("Waiting for process %s to terminate\n" % process.pid)
    # while process is running
    while process.poll() is None:
        time.sleep(5)


def make_watcher_results(watcher_log, watcher_output, topoFile, simParams, watcher_type = "delay"):
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
            if watcher_type == "delay":
                import watcher_delay as wa
            elif watcher_type == "bw":
                import watcher_bw as wa
            else:
                lg.error("Unknown watcher type passed : %s\n" % watcher_type)
                return
            try:
                wa.appendResults(wa.makeResults(watcher_output, topoFile, simParams))
            except Exception as e:
                lg.error("Error while processing results %s\n" % e)
                lg.error(traceback.format_exc())
            # prevent results from being processed twice
            os.rename(watcher_output, 'watchers/output/%s.json' % now)
        else:
            lg.error('No file watcher output was found in %s\n' % watcher_output)
