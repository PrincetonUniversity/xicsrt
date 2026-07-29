# -*- coding: utf-8 -*-
# This file includes AI generated code using Claude (Opus 5).
# ------------------------------------------------------------------------------
"""
Authors:
  | Novimir Antoniuk Pablant <npablant@pppl.gov>

Purpose:
  Create a simple profiler module.

Description:
  This module is meant to enable manual profiling with very low overhead.

  The results are held in a module level global, which means they are
  per-process. When raytracing runs are dispatched to a `multiprocessing`
  pool the worker timings are therefore invisible to the parent unless they
  are explicitly returned and merged; see `getResults` and `merge`.

"""
# ------------------------------------------------------------------------------


import os
import datetime

import logging

log = logging.getLogger(__name__)
profiler_results = dict()
flags = {}
flags['enabled'] = False

def isEnabled():
    return flags['enabled']

def startProfiler(reset=False):
    if reset:
        resetProfiler()
        
    flags['enabled'] = True

def stopProfiler():
    flags['enabled'] = False

    # Reset the start time of every entry.
    # It doesn't make sense to keep timing active when not using the profiler.
    for key in profiler_results:
        profiler_results[key]['time_start'] = None

def resetProfiler():
    global profiler_results
    profiler_results = dict()

def report(flush=True):
    names = list(profiler_results.keys())
    totals = [profiler_results[xx]['time_total'] for xx in profiler_results]
    indexes = sorted(range(len(totals)), key=totals.__getitem__, reverse=True)

    if flush:
        print('', flush=True, end='')

    log.info('{:25.25s} {:14.14s}  {:14.14s}  {:6.6s}'.format(
        'name'
        ,'total'
        ,'single'
        ,'calls'))
    
    for ii in indexes:
        name = names[ii]
        if profiler_results[name]['num_calls'] > 0:
            log.info('{:25.25s} {}  {}  {:6d}'.format(
                name
                ,profiler_results[name]['time_total']
                ,profiler_results[name]['time_total']/profiler_results[name]['num_calls']
                ,profiler_results[name]['num_calls']))

def getResults():
    """
    Return a picklable copy of the profiler results for this process.

    This is intended for transferring timings out of a `multiprocessing`
    worker so that the parent can `merge` them. The `time_start` entries are
    dropped: an in-progress timer has no meaning once it leaves its process.

    This function was AI generated using Claude (Opus 5).
    """
    results = dict()
    for name in profiler_results:
        results[name] = {
            'time_total': profiler_results[name]['time_total'],
            'num_calls': profiler_results[name]['num_calls'],
            }
    return results


def merge(results, prefix=None):
    """
    Merge an external set of profiler results into this process's results.

    Totals and call counts are summed, so merging the results from several
    workers gives the aggregate time spent across all of them (which will
    exceed the wall-clock time of the parent).

    Parameters
    ----------
    results : dict
      A results dictionary as returned by `getResults`.

    prefix : str (None)
      If given, prepended to every name. Use this to keep worker timings
      visually distinct from the parent's own, e.g. `prefix='worker: '`.

    This function was AI generated using Claude (Opus 5).
    """
    if not results:
        return

    for name in results:
        key = name if prefix is None else prefix + name
        if key not in profiler_results:
            _newProfile(key)
        profiler_results[key]['time_total'] += results[name]['time_total']
        profiler_results[key]['num_calls'] += results[name]['num_calls']


def getTimeTotal(name):
    return profiler_results[name]['time_total']

def getTimeSingle(name):
    return profiler_results[name]['time_total']/profiler_results[name]['num_calls']

def start(name):
    if flags['enabled']:
        if not name in profiler_results:
            _newProfile(name)
            
        profiler_results[name]['time_start'] = datetime.datetime.now()

def stop(name):
    if flags['enabled']:
        if profiler_results[name]['time_start'] is not None:
            profiler_results[name]['time_total'] += datetime.datetime.now() - profiler_results[name]['time_start']
            profiler_results[name]['num_calls'] += 1
            profiler_results[name]['time_start'] = None

def _newProfile(name):
    profiler_results[name] = {
        'time_total':datetime.timedelta(0)
        ,'time_start':None
        ,'num_calls':0
        }

    
