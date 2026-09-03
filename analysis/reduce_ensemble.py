#!/usr/bin/env python3
"""Reduce a twin ensemble pickle to per-run artefacts that fit in laptop memory.

The kappa=0.55 headline ensemble is 1.77 GB on disk and unpickles to well over
this machine's RAM: each row carries ~61 dense snapshots, each holding a full
networkx copy of the graph, plus a 400k-element `system_C_trajectory` whose
elements are np.float64 (every one of which the pickler memoised, three objects
apiece -- ~60M memo entries, which is what OOM-kills a naive read_pickle).

Two measures make the load fit:

  * a list-backed memo (8 B/slot instead of ~100 B for a dict) that evicts only
    the numpy-scalar reduce arguments -- a (dtype, 8-byte payload) tuple and the
    payload itself. Those are freshly built per scalar and never re-referenced,
    so eviction cannot corrupt anything. Everything else is retained, so shared
    objects (memory-buffer tuples in the event log, interned diet strings, the
    `initial_diets` list that snapshots[0] aliases) survive intact.
  * networkx graphs are reconstructed as an edge array plus the node `theta`
    vector rather than as graphs.

Output: model_output/<stem>_reduced/run_XX.pkl, one dict per run, holding the
fields the downstream analyses actually read. Graph edges are kept on a coarser
grid than the other snapshot fields (EDGE_EVERY), since only the analysis-time
graph is used for the two-DV predictors.

Usage:
    python reduce_ensemble.py <ensemble.pkl> [--edge-every 10000]
"""
import sys, os, gc, pickle, time
import numpy as np

EDGE_EVERY = 10000          # keep graph edges at multiples of this step
KEEP_EDGES_ALWAYS = (0,)    # plus these keys, whatever the grid
TRAJ_MIN = 100000           # a list this long is a per-step trajectory

_TRAJ = {}                  # id(list) -> [[float32 chunks], memo watermark]


# --- unpickler -------------------------------------------------------------

class _GraphStub:
    """Stands in for nx.Graph/DiGraph: keeps edges and node theta, drops the
    dict-of-dicts adjacency that makes the ensemble unloadable."""
    __slots__ = ('edges', 'theta', 'n')

    def __setstate__(self, state):
        adj = state.get('_adj', {})
        node = state.get('_node', {})
        src, dst = [], []
        for u, nbrs in adj.items():
            for v in nbrs:
                if u < v:
                    src.append(u); dst.append(v)
        self.edges = np.array([src, dst], dtype=np.int32).T
        self.n = len(node)
        self.theta = np.array([node[k].get('theta', np.nan) for k in sorted(node)],
                              dtype=np.float32)


_DROPPED = object()


class _LeakyMemo(list):
    """Sequential memo with numpy-scalar reduce arguments evicted: the 8-byte
    payload and the (dtype, payload) tuple, both built fresh per scalar. A
    1-byte literal such as _reconstruct's b'b' IS shared, hence the exact
    length test; any misjudgement here surfaces as a sentinel GET below rather
    than as silently corrupt data.

    Also records where each graph's reconstruction begins, so its state can be
    evicted wholesale once the stub has taken what it needs (see load_build):
    dropping the graph object is not enough on its own, because the memo holds
    every node dict, adjacency dict and edge attribute dict the state was built
    from -- ~20k dicts per graph, 3050 graphs, and they are what actually
    exhausts memory."""
    graph_mark = -1

    def __setitem__(self, k, v):
        if k == len(self):
            self.append(None)
        if (type(v) is bytes and len(v) == 8) or (
                type(v) is tuple and len(v) == 2 and isinstance(v[0], np.dtype)):
            list.__setitem__(self, k, _DROPPED)
        else:
            if type(v) is _GraphStub:
                self.graph_mark = k
            list.__setitem__(self, k, v)


class _Reducer(pickle._Unpickler):
    dispatch = dict(pickle._Unpickler.dispatch)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.memo = _LeakyMemo()
        self._traj = _TRAJ

    def find_class(self, module, name):
        if module.startswith('networkx'):
            return _GraphStub
        return super().find_class(module, name)

    def _get(self, i):
        v = self.memo[i]
        if v is _DROPPED:
            raise SystemExit(f"ERROR: memo slot {i} was evicted but is "
                             f"re-referenced -- eviction rule is unsafe")
        self.append(v)

    def load_binget(self):
        self._get(self.read(1)[0])
    dispatch[pickle.BINGET[0]] = load_binget

    def load_long_binget(self):
        self._get(int.from_bytes(self.read(4), 'little'))
    dispatch[pickle.LONG_BINGET[0]] = load_long_binget

    def load_get(self):
        self._get(int(self.readline()[:-1]))
    dispatch[pickle.GET[0]] = load_get

    # -- compress as we go ---------------------------------------------------
    # Peak memory is reached with every column simultaneously live, so nothing
    # can be freed after the load; the objects have to be shrunk at the moment
    # the stream finishes building them.

    def load_appends(self):
        """Drain trajectory lists into float32 buffers batch by batch.

        A trajectory is 400k Python floats (fraction_veg) or np.float64
        (system_C) -- ~1.5 GB across 50 runs, plus the memo slots for the
        numpy scalars. Once a list is recognised, each APPENDS batch is
        converted and the list emptied, and the memo slots created since the
        previous batch (this batch's scalars and their reduce args, referenced
        nowhere else) are evicted."""
        items = self.pop_mark()
        lst = self.stack[-1]
        key = id(lst)
        buf = self._traj.get(key)
        if buf is not None:
            buf[0].append(np.asarray(items, dtype=np.float32))
            self._evict_since(buf[1])
            buf[1] = len(self.memo)
            return
        lst.extend(items)
        if len(lst) >= TRAJ_MIN and isinstance(lst[0], (float, np.floating)):
            self._traj[key] = [[np.asarray(lst, dtype=np.float32)], len(self.memo)]
            del lst[:]
    dispatch[pickle.APPENDS[0]] = load_appends

    def _evict_since(self, start):
        for i in range(start, len(self.memo)):
            if isinstance(self.memo[i], np.floating):
                list.__setitem__(self.memo, i, _DROPPED)

    def load_build(self):
        """After a graph stub has absorbed its state, evict the container shells
        the state was built from -- the adjacency dicts, node dicts and edge
        attribute dicts, which are per-graph and never referenced again.

        Only containers. G.copy() makes new attribute dicts but shares their
        values, so a node's theta object is reached by BINGET from all 61
        snapshot graphs of the same run; evicting scalars breaks the very next
        graph, which is how the guard below caught it."""
        state = self.stack[-1]
        inst = self.stack[-2]
        is_graph = type(inst) is _GraphStub
        mark = self.memo.graph_mark
        pickle._Unpickler.load_build(self)
        if is_graph and mark >= 0:
            memo = self.memo
            for i in range(mark + 1, len(memo)):
                if type(memo[i]) in (dict, list, set):
                    list.__setitem__(memo, i, _DROPPED)
            memo.graph_mark = -1
        del state, inst
    dispatch[pickle.BUILD[0]] = load_build

    def load_setitems(self):
        """Compress a snapshot dict the moment it is complete.

        Each snapshot holds twelve 2000-element Python lists; at ~61 snapshots
        x 50 runs that is over 2 GB of boxed scalars. Converting to numpy here
        frees all but the arrays."""
        items = self.pop_mark()
        d = self.stack[-1]
        for i in range(0, len(items), 2):
            d[items[i]] = items[i + 1]
        if 'reductions' in d and 'graph' in d and 'diets' in d:
            _compress_snapshot(d)
        elif d and all(isinstance(v, dict) and 'reductions' in v for v in d.values()):
            _thin_graphs(d)
    dispatch[pickle.SETITEMS[0]] = load_setitems


def _progress(fobj, total, stop):
    import threading
    def run():
        while not stop.is_set():
            print(f"INFO: unpickle {fobj.tell()/1e9:.2f} / {total/1e9:.2f} GB", flush=True)
            stop.wait(30)
    t = threading.Thread(target=run, daemon=True); t.start()
    return t


def load(path):
    import threading
    total = os.path.getsize(path)
    stop = threading.Event()
    t0 = time.time()
    with open(path, 'rb') as f:
        _progress(f, total, stop)
        df = _Reducer(f).load()
    stop.set()
    print(f"INFO: loaded {len(df)} rows in {time.time()-t0:.0f} s", flush=True)
    return df


# --- reduction -------------------------------------------------------------

def _f32(x):
    return np.asarray(x, dtype=np.float32)


def _f64(x):
    """Per-agent quantities stay float64: attribution_ledger.validate compares a
    replay against the in-run ledger at atol=0."""
    return np.asarray(x, dtype=np.float64)


def _i32(seq):
    return np.array([-1 if v is None else v for v in seq], dtype=np.int32)


def _diet_bits(seq):
    return np.array([d == 'veg' for d in seq], dtype=np.uint8)


def _compress_snapshot(d):
    """In-place, at SETITEMS time. Per-agent floats stay float64: the ledger
    replay is validated against the in-run credit at atol=0.

    Each source list is emptied after conversion, not merely replaced in the
    dict: the memo holds a reference to every one of them, so a replaced list
    stays alive to the end of the load and the compression buys nothing. The
    exception is `diets`, which at t=0 IS the `initial_diets` column -- the same
    list object, reached later by BINGET -- and must not be emptied. It is 2000
    interned pointers, so it costs nothing to leave."""
    d['veg_fraction'] = float(d['veg_fraction'])
    d['diets'] = _diet_bits(d['diets'])
    for key, conv in (('reductions', _f64), ('reductions_unw', _f64),
                      ('change_times', _i32), ('alphas', _f64), ('rhos', _f64),
                      ('influence_parents', _i32),
                      ('direct_conversions', lambda v: np.asarray(v, dtype=np.int32)),
                      ('immune', lambda v: np.asarray(v, dtype=np.uint8))):
        src = d[key]
        d[key] = conv(src)
        del src[:]
    g = d.pop('graph')
    d['edges'], d['node_theta'], d['n_nodes'] = g.edges, g.theta, g.n


def _thin_graphs(snaps):
    """Drop the edge list from snapshots off the EDGE_EVERY grid: only the
    analysis-time graph is used, and the two-DV predictors need it there."""
    for t, snap in snaps.items():
        keep = (t in KEEP_EDGES_ALWAYS or not isinstance(t, int)
                or t % EDGE_EVERY == 0)
        if not keep:
            snap.pop('edges', None)
            snap.pop('node_theta', None)
            snap.pop('n_nodes', None)


def _traj_of(lst):
    buf = _TRAJ.get(id(lst))
    if buf is None:                       # short enough to survive as a list
        return _f32(lst)
    return np.concatenate(buf[0])


def reduce_row(row):
    return {
        'run': int(row['run']),
        'params': dict(row['params']),
        'agent_ini': row['agent_ini'],
        'initial_veg_f': float(row['initial_veg_f']),
        'final_veg_f': float(row['final_veg_f']),
        'fraction_veg': _traj_of(row['fraction_veg_trajectory']),
        'system_C': _traj_of(row['system_C_trajectory']),
        'events': row['events'],
        'initial_diets': list(row['initial_diets']),
        'individual_reductions': _f64(row['individual_reductions']),
        'individual_reductions_unw': _f64(row['individual_reductions_unw']),
        'snapshots': row['snapshots'],
    }


def main():
    path = sys.argv[1]
    global EDGE_EVERY
    if '--edge-every' in sys.argv:
        EDGE_EVERY = int(sys.argv[sys.argv.index('--edge-every') + 1])
    outdir = os.path.splitext(path)[0] + '_reduced'
    os.makedirs(outdir, exist_ok=True)

    df = load(path)
    print(f"INFO: columns {list(df.columns)}", flush=True)

    for i in range(len(df)):
        row = df.iloc[i]
        red = reduce_row(row)
        # integrity: an evicted memo slot would surface as the sentinel
        for k in ('params', 'events', 'initial_diets'):
            if red[k] is _DROPPED or (k == 'params' and 'kappa' not in red[k]):
                raise SystemExit(f"ERROR: run {i} field {k} corrupt")
        # an in-place clear of an aliased list would surface as a short array
        n = red['params']['N']
        for k in ('initial_diets', 'individual_reductions', 'individual_reductions_unw'):
            if len(red[k]) != n:
                raise SystemExit(f"ERROR: run {i} {k} has {len(red[k])} entries, expected {n}")
        for t, sn in red['snapshots'].items():
            for k in ('diets', 'reductions', 'alphas', 'rhos', 'immune'):
                if len(sn[k]) != n:
                    raise SystemExit(f"ERROR: run {i} snapshot {t} {k} has "
                                     f"{len(sn[k])} entries, expected {n}")
        dest = os.path.join(outdir, f"run_{red['run']:02d}.pkl")
        with open(dest, 'wb') as f:
            pickle.dump(red, f, protocol=4)
        print(f"INFO: wrote {dest}  kappa={red['params']['kappa']} "
              f"steps={red['params']['steps']} tau={red['params']['tau_persistence']} "
              f"snaps={len(red['snapshots'])} events={len(red['events'])} "
              f"F_end={red['final_veg_f']:.4f}", flush=True)
        df.iat[i, df.columns.get_loc('snapshots')] = None
        df.iat[i, df.columns.get_loc('events')] = None
        del row, red
        gc.collect()
    print("INFO: done", flush=True)


if __name__ == '__main__':
    main()
