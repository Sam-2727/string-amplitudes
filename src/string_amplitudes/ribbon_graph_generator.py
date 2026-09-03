"""Generate trivalent Strebel ribbon graph topologies and boundary data.
Here, "boundary data" means the ordering of edges when representing the
ribbon graph as sewn together discs.

Currently, multiple faces are supported but only the implemention of one
face (i.e., one vertex operator) has been tested thoroughly.

The public functions are:

* :func:`generate_ribbon_graphs`: given a genus and number of punctures,
  return the corresponding non-isomorphic ribbon graph topologies.
* :func:`sample_ribbon_graph`: heuristically sample one ribbon graph when generating
  all topologies is impractical and/or unnecessary.
* :func:`get_boundary_data`: from the output of :func:`generate_ribbon_graphs`,
  give the sewing data on the face boundaries of a given graph.
"""

from itertools import permutations, product as iproduct
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from numbers import Integral
import random

try:
    import networkx as nx
    from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
except ImportError:  # pragma: no cover
    nx = None
    weisfeiler_lehman_graph_hash = None


# ============================================================
# Base cubic graphs
# ============================================================

_CUBIC_GRAPH_CACHE = {}
_CUBIC_MULTIGRAPH_CACHE = {}
_DEFAULT_EXACT_MULTIGRAPH_VERTEX_LIMIT = 10


def _generate_connected_cubic_graphs(
    nv,
    target_count=None,
    max_seeds=None,
    strict_isomorphism=False,
):
    """Generate connected non-isomorphic simple cubic graphs on nv vertices.

    If target_count is provided, generation stops once that many graphs are found.
    Otherwise a heuristic is used to determine when all ribbon graphs are generated,
    so completeness is not guaranteed.
    With strict_isomorphism=False, WL hash dedup is used for speed.

    Parameters
    ----------
    nv : int
        Number of vertices. The generated graphs are simple, connected, and
        trivalent.
    target_count : int or None, optional
        Stop after finding this many non-isomorphic candidates. If ``None``,
        stop after the search has found no new candidate for the configured
        stagnation interval.
    max_seeds : int or None, optional
        Maximum number of deterministic NetworkX random-graph seeds to try.
        If ``None``, the function selects a bound when ``target_count`` is set.
    strict_isomorphism : bool, optional
        If ``True``, confirm equivalence within each Weisfeiler--Lehman hash
        bucket using graph isomorphism. If ``False``, treat each hash bucket as
        one equivalence class.

    Returns
    -------
    list of tuple
        A list of ``(edges, vertices)`` pairs. ``edges`` is a list of
        ``(vertex_a, vertex_b)`` endpoint pairs, and ``vertices`` contains the
        vertex labels.
    """
    if nv in _CUBIC_GRAPH_CACHE:
        return _CUBIC_GRAPH_CACHE[nv]

    if nx is None or weisfeiler_lehman_graph_hash is None:
        raise RuntimeError(
            "networkx is required for graph generation"
        )

    reps_by_hash = {}
    reps = []
    seed = 0
    stagnation = 0
    # For unknown sizes, stop after many seeds with no new isomorphism class.
    stagnation_limit = max(2000, 200 * nv)

    if target_count is not None and max_seeds is None:
        max_seeds = max(50000, 5000 * nv)

    while True:
        if target_count is not None and len(reps) >= target_count:
            break
        if target_count is None and stagnation >= stagnation_limit:
            break
        if max_seeds is not None and seed >= max_seeds:
            break

        g = nx.random_regular_graph(3, nv, seed=seed)
        seed += 1
        h = weisfeiler_lehman_graph_hash(g, iterations=5)
        bucket = reps_by_hash.setdefault(h, [])
        if strict_isomorphism:
            if any(nx.is_isomorphic(g, r) for r in bucket):
                stagnation += 1
                continue
            bucket.append(g)
            reps.append(g)
            stagnation = 0
        else:
            if bucket:
                stagnation += 1
                continue
            bucket.append(g)
            reps.append(g)
            stagnation = 0

    base_graphs = []
    verts = list(range(nv))
    for g in reps:
        edges = []
        for a, b in sorted(g.edges()):
            edges.append((int(a), int(b)))
        base_graphs.append((edges, verts))

    # If fast hash-only mode failed to reach the requested target, retry strictly.
    if target_count is not None and len(base_graphs) < target_count and not strict_isomorphism:
        return _generate_connected_cubic_graphs(
            nv,
            target_count=target_count,
            max_seeds=max_seeds,
            strict_isomorphism=True,
        )

    _CUBIC_GRAPH_CACHE[nv] = base_graphs
    return _CUBIC_GRAPH_CACHE[nv]


def _edge_multiplicity_counts(edges):
    """Helper function that takes in an list of graph edges and 
      returns dictionary counting how many edges connect each 
      unordered pair of vertices. Called in
      :func:`_weighted_graph_from_edges`.

    Parameters
    ----------
    edges : iterable of tuple of int
        Undirected edges represented by endpoint pairs ``(vertex_a, vertex_b)``.
        Parallel edges occur as repeated endpoint pairs.

    Returns
    -------
    collections.defaultdict
        Mapping from each canonically ordered endpoint pair to the number of
        corresponding edges.
    """
    counts = defaultdict(int)
    for a, b in edges:
        p = (a, b) if a < b else (b, a)
        counts[p] += 1
    return counts


def _weighted_graph_from_edges(edges, verts):
    """Helper function that takes list of edges and vertices and returns 
       a networkx.Graph object.

    Parameters
    ----------
    edges : iterable of tuple of int
        Undirected edges represented by endpoint pairs. Repeated pairs encode
        parallel edges.
    verts : iterable of int
        Vertex labels to include, including any isolated vertices if present.

    Returns
    -------
    networkx.Graph
        A simple weighted graph with one edge for each unordered endpoint pair.
        The edge attribute ``mult`` stores the original edge multiplicity.
    """
    g = nx.Graph()
    g.add_nodes_from(verts)
    for (a, b), mult in _edge_multiplicity_counts(edges).items():
        g.add_edge(a, b, mult=mult)
    return g


def _canonicalize_multigraph_suffix(adj, deg, start):
    """Helper function to sort vertices not yet sorted into the adjacency matrix
        into a canonical order to efficiently construct the adjacency matrix for 
        permutation-equivalent entries.

    Parameters
    ----------
    adj : list of list of int
        Partially constructed symmetric adjacency matrix. Its entries record
        edge multiplicities between vertices in the current labelling.
    deg : list of int
        Remaining degree of each vertex in the same current labelling.
    start : int
        Index of the first unprocessed vertex (i.e., vertex for which
        the adjacency matrix entrices have not yet been constructed). 
        Vertices with smaller indices form the processed prefix and 
        are not reordered.

    Returns
    -------
    tuple
        ``(new_adj, new_deg)``, where the unprocessed rows and columns of
        ``new_adj`` and the corresponding entries of ``new_deg`` have been
        permuted into the canonical order. The inputs are not modified.
    """
    n = len(deg)
    prefix = list(range(start))
    suffix = list(range(start, n))
    suffix.sort(
        key=lambda j: (deg[j], tuple(adj[k][j] for k in range(start))),
        reverse=True,
    )
    perm = prefix + suffix
    new_adj = [[adj[perm[a]][perm[b]] for b in range(n)] for a in range(n)]
    new_deg = [deg[p] for p in perm]
    return new_adj, new_deg


def _candidate_loopless_cubic_multigraph_matrices(nv):
    """Helper function to generate all (partially symmetry inequivalent) candidate 
    adjacency matrices for a given set of vertices.

    Parameters
    ----------
    nv : int
        Number of vertices in each candidate cubic multigraph. Valid cubic
        graphs require an even number of vertices.

    Returns
    -------
    list of tuple of tuple of int
        Symmetry-reduced candidate adjacency matrices. Each matrix is symmetric,
        has zero diagonal, and has row sum three. Candidates are not necessarily
        connected or pairwise non-isomorphic at this stage.
    """
    if nv == 2:
        return [((0, 3), (3, 0))]

    adj = [[0] * nv for _ in range(nv)]
    deg = [3] * nv
    candidates = []
    seen_states = set()

    def rec(i, cur_adj, cur_deg):
        cur_adj, cur_deg = _canonicalize_multigraph_suffix(cur_adj, cur_deg, i)
        key = (i, tuple(cur_deg), tuple(tuple(row) for row in cur_adj))
        if key in seen_states:
            return
        seen_states.add(key)

        while i < nv and cur_deg[i] == 0:
            i += 1
            cur_adj, cur_deg = _canonicalize_multigraph_suffix(cur_adj, cur_deg, i)
            key = (i, tuple(cur_deg), tuple(tuple(row) for row in cur_adj))
            if key in seen_states:
                return
            seen_states.add(key)

        if i == nv:
            candidates.append(tuple(tuple(row) for row in cur_adj))
            return

        required = cur_deg[i]
        blocks = []
        j = i + 1
        while j < nv:
            sig = (cur_deg[j], tuple(cur_adj[k][j] for k in range(i)))
            k = j + 1
            while k < nv and (
                cur_deg[k],
                tuple(cur_adj[t][k] for t in range(i)),
            ) == sig:
                k += 1
            blocks.append((j, k))
            j = k

        options = []

        def pick_singles(pos, need, chosen):
            if need == 0:
                options.append(tuple(chosen))
                return
            if pos == len(blocks):
                return
            start, end = blocks[pos]
            max_take = min(end - start, need)
            for take in range(max_take + 1):
                for offset in range(take):
                    chosen.append((start + offset, 1))
                pick_singles(pos + 1, need - take, chosen)
                for _ in range(take):
                    chosen.pop()

        pick_singles(0, required, [])

        if required >= 2:
            for start, end in blocks:
                for u in range(start, end):
                    if cur_deg[u] < 2:
                        continue
                    chosen = [(u, 2)]

                    def pick_more(pos, need):
                        if need == 0:
                            options.append(tuple(chosen))
                            return
                        if pos == len(blocks):
                            return
                        blk_start, blk_end = blocks[pos]
                        avail = [
                            idx
                            for idx in range(blk_start, blk_end)
                            if idx != u
                        ]
                        max_take = min(len(avail), need)
                        for take in range(max_take + 1):
                            for offset in range(take):
                                chosen.append((avail[offset], 1))
                            pick_more(pos + 1, need - take)
                            for _ in range(take):
                                chosen.pop()

                    pick_more(0, required - 2)

        uniq_options = []
        seen_options = set()
        for opt in options:
            opt = tuple(sorted(opt))
            if opt in seen_options:
                continue
            seen_options.add(opt)
            uniq_options.append(opt)

        for opt in uniq_options:
            if sum(mult for _, mult in opt) != required:
                continue
            next_adj = [row[:] for row in cur_adj]
            next_deg = cur_deg[:]
            ok = True
            for u, mult in opt:
                if next_deg[u] < mult or next_adj[i][u] + mult > 2:
                    ok = False
                    break
                next_adj[i][u] += mult
                next_adj[u][i] += mult
                next_deg[i] -= mult
                next_deg[u] -= mult
            if ok and next_deg[i] == 0 and all(d >= 0 for d in next_deg):
                rec(i + 1, next_adj, next_deg)

    rec(0, adj, deg)
    return candidates


def _matrix_to_multiedges(mult_mat):
    """Helper function to convert an adjacency matrix into a list of edges (specified by 
      endpoint vertices)

    Parameters
    ----------
    mult_mat : sequence of sequence of int
        Symmetric adjacency matrix whose entries give edge multiplicities.

    Returns
    -------
    list of tuple of int
        Edge endpoint pairs. An edge occurs once in the list for each unit of
        its matrix multiplicity.
    """
    n = len(mult_mat)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            mult = mult_mat[i][j]
            for _ in range(mult):
                edges.append((i, j))
    return edges


def _cubic_base_graphs(nv):
    """Generates candidate adjacency matrices using 
      :func:`_candidate_loopless_cubic_multigraph_matrices`, converts the matrices to
      a list of edges using :func:`_matrix_to_multiedges`, throw away disconnected graphs
      using ``nx.is_connected``, group remaining candidate adjacency matrices by their
      Weisfeiler-Lehman hash, then in each group check if individual graphs are 
      isomorphic or not. Finally, non-isomoprhic graphs are retained.

    Parameters
    ----------
    nv : int
        Number of vertices in the cubic multigraphs. Valid cubic graphs require
        an even number of vertices.

    Returns
    -------
    list of tuple
        Connected, pairwise non-isomorphic multigraphs represented as
        ``(edges, vertices)``. ``edges`` is a repeated list of endpoint pairs,
        and ``vertices`` contains the vertex labels.
    """
    if nv in _CUBIC_MULTIGRAPH_CACHE:
        return _CUBIC_MULTIGRAPH_CACHE[nv]

    if nx is None or weisfeiler_lehman_graph_hash is None:
        raise RuntimeError(
            "networkx is required for exact loopless cubic multigraph generation"
        )

    if nv == 2:
        base_graphs = [([(0, 1), (0, 1), (0, 1)], [0, 1])]
        _CUBIC_MULTIGRAPH_CACHE[nv] = base_graphs
        return base_graphs

    verts = list(range(nv))
    edge_match = nx.algorithms.isomorphism.categorical_edge_match("mult", 1)
    reps_by_hash = defaultdict(list)
    base_graphs = []

    for mult_mat in _candidate_loopless_cubic_multigraph_matrices(nv):
        edges = _matrix_to_multiedges(mult_mat)
        g = _weighted_graph_from_edges(edges, verts)
        if not nx.is_connected(g):
            continue
        h = weisfeiler_lehman_graph_hash(g, edge_attr="mult", iterations=5)
        bucket = reps_by_hash[h]
        if any(
            nx.algorithms.isomorphism.GraphMatcher(
                g,
                rep,
                edge_match=edge_match,
            ).is_isomorphic()
            for rep in bucket
        ):
            continue
        bucket.append(g)
        base_graphs.append((edges, verts))

    _CUBIC_MULTIGRAPH_CACHE[nv] = base_graphs
    return _CUBIC_MULTIGRAPH_CACHE[nv]

def _heuristic_cubic_base_graphs(nv, ne):
    """Helper method to replace exhaustive methods of finding all ribbon graphs
       with a heuristic method to potentially find some ribbon graphs. This is
       useful for checking the string integrand for higher genus, where one doesn't
       necessarily need to check all ribbon graphs. This method generates candidate base
       graphs (adjacency assignments from a set of vertices, not a cyclic ordering of 
       edges at a vertex). It is heuristic in the sense that it randomly samples a subset 
       of base graphs, discards all graphs with parallel edges, discards base graphs 
       with the same Weisfeiler-Lehamn hash (even though this hash can be the same for 
       two non-isomorphic graphs), and terminates random sampling once 2,800 consecutive
       samples have hashes that were previously encountered.

    Parameters
    ----------
    nv : int
        Number of vertices in each cubic base graph.
    ne : int
        Number of edges in each cubic base graph. For the sampled cubic-graph
        branch, this must equal ``3 * nv // 2``.

    Returns
    -------
    list of tuple
        Base-graph data represented by ``(edges, vertices)`` pairs. ``edges``
        is a list of two-integer endpoint tuples, and ``vertices`` is the list
        of integer vertex labels. Returns an empty list when the requested
        vertex and edge counts are unsupported.
    """
    if nv == 2 and ne == 3:
        # Theta graph: 2 vertices connected by 3 parallel edges. Keep this base
        # case available so genus-1 one-face generation still has a seed graph.
        return [([(0, 1), (0, 1), (0, 1)], [0, 1])]
    if nv == 4 and ne == 6:
        # K4
        return [([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], [0, 1, 2, 3])]
    if nv == 6 and ne == 9:
        # K_{3,3}
        k33 = (
            [(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)],
            [0, 1, 2, 3, 4, 5],
        )
        # Triangular prism
        prism = (
            [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)],
            [0, 1, 2, 3, 4, 5],
        )
        return [k33, prism]
    if nv >= 8 and nv % 2 == 0 and ne == (3 * nv) // 2:
        target = 19 if nv == 10 else None
        return _generate_connected_cubic_graphs(
            nv,
            target_count=target,
            strict_isomorphism=False,
        )
    return []


def _get_base_graphs(nv, ne, max_exact_multigraph_vertices=_DEFAULT_EXACT_MULTIGRAPH_VERTEX_LIMIT):
    """Helper function to choose between :func:`_cubic_base_graphs` (for number of vertices
    below input number) and :func:`_heuristic_cubic_base_graphs` (for number of vertices
    above input number). Default is 10.

    Parameters
    ----------
    nv : int
        Number of vertices in each requested cubic base graph.
    ne : int
        Number of edges in each requested cubic base graph.
    max_exact_multigraph_vertices : int or None
        Largest vertex count for which exact cubic-multigraph enumeration is
        used. If ``None``, exact enumeration is requested without a vertex
        cutoff.

    Returns
    -------
    list of tuple
        Base-graph data represented by ``(edges, vertices)`` pairs. ``edges``
        is a list of two-integer endpoint tuples, and ``vertices`` is the list
        of integer vertex labels. Depending on the vertex cutoff, the list is
        produced by either exact enumeration or heuristic sampling.
    """
    limit = max_exact_multigraph_vertices
    if limit is not None:
        limit = int(limit)
        if limit < 0:
            raise ValueError("max_exact_multigraph_vertices must be >= 0 or None")

    if nv >= 2 and nv % 2 == 0 and ne == (3 * nv) // 2:
        if limit is None or nv <= limit:
            return _cubic_base_graphs(nv)
    return _heuristic_cubic_base_graphs(nv, ne)


# ============================================================
# Core ribbon graph operations
# ============================================================

def _incident_edge_indices(edges, verts):
    """Helper function that constructs a lookup table of which edges
    meet at each vertex.

    Parameters
    ----------
    edges : sequence of tuple of int
        Edge endpoint pairs indexed by their position in the sequence.
    verts : sequence of int
        Vertex labels for which incident edge indices are requested.

    Returns
    -------
    dict of int to list of int
        Mapping from each vertex label to the indices of the edges incident at
        that vertex.
    """
    inc = {v: [] for v in verts}
    for i, (a, b) in enumerate(edges):
        inc[a].append(i)
        inc[b].append(i)
    return inc


def _next_half_edge(to, eidx, edges, rotation):
    """Given a cyclic ordering of edges around a specific vertex and the edge
      used to arrive at that given vertex, find the next edge to traverse from 
      that vertex.

    Parameters
    ----------
    to : int
        Vertex reached by the current directed edge traversal.
    eidx : int
        Index of the edge used to reach ``to``.
    edges : sequence of tuple of int
        Edge endpoint pairs indexed by ``eidx`` and the entries of
        ``rotation``.
    rotation : dict of int to sequence of int
        Mapping from each vertex label to the cyclic ordering of its incident
        edge indices.

    Returns
    -------
    tuple of int
        Directed traversal state ``(from_vertex, to_vertex, edge_index)`` for
        the next edge along the same face boundary.
    """
    rot = rotation[to]
    pos = rot.index(eidx)
    nxt = rot[(pos + 1) % len(rot)]
    a, b = edges[nxt]
    if a == to:
        return (to, b, nxt)
    else:
        return (to, a, nxt)


def _get_all_face_boundaries(edges, rotation, return_boundaries=False):
    """Given a cyclic ordering of edges associated with a set of vertices, find the
      disjoint cycles (each disjoint cycle being associated with a face boundary).

    Parameters
    ----------
    edges : sequence of tuple of int
        Edge endpoint pairs indexed by their position in the sequence.
    rotation : dict
        Mapping from each vertex label to the cyclic ordering of its incident
        edge indices.
    return_boundaries : bool, optional
        If ``False``, return only the number of face-boundary cycles. If
        ``True``, return the explicit directed half-edge cycles. The default is
        ``False``.

    Returns
    -------
    int or list of list of tuple
        Number of faces when ``return_boundaries=False``. Otherwise, the
        ordered ``(from_vertex, to_vertex, edge_index)`` triples around every
        face boundary.
    """
    visited = set()
    face_count = 0
    faces = [] if return_boundaries else None
    for i, (a, b) in enumerate(edges):
        for he in [(a, b, i), (b, a, i)]:
            if he in visited:
                continue
            face_count += 1
            face = [] if return_boundaries else None
            cur = he
            while cur not in visited:
                visited.add(cur)
                if return_boundaries:
                    face.append(cur)
                cur = _next_half_edge(cur[1], cur[2], edges, rotation)
            if return_boundaries:
                faces.append(face)
    return faces if return_boundaries else face_count


def _sample_random_rotation_system(edges, verts, rng):
    """Assign random cyclic ordering (i.e. rotation system) of edges at every
       vertex for a given base graph.

    Parameters
    ----------
    edges : sequence of tuple of int
        Edge endpoint pairs defining a cubic base graph.
    verts : sequence of int
        Vertex labels of the base graph.
    rng : random.Random
        Pseudorandom-number generator used to draw the orientation mask.

    Returns
    -------
    tuple of dict and int
        The sampled rotation mapping and a bindary integer encoding the rotation system
        efficiently. Bit ``i`` of the binary integer selects one of the two cyclic
        orientations at ``verts[i]``.

    Raises
    ------
    ValueError
        If any vertex is not incident to exactly three edges.
    """
    incident = _incident_edge_indices(edges, verts)
    orientation_mask = rng.getrandbits(len(verts))
    rotation = {}
    for offset, vertex in enumerate(verts):
        local = sorted(incident[vertex])
        if len(local) != 3:
            raise ValueError(
                "Random rotation-system sampling requires a cubic base graph; "
                f"vertex {vertex} has degree {len(local)}."
            )
        if (orientation_mask >> offset) & 1:
            rotation[vertex] = [local[0], local[2], local[1]]
        else:
            rotation[vertex] = local
    return rotation, orientation_mask


def _unique_valid_rotations_for_base(edges, verts, target_faces, verbose=False):
    """For a given base graph, find all inequivalent cyclic orderings (in other words,
       "rotations") of edges that give the requested number of faces.

    Parameters
    ----------
    edges : sequence of tuple of int
        Edge endpoint pairs defining the base graph.
    verts : sequence of int
        Vertex labels of the base graph.
    target_faces : int
        Required number of face-boundary cycles in each retained rotation
        system.
    verbose : bool, optional
        If ``True``, print the numbers of automorphisms, valid labeled rotation
        systems, and inequivalent ribbon graphs. The default is ``False``.

    Returns
    -------
    tuple of list and int
        The first entry is the list of inequivalent ribbon graphs represented
        by ``(edges, vertices, rotation)`` tuples. The second entry is the
        number of labeled rotation systems (i.e., cyclic orderings of edges
        associated with each vertex) with the requested face count before
        isomorphism reduction.
    """
    autos = _compute_automorphisms(edges, verts)
    valid_count = 0
    seen = set()
    unique = []

    inc = _incident_edge_indices(edges, verts)
    vert_perms = []
    for v in verts:
        fixed_first = min(inc[v])
        reps = [list(p) for p in permutations(inc[v]) if p[0] == fixed_first]
        vert_perms.append(reps)

    for combo in iproduct(*vert_perms):
        rotation = {verts[i]: list(combo[i]) for i in range(len(verts))}
        if _get_all_face_boundaries(edges, rotation) != target_faces:
            continue
        valid_count += 1
        rg = (edges, verts, rotation)
        key_rg = _canonical_ribbon_key(rg, autos)
        if key_rg in seen:
            continue
        seen.add(key_rg)
        unique.append(rg)

    if verbose:
        print(
            f"  Base graph ({len(verts)}V, {len(edges)}E): "
            f"{len(autos)} automorphisms, {valid_count} rotation systems, {len(unique)} unique"
        )
    return unique, valid_count


def _unique_valid_rotations_for_base_task(args):
    """Multiprocessing wrapper for :func:`_unique_valid_rotations_for_base` since finding all
       valid cyclic orderings can be quite computationally taxing.

    Parameters
    ----------
    args : tuple
        Packed task data ``(edges, vertices, target_faces)``. ``edges`` is a
        sequence of endpoint pairs, ``vertices`` is a sequence of vertex
        labels, and ``target_faces`` is the required number of face-boundary
        cycles.

    Returns
    -------
    tuple
        Four entries ``(unique, valid_count, vertex_count, edge_count)``.
        ``unique`` is the list of inequivalent ribbon graphs, ``valid_count``
        is the number of valid labeled rotation systems before isomorphism
        reduction, and the final two entries record the base graph's numbers
        of vertices and edges.
    """
    edges, verts, target_faces = args
    unique, valid_count = _unique_valid_rotations_for_base(
        edges, verts, target_faces, verbose=False
    )
    return unique, valid_count, len(verts), len(edges)


# ============================================================
# Isomorphism detection
# ============================================================

def _compute_automorphisms(edges, verts):
    """Given a base graph, compute the full set of automorphisms
    mapping the base graph to itself.

    Parameters
    ----------
    edges : sequence of tuple of int
        Edge endpoint pairs defining the base graph. Repeated endpoint pairs
        represent parallel edges and are included in the automorphism test.
    verts : sequence of int
        Vertex labels of the base graph.

    Returns
    -------
    list of dict
        Maps between vertices representing every automorphism. Each dictionary maps an
        original vertex label to its image under one edge-preserving vertex
        relabeling; the identity map is included.
    """
    n = len(verts)
    if nx is not None and n >= 8:
        edge_match = nx.algorithms.isomorphism.categorical_edge_match("mult", 1)
        g = _weighted_graph_from_edges(edges, verts)
        gm = nx.algorithms.isomorphism.GraphMatcher(g, g, edge_match=edge_match)
        return [dict(m) for m in gm.isomorphisms_iter()]

    edge_ms = sorted(tuple(sorted(e)) for e in edges)
    autos = []
    for perm in permutations(range(n)):
        vmap = {verts[i]: verts[perm[i]] for i in range(n)}
        mapped = sorted(tuple(sorted([vmap[a], vmap[b]])) for a, b in edges)
        if mapped == edge_ms:
            autos.append(vmap)
    return autos


def _min_cyclic_tuple(seq):
    """For a given cylic ordering of edges associated with a given vertex, reorder
       to a canonical ordering (with the lowest integer labeled edge first).

    Parameters
    ----------
    seq : sequence of int
        Nonempty cyclic sequence of integer edge indices.

    Returns
    -------
    tuple of int
        Cyclic rotation of ``seq``. The edge with smallest edge number is the first entry.
    """
    n = len(seq)
    return min(tuple(seq[i:] + seq[:i]) for i in range(n))


def _build_pair_groups(edges, vmap):
    """For a given automorphism of the base graph that specifies a map of vertices,
       find the associated map of edges. For parallel edges
       (i.e. two or three edges that connect the same pair of vertices), no unique map is determined,
       but one can still determine what input grouping of parallel edges is mapped to what
       output grouping. In this case, a tuple of the input parallel edges and output parallel
       edges is given.

    Parameters
    ----------
    edges : sequence of tuple of int
        Edge endpoint pairs defining the base graph. The position of each pair
        in the sequence is defined to be its edge index (a number keeping track
        of the edges).
    vmap : dict of int to int
        Vertex map representing an automorphism of the base graph.

    Returns
    -------
    list of tuple or None
        Source/target edge-index groups ``(source_indices, target_indices)``
        compatible with the mapped endpoint pairs. Each group contains lists
        of edge indices. Returns ``None`` if a mapped pair of vertices has
        different source and target multiplicities (i.e., the automorphism
        cannot be a valid automorphism).
    """
    edge_pairs = [tuple(sorted(e)) for e in edges]
    mapped_pairs = [tuple(sorted([vmap[a], vmap[b]])) for a, b in edges]

    src_by_pair = defaultdict(list)
    for i, mp in enumerate(mapped_pairs):
        src_by_pair[mp].append(i)

    tgt_by_pair = defaultdict(list)
    for j, ep in enumerate(edge_pairs):
        tgt_by_pair[ep].append(j)

    pair_groups = []
    for pair, srcs in src_by_pair.items():
        tgts = tgt_by_pair.get(pair, [])
        if len(srcs) != len(tgts):
            return None
        pair_groups.append((srcs, tgts))
    return pair_groups


def _iter_edge_maps(pair_groups, idx=0, emap=None):
    """Recursive function that expands the mapping of edges returned by
       :func:`_build_pair_groups`
       (which returns mappings of groups of parallel vertices) into every edge mapping
       compatible with the automorphism of vertices.

    Parameters
    ----------
    pair_groups : sequence of tuple
        Source/target edge-index groups produced by :func:`_build_pair_groups`.
        Each entry has the form ``(source_indices, target_indices)``.
    idx : int, optional
        Index of the group of edges currently being expanded. This is used by the recursion
        structure of the function. Defaults to ``0``.
    emap : dict of int to int or None, optional
        Partial source-to-target edge map accumulated by the recursion. This is
        internal recursion state and defaults to ``None``.

    Yields
    ------
    dict of int to int
        A complete source-edge-index to target-edge-index map compatible with
        every group in ``pair_groups``.
    """
    if emap is None:
        emap = {}
    if idx == len(pair_groups):
        yield emap
        return
    srcs, tgts = pair_groups[idx]
    for tp in permutations(tgts):
        new_emap = dict(emap)
        for s, t in zip(srcs, tp):
            new_emap[s] = t
        yield from _iter_edge_maps(pair_groups, idx + 1, new_emap)


def _is_simple_graph(edges):
    """Return True iff there are no parallel edges (two edges connecting the same vertices)
      or loops (edge that begins/terminates at the same vertex).

    Parameters
    ----------
    edges : sequence of tuple of int
        Edge endpoint pairs defining the graph.

    Returns
    -------
    bool
        ``True`` when every edge joins distinct vertices and every unordered
        endpoint pair occurs at most once; otherwise ``False``.
    """
    seen = set()
    for a, b in edges:
        if a == b:
            return False
        p = (a, b) if a < b else (b, a)
        if p in seen:
            return False
        seen.add(p)
    return True


def _canonical_rotation_ordering(rg, vmap, emap):
    """Convert all edge orderings at vertices into canonical order using 
       :func:`_min_cyclic_tuple`.

    Parameters
    ----------
    rg : tuple
        Ribbon graph represented by ``(edges, vertices, rotation)``.
    vmap : dict of int to int
        Vertex automorphism mapping each original vertex label to its image.
    emap : dict of int to int
        Edge map compatible with ``vmap``, mapping each original edge index to
        its image.

    Returns
    -------
    tuple
        Immutable signature consisting of ``(mapped_vertex,
        canonical_mapped_rotation)`` pairs sorted by mapped vertex. Each cyclic ordering
        of edges associated with a vertex has the edge with the smallest edge number first.
    """
    _, verts, rot = rg
    by_vertex = []
    for v in sorted(verts):
        mapped_v = vmap[v]
        mapped_rot = [emap[e] for e in rot[v]]
        by_vertex.append((mapped_v, _min_cyclic_tuple(mapped_rot)))
    return tuple(sorted(by_vertex))


def _canonical_ribbon_key(rg, autos):
    """Given a candidate ribbon graph (i.e. a set of vertices, edges, and 
       cyclical ordering of edges at each vertex) and automorphisms of the
       vertices, determine all compatible edge relabelings. For each vertex
       and edge relabeling, put each local cyclic ordering into canonical form,
       and sort the reuslting (mapped verte, rotation) pairs by increasing vertex
       label. Select the tuple with the smallest edge labels in the first position
       they differ.

       The upshot is that a canonical edge/base graph is chosen amongst all automorphisms
       such that isomorphic ribbon graphs can be determined.

    Parameters
    ----------
    rg : tuple
        Ribbon graph represented by ``(edges, vertices, rotation)``.
    autos : sequence of dict
        Automorphisms of the base graph (i.e. the vertices), as returned
        by :func:`_compute_automorphisms`. Each map sends an original vertex
        label to its image.

    Returns
    -------
    tuple or None
        Canonical ordered tuple of (mapped_vertex, canonical_rotation) pairs, sorted by increasing
        vertex label. Every isomorphic ribbon graph will return the same tuple.
    """
    edges, _, _ = rg
    simple = _is_simple_graph(edges)
    pair_to_idx = None
    if simple:
        pair_to_idx = {}
        for i, (a, b) in enumerate(edges):
            p = (a, b) if a < b else (b, a)
            pair_to_idx[p] = i

    best = None
    for vmap in autos:
        if simple:
            # For simple graphs, vertex map determines edge map uniquely.
            emap = {}
            for i, (a, b) in enumerate(edges):
                ma, mb = vmap[a], vmap[b]
                p = (ma, mb) if ma < mb else (mb, ma)
                emap[i] = pair_to_idx[p]
            canonical_ordering = _canonical_rotation_ordering(rg, vmap, emap)
            if best is None or canonical_ordering < best:
                best = canonical_ordering
            continue

        pair_groups = _build_pair_groups(edges, vmap)
        if pair_groups is None:
            continue
        for emap in _iter_edge_maps(pair_groups):
            canonical_ordering = _canonical_rotation_ordering(rg, vmap, emap)
            if best is None or canonical_ordering < best:
                best = canonical_ordering
    return best


# ============================================================
# Main generation function
# ============================================================

def _generate_ribbon_graphs_from_counts(
    n_faces,
    n_edges,
    verbose=False,
    workers=None,
    max_exact_multigraph_vertices=_DEFAULT_EXACT_MULTIGRAPH_VERTEX_LIMIT,
):
    """Generate non-isomorphic cubic ribbon graphs.

    Parameters
    ----------
    n_faces : int
        Number of faces.
    n_edges : int
        Number of edges.
    verbose : bool, optional
        Print graph counts and intermediate information.
    workers : int or None, optional
        Optional multiprocessing. Number of worker processes used to 
        enumerate rotation systems.
    max_exact_multigraph_vertices : int or None, optional
        Maximum vertex count for which the base cubic multigraphs are
        enumerated exactly. Use ``None`` to force exact multigraph bases at all
        sizes, which can be slow.

    Returns
    -------
    list of tuple
        Ribbon graphs represented as ``(edges, vertices, rotation)`` tuples.
    """
    n_vertices = 2 * n_edges // 3
    genus = (2 - n_vertices + n_edges - n_faces) // 2
    if verbose:
        print(f"V={n_vertices}, E={n_edges}, F={n_faces}, g={genus}")

    base_graphs = _get_base_graphs(
        n_vertices,
        n_edges,
        max_exact_multigraph_vertices=max_exact_multigraph_vertices,
    )
    if not base_graphs:
        if verbose:
            print("No base cubic graphs available")
        return []
    if verbose:
        print(f"Using {len(base_graphs)} base cubic graph(s)")

    unique_all = []
    total_valid_rotation_systems = 0

    if workers is None:
        # Safe default across restricted runtimes. Use >1 explicitly to parallelize.
        workers = 1
    workers = max(1, int(workers))

    if workers == 1 or len(base_graphs) <= 1:
        for edges, verts in base_graphs:
            unique_base, valid_count = _unique_valid_rotations_for_base(
                edges, verts, n_faces, verbose=verbose
            )
            total_valid_rotation_systems += valid_count
            unique_all.extend(unique_base)
    else:
        tasks = [(edges, verts, n_faces) for edges, verts in base_graphs]
        executor = None
        try:
            executor = ProcessPoolExecutor(max_workers=workers)
        except (PermissionError, OSError):
            # Sandboxed environments may disallow process semaphores.
            executor = ThreadPoolExecutor(max_workers=workers)

        with executor as ex:
            futures = [ex.submit(_unique_valid_rotations_for_base_task, t) for t in tasks]
            for fut in as_completed(futures):
                unique_base, valid_count, nv, ne = fut.result()
                total_valid_rotation_systems += valid_count
                unique_all.extend(unique_base)
                if verbose:
                    print(
                        f"  Base graph ({nv}V, {ne}E): "
                        f"{valid_count} rotation systems, {len(unique_base)} unique"
                    )
    if verbose:
        print(f"Total valid rotation systems: {total_valid_rotation_systems}")
        print(f"Non-isomorphic ribbon graphs: {len(unique_all)}")

    return unique_all


def _edge_count_for_cubic_graph(genus, n_faces):
    """For connected cubic graphs, calculate number of edges
       from genus and number of faces. Just uses Euler's relation.

    Parameters
    ----------
    genus : int
        Genus of the oriented surface.
    n_faces : int
        Number of faces of the ribbon graph.

    Returns
    -------
    int
        Number of edges required by Euler's relation and the fact that all vertices
        are connected to three edges, ``3 * (n_faces - 2 + 2 * genus)``.

    Raises
    ------
    ValueError
        If ``genus`` is negative, ``n_faces`` is not positive, or the supplied
        topology does not admit a positive integral cubic-graph edge and vertex
        count.
    """
    # Euler + cubic condition:
    # 2 - 2g = V - E + F and 3V = 2E -> E = 3(F - 2 + 2g)
    if genus < 0 or n_faces <= 0:
        raise ValueError("genus must be >= 0 and n_faces must be >= 1")
    n_edges = 3 * (n_faces - 2 + 2 * genus)
    if n_edges <= 0:
        raise ValueError("Invalid (genus, faces) for connected cubic ribbon graphs")
    # Cubic graph requires V = 2E/3 to be an integer.
    if (2 * n_edges) % 3 != 0:
        raise ValueError("Invalid (genus, faces): cubic vertex count is non-integer")
    return n_edges


def generate_ribbon_graphs(
    genus,
    n_faces=1,
    verbose=False,
    n_processes=None,
    max_vertices_all_ribbon_graphs=_DEFAULT_EXACT_MULTIGRAPH_VERTEX_LIMIT,
):
    """Generate non-isomorphic (cubic) ribbon graphs at fixed genus and number
    of faces.

    The number of faces is the number of punctures on the associated Riemann surface.  The number of edges and vertices follows from Euler's
    relation and the fact that all ribbon graph vertices are the endpoints of three edges.

    Parameters
    ----------
    genus : int
        Genus of the oriented surface. Must be nonnegative.
    n_faces : int, optional
        Number of ribbon graph faces, equivalently the number of punctures in the
        Riemann surface. Must be positive. Default is 1.
    verbose : bool, optional
        Print graph counts and intermediate enumeration information.
    n_processes : int or None, optional
        Optional multiprocessing. Number of worker processes used to enumerate rotation systems. ``None``
        means 1 worker.
    max_vertices_all_ribbon_graphs : int or None, optional
        Largest total number of vertices for which all ribbon graphs are enumerated. Above this number,
        heuristics are used to generate a subset of ribbon graphs. Defaults to 10.

    Returns
    -------
    list of tuple
        Non-isomorphic ribbon graphs represented as ``(edges, vertices,
        rotation)``. ``rotation[vertex]`` gives the cyclic ordering of the
        incident edges at ``[vertex]``. Edges emanating from a given vertex are labeled by the other vertex they
        end on.

    Notes
    -----
    This function has been tested against known counts for genus 1 to 3 with 1 face.
    """
    n_edges = _edge_count_for_cubic_graph(genus, n_faces)
    return _generate_ribbon_graphs_from_counts(
        n_faces,
        n_edges,
        verbose=verbose,
        workers=n_processes,
        max_exact_multigraph_vertices=max_vertices_all_ribbon_graphs,
    )


def sample_ribbon_graph(
    genus,
    n_faces=1,
    seed=0,
    max_graph_trials=100,
    max_rotation_trials=100_000,
):
    """Sample one simple cubic ribbon graph at fixed genus using heuristic (but faster) methods.

    Parameters
    ----------
    genus : int
        Genus of the oriented surface. Must be nonnegative.
    n_faces : int, optional
        Number of ribbon graph faces, equivalently the number of punctures in the
                Riemann surface. Must be positive. Default is 1.
    seed : int, optional
        Seed for random sampling. Default is 0.
    max_graph_trials : int, optional
        Maximum number of (randomly chosen) vertices/edges (base graphs) to sample. 
        The default is ``100``.
    max_rotation_trials : int, optional
        Maximum number of random rotation systems (i.e. orderings of edges associated with a vertex)
        to try for each accepted connected base graph. The default is ``100_000``.

    Returns
    -------
    tuple of tuple and dict
        The sampled ribbon graph ``(edges, vertices, rotation)`` and metadata
        recording the seed, graph trial number, graph seed, rotation trial number, and
        cyclic orientation at each vertex. The latter is encoded as a binary string, with each bit
        representing one of the two inequivalent cylic orientations. For a given vertex, with the smallest edge
        index first reading, if the second edge index is the second highest, then the bit is 0.
        Alternatively, if the second edge index is the highest, then the bit is 1.

    Raises
    ------
    RuntimeError
        If NetworkX is unavailable or no valid rotation system is found within
        the the ``max_graph_trials`` and ``max_rotation_trials`` limits.
    ValueError
        If the requested graph is too small for a simple cubic base graph or
        either trial limit is not positive.

    Notes
    -----
    The methods used in this function are partially heuristic. They guarantee that
    any ribbon graph returned is an actual ribbon graph structure, but do not guarantee
    the return of a ribbon graph structure. It is best used for higher genus/higher number of
    faces, when one is only interested in the string integrand and not amplitude.
    """
    if nx is None:
        raise RuntimeError("networkx is required for random ribbon-graph sampling")

    n_edges = _edge_count_for_cubic_graph(genus, n_faces)
    n_vertices = (2 * n_edges) // 3
    if n_vertices <= 3:
        raise ValueError(
            "Random simple-cubic sampling requires at least four vertices."
        )

    max_graph_trials = int(max_graph_trials)
    max_rotation_trials = int(max_rotation_trials)
    if max_graph_trials <= 0 or max_rotation_trials <= 0:
        raise ValueError("max_graph_trials and max_rotation_trials must be positive")

    seed = int(seed)
    rng = random.Random(seed)
    for graph_trial in range(max_graph_trials):
        graph_seed = rng.randrange(2**32)
        graph = nx.random_regular_graph(3, n_vertices, seed=graph_seed)
        if not nx.is_connected(graph):
            continue

        edges = sorted(
            (min(int(a), int(b)), max(int(a), int(b)))
            for a, b in graph.edges()
        )
        vertices = list(range(n_vertices))

        for rotation_trial in range(max_rotation_trials):
            rotation, orientation_mask = _sample_random_rotation_system(
                edges,
                vertices,
                rng,
            )
            if _get_all_face_boundaries(edges, rotation) != n_faces:
                continue
            metadata = {
                "topology_seed": seed,
                "graph_trial": graph_trial,
                "graph_seed": int(graph_seed),
                "rotation_trial": rotation_trial,
                "orientation_mask": int(orientation_mask),
            }
            return (edges, vertices, rotation), metadata

    raise RuntimeError(
        "No valid rotation system found within the stated graph and rotation "
        "trial limits."
    )


# ============================================================
# Boundary and sewing data
# ============================================================

def get_boundary_data(ribbon_graph, edge_lengths=None):
    """Return the combinatorial and (if edge lengths are provided) discretized boundary data.

    For the case of one face, which is the only case that has been tested, every graph edge
    occurs exactly twice across the edge of the single face, and ``sewing`` records the two locations
    of the edges sewn together.

    The returned dictionary has the following structure::

        {
            "genus": int,
            "n_faces": int,
            "boundaries": tuple[tuple[(from_vertex, to_vertex, edge_index), ...], ...],
            "edge_sequences": tuple[tuple[edge_index, ...], ...],
            "vertex_sequences": tuple[tuple[vertex_index, ...], ...],
            "sewing": {
                edge_index: ((face_index, position), (face_index, position)),
                ...,
            },
            "edge_lengths": tuple[int, ...] | None,
            "boundary_segment_starts": tuple[tuple[int, ...], ...] | None,
            "boundary_lengths": tuple[int, ...] | None,
        }

    The final three entries are ``None`` when ``edge_lengths`` is omitted.

    Parameters
    ----------
    ribbon_graph : tuple
        A graph ``(edges, vertices, rotation)`` returned by
        :func:`generate_ribbon_graphs`.
    edge_lengths : sequence of int or None, optional
        Positive integer discretization length of every graph edge, ordered by
        edge index. When this parameter is supplied, the returned dictionary includes 
        the starting site of every boundary segment and the length of every face boundary.

    Returns
    -------
    dict
        Genus, face boundaries, edge and vertex sequences, sewing map, and
        optional discretized boundary geometry in the structure specified
        above.

    Raises
    ------
    TypeError
        If a supplied edge length is not an integer.
    ValueError
        If the graph data are inconsistent, an edge does not occur exactly
        twice across all face boundaries, or a supplied edge length is not
        positive.
    """
    try:
        edges, vertices, rotation = ribbon_graph
    except (TypeError, ValueError) as error:
        raise ValueError(
            "ribbon_graph must be an (edges, vertices, rotation) tuple"
        ) from error

    vertices = tuple(vertices)
    if vertices != tuple(range(len(vertices))):
        raise ValueError("ribbon-graph vertex indices must be 0, ..., n_vertices - 1")
    if not edges:
        raise ValueError("ribbon_graph must contain at least one edge")

    vertex_set = set(vertices)
    adjacency = {vertex: set() for vertex in vertices}
    for start, end in edges:
        if start not in vertex_set or end not in vertex_set:
            raise ValueError(f"edge ({start}, {end}) references an unknown vertex")
        adjacency[start].add(end)
        adjacency[end].add(start)
    reached = {vertices[0]}
    queue = deque([vertices[0]])
    while queue:
        vertex = queue.popleft()
        for neighbor in sorted(adjacency[vertex]):
            if neighbor not in reached:
                reached.add(neighbor)
                queue.append(neighbor)
    if reached != vertex_set:
        raise ValueError("ribbon_graph must be connected")

    raw_faces = _get_all_face_boundaries(
        edges,
        rotation,
        return_boundaries=True,
    )
    n_faces = len(raw_faces)
    euler_numerator = 2 - len(vertices) + len(edges) - n_faces
    if euler_numerator < 0 or euler_numerator % 2:
        raise ValueError("ribbon_graph does not define a nonnegative integer genus")
    genus = euler_numerator // 2

    boundaries = []
    edge_sequences = []
    vertex_sequences = []
    edge_positions = defaultdict(list)

    for face_idx, face in enumerate(raw_faces):
        boundary = []
        edge_sequence = []
        vertex_sequence = []
        for pos_idx, (frm, to, edge_index) in enumerate(face):
            boundary.append((frm, to, edge_index))
            edge_sequence.append(edge_index)
            vertex_sequence.append(frm)
            edge_positions[edge_index].append((face_idx, pos_idx))

        boundaries.append(tuple(boundary))
        edge_sequences.append(tuple(edge_sequence))
        vertex_sequences.append(tuple(vertex_sequence))

    sewing = {}
    for edge_index, positions in edge_positions.items():
        if len(positions) != 2:
            raise ValueError(
                f"Expected edge {edge_index} to appear twice across face boundaries, "
                f"got {len(positions)} times"
            )
        sewing[edge_index] = tuple(positions)

    validated_lengths = None
    boundary_segment_starts = None
    boundary_lengths = None
    if edge_lengths is not None:
        lengths = tuple(edge_lengths)
        if len(lengths) != len(edges):
            raise ValueError(f"expected {len(edges)} edge lengths, got {len(lengths)}")
        normalized_lengths = []
        for edge_index, length in enumerate(lengths):
            if isinstance(length, bool) or not isinstance(length, Integral):
                raise TypeError(f"edge_lengths[{edge_index}] must be an integer")
            length = int(length)
            if length <= 0:
                raise ValueError(f"edge_lengths[{edge_index}] must be positive")
            normalized_lengths.append(length)
        validated_lengths = tuple(normalized_lengths)

        starts_by_face = []
        lengths_by_face = []
        for edge_sequence in edge_sequences:
            starts = []
            position = 0
            for edge_index in edge_sequence:
                starts.append(position)
                position += validated_lengths[edge_index]
            starts_by_face.append(tuple(starts))
            lengths_by_face.append(position)
        boundary_segment_starts = tuple(starts_by_face)
        boundary_lengths = tuple(lengths_by_face)
        if sum(boundary_lengths) != 2 * sum(validated_lengths):
            raise ValueError(
                "boundary lengths are inconsistent with pairwise edge sewing"
            )

    return {
        "genus": genus,
        "n_faces": n_faces,
        "boundaries": tuple(boundaries),
        "edge_sequences": tuple(edge_sequences),
        "vertex_sequences": tuple(vertex_sequences),
        "sewing": sewing,
        "edge_lengths": validated_lengths,
        "boundary_segment_starts": boundary_segment_starts,
        "boundary_lengths": boundary_lengths,
    }


__all__ = (
    "generate_ribbon_graphs",
    "get_boundary_data",
)
