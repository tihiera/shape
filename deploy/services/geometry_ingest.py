"""
services/geometry_ingest.py
───────────────────────────
Geometry ingest + conversion service.

Responsibilities:
  - Read STEP files via gmsh/meshio
  - Extract surface mesh
  - Normalize units (mm vs m) + coordinate normalization
  - Extract REAL centerline graph from pipe meshes (VMTK or fallback)
  - Preserve original node mapping for UI highlighting
  - Store result locally per session

Centerline extraction strategy:
  1. If input already looks like a centerline graph (degree ≤ 2) → pass through
  2. Detect open boundary loops on the surface mesh (pipe ends)
  3. Use VMTK (via VTK Python API) to compute centerlines between pipe ends
  4. Map centerline points back to nearest mesh vertices
"""

from __future__ import annotations

import json
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════════════
# STEP → SURFACE MESH
# ══════════════════════════════════════════════════════════════════════

def read_step_to_mesh(step_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a STEP file and extract surface mesh via gmsh + meshio.

    Returns:
        vertices: (V, 3) mesh vertices
        faces:    (F, 3) triangle face indices
        edges:    (E, 2) edge connectivity from triangles
    """
    try:
        import gmsh
    except ImportError:
        raise ImportError("gmsh is required: pip install gmsh")
    try:
        import meshio
    except ImportError:
        raise ImportError("meshio is required: pip install meshio")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # quiet
    gmsh.open(step_path)
    gmsh.model.mesh.generate(2)  # surface mesh

    with tempfile.NamedTemporaryFile(suffix=".vtk", delete=False) as tmp:
        tmp_path = tmp.name
    gmsh.write(tmp_path)
    gmsh.finalize()

    mesh = meshio.read(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)

    vertices = mesh.points.astype(np.float64)

    # Extract triangles
    faces = np.empty((0, 3), dtype=np.int64)
    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            faces = np.vstack([faces, cell_block.data])

    # Extract edges from triangles
    edge_set: set = set()
    for f in faces:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            edge_set.add((min(a, b), max(a, b)))
    edges = np.array(sorted(edge_set), dtype=np.int64) if edge_set else np.empty((0, 2), dtype=np.int64)

    return vertices, faces, edges


# ══════════════════════════════════════════════════════════════════════
# COORDINATE NORMALIZATION
# ══════════════════════════════════════════════════════════════════════

def normalize_coordinates(vertices: np.ndarray,
                          target_unit: str = "mm") -> Tuple[np.ndarray, float]:
    """
    Center and optionally scale coordinates.
    Returns (normalized_vertices, scale_factor).
    """
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid

    bbox_diag = np.linalg.norm(centered.max(axis=0) - centered.min(axis=0))

    scale = 1.0
    if bbox_diag < 1.0 and target_unit == "mm":
        scale = 1000.0
        centered *= scale

    return centered, scale


# ══════════════════════════════════════════════════════════════════════
# HELPER: GRAPH DEGREE CHECK
# ══════════════════════════════════════════════════════════════════════

def _degrees(num_nodes: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    """Compute degree of each node."""
    deg = np.zeros(num_nodes, dtype=np.int32)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return deg


def _looks_like_centerline(vertices: np.ndarray, edges: List[Tuple[int, int]]) -> bool:
    """
    Heuristic: a centerline graph has most nodes with degree 1 or 2.
    A triangle surface mesh has degree 4-8+ everywhere.
    """
    if len(edges) == 0 or len(vertices) < 2:
        return False
    deg = _degrees(len(vertices), edges)
    frac_ok = float(np.mean(deg <= 2))
    # Allow a few junctions (deg 3+), but not triangle mesh chaos
    return frac_ok > 0.98


# ══════════════════════════════════════════════════════════════════════
# HELPER: BOUNDARY LOOP DETECTION
# ══════════════════════════════════════════════════════════════════════

def _boundary_loops_from_faces(
    faces: np.ndarray,
) -> List[List[Tuple[int, int]]]:
    """
    Find boundary edge components on a triangle mesh.
    Boundary edge = belongs to exactly one triangle (open pipe ends).

    Returns list of components, each component is a list of boundary edges.
    """
    if faces is None or len(faces) == 0:
        return []

    # Count how many triangles each undirected edge belongs to
    edge_count: Dict[Tuple[int, int], int] = {}
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        for u, v in [(a, b), (b, c), (c, a)]:
            e = (u, v) if u < v else (v, u)
            edge_count[e] = edge_count.get(e, 0) + 1

    # Boundary = edges with count == 1
    boundary_edges = [e for e, c in edge_count.items() if c == 1]
    if not boundary_edges:
        return []

    # Build adjacency over boundary edges
    adj: Dict[int, List[int]] = {}
    for u, v in boundary_edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    # Connected components via BFS
    visited: set = set()
    comps: List[List[int]] = []

    for start in adj:
        if start in visited:
            continue
        stack = [start]
        comp: List[int] = []
        visited.add(start)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adj.get(x, []):
                if y not in visited:
                    visited.add(y)
                    stack.append(y)
        comps.append(comp)

    # Collect edges per component
    bset = set(boundary_edges)
    comp_edges: List[List[Tuple[int, int]]] = []
    for comp_verts in comps:
        s = set(comp_verts)
        edges = [(u, v) for u, v in bset if u in s and v in s]
        if edges:
            comp_edges.append(edges)

    return comp_edges


def _loop_centroid(vertices: np.ndarray, loop_edges: List[Tuple[int, int]]) -> np.ndarray:
    """Compute the centroid of all vertices in a boundary loop."""
    verts: set = set()
    for u, v in loop_edges:
        verts.add(u)
        verts.add(v)
    pts = vertices[list(verts)]
    return pts.mean(axis=0)


# ══════════════════════════════════════════════════════════════════════
# HELPER: NEAREST VERTEX MAP
# ══════════════════════════════════════════════════════════════════════

def _nearest_vertex_map(mesh_vertices: np.ndarray, cl_nodes: np.ndarray) -> Dict[int, int]:
    """
    Map each centerline point to its nearest mesh vertex index.
    Used for UI highlighting (click centerline → highlight surface region).
    """
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(mesh_vertices)
        _, idx = tree.query(cl_nodes, k=1)
        return {i: int(idx[i]) for i in range(len(cl_nodes))}
    except ImportError:
        # Chunked brute force fallback (no scipy)
        out: Dict[int, int] = {}
        for i in range(len(cl_nodes)):
            p = cl_nodes[i]
            best_j = 0
            best_d = float("inf")
            for j0 in range(0, len(mesh_vertices), 50000):
                chunk = mesh_vertices[j0:j0 + 50000]
                d2 = np.sum((chunk - p) ** 2, axis=1)
                j = int(np.argmin(d2))
                if float(d2[j]) < best_d:
                    best_d = float(d2[j])
                    best_j = j0 + j
            out[i] = best_j
        return out


# ══════════════════════════════════════════════════════════════════════
# VMTK CENTERLINE EXTRACTION (via VTK Python API)
# ══════════════════════════════════════════════════════════════════════

def _run_vmtk_centerlines(
    vertices: np.ndarray,
    faces: np.ndarray,
    seed_points: List[np.ndarray],
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Compute centerlines using vmtkcenterlines CLI.

    Builds a temporary .vtp surface, runs vmtkcenterlines with the
    boundary loop centroids as source/target seeds, reads back the
    resulting polyline centerline.

    Requires: vmtk + vtk installed in the environment.
    """
    import vtk

    # Build vtkPolyData surface
    pts = vtk.vtkPoints()
    pts.SetNumberOfPoints(len(vertices))
    for i, p in enumerate(vertices):
        pts.SetPoint(i, float(p[0]), float(p[1]), float(p[2]))

    polys = vtk.vtkCellArray()
    for f in faces:
        tri = vtk.vtkTriangle()
        tri.GetPointIds().SetId(0, int(f[0]))
        tri.GetPointIds().SetId(1, int(f[1]))
        tri.GetPointIds().SetId(2, int(f[2]))
        polys.InsertNextCell(tri)

    surf = vtk.vtkPolyData()
    surf.SetPoints(pts)
    surf.SetPolys(polys)

    # Clean + triangulate for robustness
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(surf)
    cleaner.Update()

    tri_filter = vtk.vtkTriangleFilter()
    tri_filter.SetInputData(cleaner.GetOutput())
    tri_filter.Update()

    surf = tri_filter.GetOutput()

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        in_vtp = td_path / "surface.vtp"
        out_vtp = td_path / "centerlines.vtp"

        w = vtk.vtkXMLPolyDataWriter()
        w.SetFileName(str(in_vtp))
        w.SetInputData(surf)
        w.Write()

        # Need at least 2 open boundaries (source + target)
        if len(seed_points) < 2:
            raise RuntimeError(
                "Need at least 2 open boundary loops to compute centerline. "
                "Found only {len(seed_points)}."
            )

        source = seed_points[0]
        targets = seed_points[1:]

        # Build vmtkcenterlines CLI command
        cmd = [
            "vmtkcenterlines",
            "-ifile", str(in_vtp),
            "-ofile", str(out_vtp),
            "-seedselector", "pointlist",
            "-sourcepoints",
            ",".join(str(float(v)) for v in source),
            "-targetpoints",
            ";".join(
                ",".join(str(float(v)) for v in t)
                for t in targets
            ),
        ]

        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if p.returncode != 0:
            raise RuntimeError(
                f"vmtkcenterlines failed (exit {p.returncode}).\n"
                f"STDOUT:\n{p.stdout}\n"
                f"STDERR:\n{p.stderr}\n"
                "Hint: if seed flags differ in your VMTK build, "
                "try seedselector=openprofiles instead."
            )

        # Read back the centerline polyline
        r = vtk.vtkXMLPolyDataReader()
        r.SetFileName(str(out_vtp))
        r.Update()
        cl = r.GetOutput()

        cl_pts = cl.GetPoints()
        if cl_pts is None or cl_pts.GetNumberOfPoints() == 0:
            raise RuntimeError("VMTK produced no centerline points.")

        cl_nodes = np.array(
            [cl_pts.GetPoint(i) for i in range(cl_pts.GetNumberOfPoints())],
            dtype=np.float64,
        )

        cl_edges: List[Tuple[int, int]] = []
        lines = cl.GetLines()
        lines.InitTraversal()
        id_list = vtk.vtkIdList()
        while lines.GetNextCell(id_list):
            for i in range(id_list.GetNumberOfIds() - 1):
                u = int(id_list.GetId(i))
                v = int(id_list.GetId(i + 1))
                cl_edges.append((u, v))

        # Deduplicate
        cl_edges = list({(min(u, v), max(u, v)) for u, v in cl_edges})

        return cl_nodes, cl_edges


# ══════════════════════════════════════════════════════════════════════
# MAIN CENTERLINE EXTRACTOR
# ══════════════════════════════════════════════════════════════════════

def extract_centerline_from_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    edges: Optional[np.ndarray] = None,
    method: str = "vmtk",
) -> Tuple[np.ndarray, List[Tuple[int, int]], Dict[int, int]]:
    """
    Real centerline extraction for a pipe-like surface mesh.

    Strategy:
      1. If input already looks like a centerline graph (degree ≤ 2) → pass through
      2. Find open boundary loops on the surface mesh (pipe ends)
      3. Use VMTK to compute centerlines between boundary centroids
      4. Map centerline points back to nearest mesh vertices

    Parameters:
        vertices: (V, 3) mesh vertices
        faces:    (F, 3) triangle face indices
        edges:    (E, 2) optional edge array (used for passthrough check)
        method:   'vmtk' (default)

    Returns:
        cl_nodes:      (N, 3) centerline points
        cl_edges:      list of (i, j) connectivity
        cl_to_mesh:    {centerline_idx -> nearest_mesh_vertex_idx}
    """
    # Check if input already looks like a centerline (not a surface mesh)
    if edges is not None and len(edges) > 0:
        edge_list = [(int(e[0]), int(e[1])) for e in edges]
        if _looks_like_centerline(vertices, edge_list):
            cl_to_mesh = {i: i for i in range(len(vertices))}
            return vertices.copy(), edge_list, cl_to_mesh

    # --- Real extraction from surface mesh ---

    # Step 1: Find open boundary loops (pipe ends)
    loops = _boundary_loops_from_faces(faces)
    if len(loops) < 2:
        raise RuntimeError(
            f"Could not find ≥ 2 open boundary loops on the mesh "
            f"(found {len(loops)}). Either the mesh is fully closed "
            f"(no pipe ends) or meshing produced no clean boundaries. "
            f"Mesh has {len(vertices)} vertices and {len(faces)} faces."
        )

    # Compute centroid of each boundary loop (these are the pipe end centers)
    seed_points = [_loop_centroid(vertices, loop) for loop in loops]

    if method == "vmtk":
        cl_nodes, cl_edges = _run_vmtk_centerlines(vertices, faces, seed_points)
    else:
        raise ValueError(f"Unknown centerline extraction method: {method}")

    # Map centerline points back to nearest mesh vertices
    cl_to_mesh = _nearest_vertex_map(vertices, cl_nodes)

    return cl_nodes, cl_edges, cl_to_mesh


# ══════════════════════════════════════════════════════════════════════
# MSH FILE READER (Gmsh format)
# ══════════════════════════════════════════════════════════════════════

def read_msh_file(msh_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a Gmsh .msh file (v2 or v4) using meshio.

    Returns:
        vertices: (V, 3) mesh vertices
        faces:    (F, 3) triangle face indices
        edges:    (E, 2) edge connectivity from triangles
    """
    try:
        import meshio
    except ImportError:
        raise ImportError("meshio is required: pip install meshio")

    mesh = meshio.read(msh_path)
    vertices = mesh.points.astype(np.float64)

    # Extract triangles
    faces = np.empty((0, 3), dtype=np.int64)
    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            faces = np.vstack([faces, cell_block.data.astype(np.int64)])

    # Extract edges from triangles
    edge_set: set = set()
    for f in faces:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            edge_set.add((min(a, b), max(a, b)))
    edges = np.array(sorted(edge_set), dtype=np.int64) if edge_set else np.empty((0, 2), dtype=np.int64)

    return vertices, faces, edges


# ══════════════════════════════════════════════════════════════════════
# SINGLE-LAYER PIPE: CENTERLINE VIA CROSS-SECTION RING DETECTION
# ══════════════════════════════════════════════════════════════════════

def _mesh_adjacency(n_verts: int, faces: np.ndarray) -> Dict[int, set]:
    """Build vertex adjacency from triangle faces."""
    adj: Dict[int, set] = {i: set() for i in range(n_verts)}
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        adj[a].update([b, c])
        adj[b].update([a, c])
        adj[c].update([a, b])
    return adj


def _detect_rings(
    n_verts: int,
    faces: np.ndarray,
    vertices: np.ndarray,
    n_circumference: int = 0,
) -> List[List[int]]:
    """
    Detect cross-section rings in a single-layer tubular mesh.

    Strategy:
      1. Find boundary loops (open pipe ends)
      2. BFS from one boundary, layer by layer
      3. Each BFS layer = one cross-section ring
      4. The centroid of each ring = one centerline point

    This works perfectly for single-layer pipe meshes where each
    "row" of vertices forms a ring around the pipe.

    Returns list of rings, each ring is a list of vertex indices.
    """
    # Find boundary edges
    loops = _boundary_loops_from_faces(faces)

    if len(loops) == 0:
        raise RuntimeError("No boundary loops found. Is this a closed mesh?")

    # Get vertices in the first boundary loop (= one pipe end)
    start_verts: set = set()
    for u, v in loops[0]:
        start_verts.add(u)
        start_verts.add(v)

    # Build vertex adjacency
    adj = _mesh_adjacency(n_verts, faces)

    # BFS layer by layer
    visited: set = set()
    current_ring = sorted(start_verts)
    rings: List[List[int]] = [current_ring]
    visited.update(current_ring)

    while True:
        # Find all unvisited neighbors of the current ring
        next_ring_set: set = set()
        for v in current_ring:
            for nb in adj[v]:
                if nb not in visited:
                    next_ring_set.add(nb)

        if not next_ring_set:
            break

        next_ring = sorted(next_ring_set)
        rings.append(next_ring)
        visited.update(next_ring)
        current_ring = next_ring

    return rings


def extract_centerline_from_single_layer_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Tuple[np.ndarray, List[Tuple[int, int]], Dict[int, List[int]]]:
    """
    Extract centerline from a single-layer tubular mesh.

    Algorithm:
      1. Detect cross-section rings via BFS from a boundary loop
      2. Compute centroid of each ring → centerline point
      3. Connect consecutive centroids → centerline edges
      4. Detect junctions where rings branch (multiple next-rings)

    Returns:
        cl_nodes:      (N, 3) centerline points
        cl_edges:      list of (i, j) edges
        cl_to_mesh:    {cl_idx -> [mesh_vertex_indices_in_ring]}
    """
    rings = _detect_rings(len(vertices), faces, vertices)

    if len(rings) < 2:
        raise RuntimeError(f"Only {len(rings)} ring(s) detected. Need at least 2 for a centerline.")

    # Compute centroid of each ring
    cl_nodes = np.array([vertices[ring].mean(axis=0) for ring in rings])

    # Sequential edges
    cl_edges = [(i, i + 1) for i in range(len(rings) - 1)]

    # Mapping: cl_idx -> list of mesh vertex indices in that ring
    cl_to_mesh = {i: rings[i] for i in range(len(rings))}

    return cl_nodes, cl_edges, cl_to_mesh


# ══════════════════════════════════════════════════════════════════════
# JSON READER (for pre-extracted centerlines)
# ══════════════════════════════════════════════════════════════════════

def read_centerline_json(json_path: str) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Read a centerline graph from JSON format (nodes + edges)."""
    obj = json.loads(Path(json_path).read_text())
    nodes = np.asarray(obj["nodes"], dtype=np.float64)
    edges = [(int(e[0]), int(e[1])) for e in obj["edges"]]
    return nodes, edges


# ══════════════════════════════════════════════════════════════════════
# FULL INGEST PIPELINE
# ══════════════════════════════════════════════════════════════════════

def ingest_file(
    file_path: str,
    session_dir: Path,
) -> Dict[str, Any]:
    """
    Full ingest pipeline for a single file.

    Supports:
      - .json       -> direct centerline graph (already extracted)
      - .msh        -> Gmsh mesh -> single-layer pipe -> ring-based centerline
      - .step / .stp -> STEP via gmsh+meshio -> surface mesh -> VMTK centerline

    Returns metadata dict with paths to stored artifacts.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    session_dir.mkdir(parents=True, exist_ok=True)

    if suffix == ".json":
        nodes, edges = read_centerline_json(file_path)
        cl_to_mesh: Any = {i: [i] for i in range(len(nodes))}
        mesh_vertices = nodes
        mesh_faces = np.empty((0, 3))

    elif suffix == ".msh":
        # MSH file → read mesh → extract centerline from single-layer pipe
        mesh_vertices, mesh_faces, mesh_edges = read_msh_file(file_path)
        mesh_vertices, scale = normalize_coordinates(mesh_vertices)

        # Check if it already looks like a centerline (edge graph, not triangle mesh)
        edge_list = [(int(e[0]), int(e[1])) for e in mesh_edges]
        if _looks_like_centerline(mesh_vertices, edge_list):
            nodes = mesh_vertices
            edges = edge_list
            cl_to_mesh = {i: [i] for i in range(len(nodes))}
        else:
            # Single-layer pipe mesh → ring-based centerline extraction
            nodes, edges, cl_to_mesh = extract_centerline_from_single_layer_mesh(
                mesh_vertices, mesh_faces
            )

    elif suffix in (".step", ".stp"):
        # STEP → surface mesh → VMTK centerline
        mesh_vertices, mesh_faces, mesh_edges = read_step_to_mesh(file_path)
        mesh_vertices, scale = normalize_coordinates(mesh_vertices)

        nodes, edges, cl_to_mesh_simple = extract_centerline_from_mesh(
            mesh_vertices, mesh_faces, edges=mesh_edges, method="vmtk"
        )
        # Convert int map to list map for consistency
        cl_to_mesh = {k: [v] for k, v in cl_to_mesh_simple.items()}
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .json, .msh, .step, or .stp")

    # Store centerline graph
    graph_path = session_dir / "centerline.json"
    with open(graph_path, "w") as f:
        json.dump({
            "nodes": nodes.tolist() if isinstance(nodes, np.ndarray) else nodes,
            "edges": [[int(u), int(v)] for u, v in edges],
        }, f)

    # Store mesh + mapping (internal, includes cl_to_mesh_map for face_ids computation)
    mesh_path = session_dir / "mesh.json"
    with open(mesh_path, "w") as f:
        json.dump({
            "vertices": mesh_vertices.tolist(),
            "faces": mesh_faces.tolist() if len(mesh_faces) > 0 else [],
            "cl_to_mesh_map": {str(k): v for k, v in cl_to_mesh.items()},
        }, f)

    # Store clean surface mesh for frontend (GET /mesh/{uid}/{session_id})
    # Sanitize: replace NaN/Inf with 0.0 before writing
    clean_vertices = np.where(np.isfinite(mesh_vertices), mesh_vertices, 0.0)
    surface_path = session_dir / "mesh_surface.json"
    with open(surface_path, "w") as f:
        json.dump({
            "vertices": clean_vertices.tolist(),
            "faces": mesh_faces.tolist() if len(mesh_faces) > 0 else [],
        }, f)

    n_boundary = len(_boundary_loops_from_faces(mesh_faces)) if len(mesh_faces) > 0 else 0

    return {
        "centerline_path": str(graph_path),
        "mesh_path": str(mesh_path),
        "surface_mesh_path": str(surface_path),
        "num_cl_nodes": len(nodes),
        "num_cl_edges": len(edges),
        "num_mesh_vertices": len(mesh_vertices),
        "num_mesh_faces": len(mesh_faces),
        "num_boundary_loops": n_boundary,
        "file_type": suffix,
    }
