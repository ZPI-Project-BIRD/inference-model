from __future__ import annotations

from collections.abc import Iterator
import math
from typing import Sequence

import numpy as np
from rasterio.features import shapes
from shapely.geometry import Polygon, MultiPolygon, shape
from shapely.geometry.base import BaseGeometry


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------

_EPS = 1e-9


def count_vertices(geom: BaseGeometry) -> int:
    """Count ring coordinates (including each ring's closing coordinate)."""
    if geom.geom_type == "Polygon":
        return len(geom.exterior.coords) + sum(
            len(ring.coords) for ring in geom.interiors
        )
    if geom.geom_type == "MultiPolygon":
        return sum(count_vertices(part) for part in geom.geoms)
    return 0


def normalize_angle(angle: float) -> float:
    """Normalize an unoriented line angle to [0, pi)."""
    return angle % math.pi


def angle_difference(a: float, b: float) -> float:
    """Smallest angular difference between unoriented lines, in [0, pi/2]."""
    difference = abs(normalize_angle(a) - normalize_angle(b))
    return min(difference, math.pi - difference)


def edge_angle(p1: Sequence[float], p2: Sequence[float]) -> float:
    return normalize_angle(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))


def edge_length(p1: Sequence[float], p2: Sequence[float]) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def interior_angle(prev, current, nxt) -> float:
    """Smaller angle between the two rays from current (0 to pi)."""
    v1 = (prev[0] - current[0], prev[1] - current[1])
    v2 = (nxt[0] - current[0], nxt[1] - current[1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 < _EPS or n2 < _EPS:
        return math.pi
    cosine = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return math.acos(max(-1.0, min(1.0, cosine)))


def _ring_coords(ring) -> list[tuple[float, float]]:
    """Return unique ring vertices without the repeated closing coordinate."""
    coords = [(float(x), float(y)) for x, y, *_ in ring.coords]
    if len(coords) > 1 and math.dist(coords[0], coords[-1]) < _EPS:
        coords.pop()
    return coords


def _remove_consecutive_duplicates(
    coords: list[tuple[float, float]],
    eps: float = _EPS,
) -> list[tuple[float, float]]:
    out = []
    for p in coords:
        p = (float(p[0]), float(p[1]))
        if not out or math.dist(out[-1], p) > eps:
            out.append(p)
    if len(out) > 1 and math.dist(out[0], out[-1]) <= eps:
        out.pop()
    return out


def _safe_polygon(coords, holes=None) -> Polygon | None:
    """Construct a polygon, rejecting invalid/collapsed candidates."""
    coords = _remove_consecutive_duplicates(list(coords))
    if len(coords) < 3:
        return None
    try:
        candidate = Polygon(coords, holes=holes)
    except (TypeError, ValueError):
        return None
    if candidate.is_empty or candidate.area <= _EPS or not candidate.is_valid:
        return None
    return candidate


# ---------------------------------------------------------------------------
# Area-dependent parameters
# ---------------------------------------------------------------------------

def get_parameters(area: float) -> dict[str, float]:
    """
    Area-dependent starting values reported for the paper's regularization.

    Requires a projected CRS with linear units in metres and area in m².
    """
    if area < 200:
        return {"D": 0.3, "Td": 0.5, "W": 6.0, "S": 20.0, "d": 0.8}
    return {"D": 1.0, "Td": 0.9, "W": 8.0, "S": 20.0, "d": 1.2}


# ---------------------------------------------------------------------------
# Initial simplification
# ---------------------------------------------------------------------------

def initial_polygon(polygon: Polygon, tolerance: float) -> BaseGeometry:
    """Douglas-Peucker simplification with topology preservation."""
    simplified = polygon.simplify(tolerance, preserve_topology=True)
    if (
        simplified.is_empty
        or simplified.area <= _EPS
        or not simplified.is_valid
    ):
        return polygon
    return simplified


# ---------------------------------------------------------------------------
# Coarse adjustment
# ---------------------------------------------------------------------------

def _remove_short_edges(
    coords: list[tuple[float, float]],
    threshold: float,
    min_vertices: int = 3,
) -> list[tuple[float, float]]:
    """
    Remove a vertex only when its outgoing edge is short, and only if the
    resulting ring remains a valid polygon. This avoids the repeated
    indiscriminate vertex deletion that can create indents.
    """
    coords = _remove_consecutive_duplicates(coords)
    changed = True

    while changed and len(coords) > min_vertices:
        changed = False
        n = len(coords)

        # Consider shortest edges first.
        edge_indices = sorted(
            range(n),
            key=lambda i: edge_length(coords[i], coords[(i + 1) % n]),
        )

        for i in edge_indices:
            if edge_length(coords[i], coords[(i + 1) % n]) >= threshold:
                break

            # Remove the endpoint at i+1, retaining vertex i.
            remove_index = (i + 1) % n
            candidate_coords = coords[:remove_index] + coords[remove_index + 1:]

            if len(candidate_coords) < min_vertices:
                continue

            candidate = _safe_polygon(candidate_coords)
            if candidate is not None:
                coords = candidate_coords
                changed = True
                break

    return coords


def _remove_extreme_angle_vertices(
    coords: list[tuple[float, float]],
    min_angle_deg: float = 30.0,
    max_angle_deg: float = 170.0,
    min_vertices: int = 3,
) -> list[tuple[float, float]]:
    """
    Conservatively remove extreme-angle vertices only if the replacement
    polygon is valid and the area change is modest. Concave corners are
    not classified using a signed interior angle here, so this is deliberately
    conservative; callers may disable it for U-shaped or highly concave
    footprints.
    """
    coords = _remove_consecutive_duplicates(coords)
    changed = True
    min_angle = math.radians(min_angle_deg)
    max_angle = math.radians(max_angle_deg)

    while changed and len(coords) > min_vertices:
        changed = False
        n = len(coords)

        for i in range(n):
            prev, cur, nxt = coords[(i - 1) % n], coords[i], coords[(i + 1) % n]
            angle = interior_angle(prev, cur, nxt)
            if min_angle <= angle <= max_angle:
                continue

            candidate_coords = coords[:i] + coords[i + 1:]
            if len(candidate_coords) < min_vertices:
                continue

            old_poly = _safe_polygon(coords)
            new_poly = _safe_polygon(candidate_coords)
            if old_poly is None or new_poly is None:
                continue

            relative_area_change = abs(new_poly.area - old_poly.area) / max(old_poly.area, _EPS)
            if relative_area_change <= 0.03:
                coords = candidate_coords
                changed = True
                break

    return coords


def coarse_adjust_polygon(polygon: Polygon) -> BaseGeometry | None:
    """
    Coarse adjustment with conservative topology checks.

    The original polygon's holes are retained. If an exterior-ring edit would
    invalidate the polygon, the unedited polygon is returned instead.
    """
    if polygon.is_empty or polygon.area <= _EPS:
        return None

    params = get_parameters(polygon.area)
    if polygon.area < params["S"]:
        return None

    exterior = _ring_coords(polygon.exterior)
    if len(exterior) < 3:
        return None

    # Avoid destructive coarse edits to strongly concave shapes such as U's.
    # Short-edge removal is attempted, but every deletion must remain valid.
    exterior = _remove_short_edges(exterior, params["Td"])
    candidate = _safe_polygon(
        exterior,
        holes=[_ring_coords(ring) for ring in polygon.interiors] or None,
    )
    if candidate is None:
        return polygon

    # Only simplify extreme angles if it does not damage a concave footprint.
    # This operation is intentionally skipped for polygons with reflex corners.
    if not _has_reflex_corner(exterior):
        cleaned = _remove_extreme_angle_vertices(exterior)
        candidate2 = _safe_polygon(
            cleaned,
            holes=[_ring_coords(ring) for ring in polygon.interiors] or None,
        )
        if candidate2 is not None:
            candidate = candidate2

    return candidate


def _has_reflex_corner(coords: list[tuple[float, float]]) -> bool:
    """Detect whether a ring has a reflex corner, using signed turn angles."""
    if len(coords) < 4:
        return False

    # Signed area gives ring orientation.
    signed_area = sum(
        coords[i][0] * coords[(i + 1) % len(coords)][1]
        - coords[(i + 1) % len(coords)][0] * coords[i][1]
        for i in range(len(coords))
    ) / 2.0
    orientation = 1.0 if signed_area >= 0 else -1.0

    for i, current in enumerate(coords):
        prev, nxt = coords[i - 1], coords[(i + 1) % len(coords)]
        v1 = (current[0] - prev[0], current[1] - prev[1])
        v2 = (nxt[0] - current[0], nxt[1] - current[1])
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if orientation * cross < -1e-8:
            return True
    return False


# ---------------------------------------------------------------------------
# Dominant-direction detection
# ---------------------------------------------------------------------------

def find_main_directions(
    coords: list[tuple[float, float]],
    long_edge_threshold: float,
    direction_threshold: float = math.pi / 12,
    max_directions: int = 4,
) -> list[float]:
    """
    Estimate dominant undirected orientations from long edges.

    Edges are sorted by length and clustered using doubled-angle circular
    means, so angles near 0 and pi are treated as the same orientation.
    Directions are returned in radians in [0, pi).
    """
    coords = _remove_consecutive_duplicates(coords)
    n = len(coords)
    edges = []

    for i in range(n):
        p1, p2 = coords[i], coords[(i + 1) % n]
        length = edge_length(p1, p2)
        if length >= long_edge_threshold and length > _EPS:
            edges.append((length, edge_angle(p1, p2)))

    if not edges:
        # Fallback: use the longest edges if no edge passes W.
        for i in range(n):
            p1, p2 = coords[i], coords[(i + 1) % n]
            length = edge_length(p1, p2)
            if length > _EPS:
                edges.append((length, edge_angle(p1, p2)))

    if not edges:
        return []

    edges.sort(key=lambda item: item[0], reverse=True)
    clusters: list[list[tuple[float, float]]] = []
    tolerance = direction_threshold

    for length, angle in edges:
        best_index = None
        best_diff = float("inf")

        for index, cluster in enumerate(clusters):
            sin_sum = sum(w * math.sin(2 * a) for w, a in cluster)
            cos_sum = sum(w * math.cos(2 * a) for w, a in cluster)
            mean_angle = normalize_angle(0.5 * math.atan2(sin_sum, cos_sum))
            diff = angle_difference(angle, mean_angle)

            if diff <= tolerance and diff < best_diff:
                best_index, best_diff = index, diff

        if best_index is None:
            if len(clusters) < max_directions:
                clusters.append([(length, angle)])
        else:
            clusters[best_index].append((length, angle))

    directions = []
    for cluster in clusters:
        sin_sum = sum(w * math.sin(2 * a) for w, a in cluster)
        cos_sum = sum(w * math.cos(2 * a) for w, a in cluster)
        directions.append(normalize_angle(0.5 * math.atan2(sin_sum, cos_sum)))

    # Keep directions supported by the most total edge length first.
    directions_with_support = []
    for direction in directions:
        support = sum(
            length for length, angle in edges
            if angle_difference(angle, direction) <= tolerance
        )
        directions_with_support.append((support, direction))

    directions_with_support.sort(reverse=True)
    return [direction for _, direction in directions_with_support]


def nearest_main_direction(
    angle: float,
    directions: list[float],
    include_perpendicular: bool = True,
) -> float:
    """Return the closest detected direction (optionally including normals)."""
    candidates = list(directions)
    if include_perpendicular:
        candidates.extend(normalize_angle(d + math.pi / 2) for d in directions)
    return min(candidates, key=lambda candidate: angle_difference(angle, candidate))


# ---------------------------------------------------------------------------
# Fine adjustment and reconstruction
# ---------------------------------------------------------------------------

def _line_intersection(line1, line2):
    """Intersect lines represented as (midpoint_x_y, angle); None if parallel."""
    (x1, y1), a1 = line1
    (x2, y2), a2 = line2
    dx1, dy1 = math.cos(a1), math.sin(a1)
    dx2, dy2 = math.cos(a2), math.sin(a2)
    determinant = dx1 * dy2 - dy1 * dx2

    if abs(determinant) < 1e-8:
        return None

    t = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / determinant
    return (x1 + t * dx1, y1 + t * dy1)


def _distance_point_to_line(point, line_point, angle) -> float:
    dx, dy = math.cos(angle), math.sin(angle)
    return abs((point[0] - line_point[0]) * dy - (point[1] - line_point[1]) * dx)


def intersect_adjusted_edges(
    coords: list[tuple[float, float]],
    directions: list[float],
    long_edge_threshold: float,
    short_edge_threshold: float = math.pi / 4,
    long_angle_tolerance: float = math.pi / 12,
    max_vertex_shift: float = 2.0,
) -> BaseGeometry | None:
    """
    Align reliable edges to nearby dominant directions and reconstruct corners
    from adjacent adjusted lines. Unstable intersections fall back to the
    original vertex. Invalid or excessively changed candidates are rejected.
    """
    coords = _remove_consecutive_duplicates(coords)
    n = len(coords)
    if n < 3 or not directions:
        return None

    original = Polygon(coords)
    if original.is_empty or not original.is_valid:
        return None

    adjusted_lines = []
    for i in range(n):
        p1, p2 = coords[i], coords[(i + 1) % n]
        length = edge_length(p1, p2)
        if length < _EPS:
            return None

        angle = edge_angle(p1, p2)
        target = nearest_main_direction(angle, directions)
        difference = angle_difference(angle, target)

        if length >= long_edge_threshold:
            if difference <= long_angle_tolerance:
                angle = target
        else:
            if difference <= short_edge_threshold:
                angle = target

        midpoint = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        adjusted_lines.append((midpoint, angle))

    new_coords = []
    for i in range(n):
        previous_line = adjusted_lines[(i - 1) % n]
        current_line = adjusted_lines[i]
        intersection = _line_intersection(previous_line, current_line)
        original_vertex = coords[i]

        if intersection is None:
            candidate = original_vertex
        elif math.dist(intersection, original_vertex) > max_vertex_shift:
            candidate = original_vertex
        else:
            candidate = intersection

        new_coords.append(candidate)

    candidate = _safe_polygon(new_coords)
    if candidate is None:
        return None

    # Prevent regularization from dramatically changing area or footprint.
    area_change = abs(candidate.area - original.area) / max(original.area, _EPS)
    symmetric_change = candidate.symmetric_difference(original).area / max(original.area, _EPS)

    if area_change > 0.25 or symmetric_change > 0.35:
        return None

    return candidate


def _merge_nearby_parallel_edges(
    polygon: Polygon,
    distance_threshold: float,
    angle_tolerance_deg: float = 8.0,
    min_edge_length: float = 3.0,
) -> Polygon:
    """
    Conservatively merge redundant, nearly collinear consecutive edges.

    This is a safe local reconstruction step, not a global operation that
    moves distinct parallel walls onto one line. Global merging can collapse
    the arms or opening of a U-shaped building, so it is intentionally avoided.

    A vertex is removed only when its adjacent edges are nearly collinear,
    the shorter edge is small, and the resulting polygon remains valid with
    limited area change.
    """
    if (
        distance_threshold <= 0
        or polygon.is_empty
        or not polygon.is_valid
        or polygon.geom_type != "Polygon"
    ):
        return polygon

    coords = _ring_coords(polygon.exterior)
    if len(coords) < 4:
        return polygon

    angle_tol = math.radians(angle_tolerance_deg)
    changed = True

    while changed and len(coords) > 3:
        changed = False
        n = len(coords)

        for i in range(n):
            prev = coords[(i - 1) % n]
            cur = coords[i]
            nxt = coords[(i + 1) % n]

            len1 = edge_length(prev, cur)
            len2 = edge_length(cur, nxt)
            if min(len1, len2) < _EPS:
                continue

            a1 = edge_angle(prev, cur)
            a2 = edge_angle(cur, nxt)
            if angle_difference(a1, a2) > angle_tol:
                continue

            # A vertex is redundant when it lies close to the line joining
            # its neighbors. Only merge if at least one segment is relatively
            # short; do not flatten two substantial building walls.
            baseline = edge_length(prev, nxt)
            if baseline < _EPS:
                continue

            deviation = _distance_point_to_line(cur, prev, edge_angle(prev, nxt))
            if deviation > distance_threshold:
                continue

            if min(len1, len2) > min_edge_length:
                continue

            candidate_coords = coords[:i] + coords[i + 1:]
            candidate = _safe_polygon(candidate_coords)
            if candidate is None:
                continue

            area_change = abs(candidate.area - polygon.area) / max(polygon.area, _EPS)
            if area_change > 0.02:
                continue

            # Preserve holes by rebuilding with the original interior rings.
            if polygon.interiors:
                with_holes = Polygon(
                    _ring_coords(candidate.exterior),
                    holes=[_ring_coords(ring) for ring in polygon.interiors],
                )
                if not with_holes.is_valid or with_holes.area <= _EPS:
                    continue
                candidate = with_holes

            polygon = candidate
            coords = _ring_coords(polygon.exterior)
            changed = True
            break

    return polygon


def regularize_polygon(
    polygon: Polygon,
    *,
    merge_parallel_lines: bool = True,
) -> BaseGeometry | None:
    """Run simplification, conservative coarse adjustment and fine adjustment."""
    if polygon.is_empty or polygon.area <= _EPS:
        return None

    params = get_parameters(polygon.area)
    if polygon.area < params["S"]:
        return None

    simplified = initial_polygon(polygon, params["D"])
    if simplified.geom_type != "Polygon":
        return simplified

    coarse = coarse_adjust_polygon(simplified)
    if coarse is None:
        return None
    if coarse.geom_type != "Polygon":
        return coarse

    coords = _ring_coords(coarse.exterior)
    directions = find_main_directions(coords, params["W"])
    if not directions:
        return coarse

    fine = intersect_adjusted_edges(
        coords,
        directions,
        long_edge_threshold=params["W"],
        short_edge_threshold=math.radians(60),
        long_angle_tolerance=math.radians(30),
        max_vertex_shift=max(3.0, 2 * params["d"]),
    )
    if fine is None:
        return coarse

    # Preserve interior rings. If fine adjustment would discard holes, reject it.
    if len(coarse.interiors) > 0:
        fine_with_holes = Polygon(
            _ring_coords(fine.exterior),
            holes=[_ring_coords(ring) for ring in coarse.interiors],
        )
        if fine_with_holes.is_valid and fine_with_holes.area > _EPS:
            fine = fine_with_holes
        else:
            return coarse

    if merge_parallel_lines:
        fine = _merge_nearby_parallel_edges(
            fine,
            distance_threshold=params["d"],
        )

    result = Polygon(fine)
    result = remove_near_collinear_vertices(
        result,
        angle_tolerance=30.0,
    )

    return result if result.is_valid and result.area > _EPS else coarse


# ---------------------------------------------------------------------------
# Raster polygonization
# ---------------------------------------------------------------------------

def polygonize_mask(
    mask: np.ndarray,
    transform,
    connectivity: int = 8,
    apply_regularization: bool = True,
    min_area: float | None = None,
) -> Iterator[BaseGeometry]:
    """
    Polygonize a 2D mask and optionally regularize each polygon.

    Parameters
    ----------
    mask:
        2D NumPy mask. Nonzero values are treated as foreground.
    transform:
        Rasterio affine transform mapping pixel coordinates to map coordinates.
    connectivity:
        4 or 8 pixel connectivity.
    apply_regularization:
        Apply the regularization pipeline to each polygon.
    min_area:
        Optional output-area filter in the geometry CRS units squared.
        If omitted, the paper-style area threshold is applied by regularize_polygon.

    Notes
    -----
    For paper-style metric thresholds, transform must use a projected CRS in
    metres. A geographic CRS (longitude/latitude) is not suitable.
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a NumPy array")
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be either 4 or 8")

    binary_mask = mask.astype(bool, copy=False)
    if not binary_mask.any():
        return

    vertices_before_total = 0
    vertices_after_total = 0

    for geometry_mapping, value in shapes(
        binary_mask.astype(np.uint8),
        mask=binary_mask,
        transform=transform,
        connectivity=connectivity,
    ):
        if value != 1:
            continue

        geom = shape(geometry_mapping)
        parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]

        for part in parts:
            if part.geom_type != "Polygon" or part.is_empty or part.area <= _EPS:
                continue
            if min_area is not None and part.area < min_area:
                continue

            vertices_before_total += count_vertices(part)

            result = regularize_polygon(part) if apply_regularization else part
            if result is None or result.is_empty:
                continue

            results = list(result.geoms) if result.geom_type == "MultiPolygon" else [result]
            for final_polygon in results:
                if final_polygon.geom_type != "Polygon":
                    continue
                if final_polygon.is_empty or final_polygon.area <= _EPS or not final_polygon.is_valid:
                    continue
                if min_area is not None and final_polygon.area < min_area:
                    continue

                vertices_after_total += count_vertices(final_polygon)
                yield final_polygon

    print(f"Vertices before regularization: {vertices_before_total:,}")
    print(f"Vertices after regularization:  {vertices_after_total:,}")
    if vertices_before_total:
        reduction = 100.0 * (1.0 - vertices_after_total / vertices_before_total)
        print(f"Vertex reduction: {reduction:.1f}%")


def remove_near_collinear_vertices(
    polygon,
    angle_tolerance=12.0,
    min_edge_length=0.0,
):
    """
    Remove redundant vertices where adjoining edges are almost
    collinear (interior angle close to 180 degrees).

    angle_tolerance:
        Maximum deviation from 180 degrees.

    min_edge_length:
        Optional minimum length for both adjoining edges.
    """
    if polygon.is_empty or polygon.geom_type != "Polygon":
        return polygon

    coords = list(polygon.exterior.coords)[:-1]

    if len(coords) < 4:
        return polygon

    changed = True

    while changed and len(coords) > 3:
        changed = False
        cleaned = []
        n = len(coords)

        for i in range(n):
            prev = coords[(i - 1) % n]
            curr = coords[i]
            nxt = coords[(i + 1) % n]

            v1 = (curr[0] - prev[0], curr[1] - prev[1])
            v2 = (nxt[0] - curr[0], nxt[1] - curr[1])

            len1 = math.hypot(*v1)
            len2 = math.hypot(*v2)

            if len1 < 1e-9 or len2 < 1e-9:
                changed = True
                continue

            if min(len1, len2) < min_edge_length:
                cleaned.append(curr)
                continue

            dot = v1[0] * v2[0] + v1[1] * v2[1]
            cos_angle = max(-1.0, min(1.0, dot / (len1 * len2)))
            deviation = math.degrees(math.acos(cos_angle))

            # 0 degrees means straight continuation.
            if deviation <= angle_tolerance:
                changed = True
                continue

            cleaned.append(curr)

        if len(cleaned) < 3:
            break

        coords = cleaned

    result = Polygon(coords)

    if result.is_valid and not result.is_empty:
        return result

    return polygon