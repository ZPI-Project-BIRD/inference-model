"""Fast polygon cleanup and greedy rectangle decomposition.

Coordinates must be in a suitable projected CRS. Returned decompositions are
lists of (rectangle_polygon, waste_ratio).
"""
from __future__ import annotations

import math
from typing import Iterator

from shapely.affinity import rotate
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiPolygon,
    Polygon,
    box,
)
from shapely.ops import split, unary_union
from shapely.validation import make_valid

_EPS = 1e-9


def polygon_parts(geom: object) -> Iterator[Polygon]:
    """Yield polygon components recursively."""
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, Polygon):
        if geom.area > _EPS:
            yield geom
    elif isinstance(geom, (MultiPolygon, GeometryCollection)):
        for part in geom.geoms:
            yield from polygon_parts(part)


def fill_small_holes(poly: Polygon, max_area: float) -> Polygon:
    """Fill interior holes smaller than max_area."""
    if max_area <= 0 or poly.is_empty or not poly.interiors:
        return poly
    kept_holes = []
    for ring in poly.interiors:
        if Polygon(ring).area >= max_area:
            kept_holes.append(list(ring.coords))
    candidate = Polygon(poly.exterior.coords, kept_holes)
    return candidate if candidate.is_valid else poly


def orthogonalize(poly: Polygon, max_angle_deg: float) -> Polygon:
    """Snap near-axis-aligned edges, falling back if topology/area changes too much."""
    if poly.is_empty or poly.geom_type != "Polygon" or max_angle_deg <= 0:
        return poly

    max_angle = math.radians(max_angle_deg)

    def snap_ring(coords):
        pts = list(coords)
        if len(pts) < 4:
            return pts
        out = [pts[0]]
        for x1, y1 in pts[1:]:
            x0, y0 = out[-1]
            dx, dy = x1 - x0, y1 - y0
            length = math.hypot(dx, dy)
            if length < 0.5:
                out.append((x1, y1))
                continue
            angle = math.atan2(dy, dx)
            nearest = round(angle / (math.pi / 2)) * (math.pi / 2)
            delta = abs((angle - nearest + math.pi) % (2 * math.pi) - math.pi)
            if delta <= max_angle:
                if abs(math.cos(nearest)) > 0.5:
                    out.append((x1, y0))
                else:
                    out.append((x0, y1))
            else:
                out.append((x1, y1))
        out[-1] = out[0]
        return out

    try:
        candidate = Polygon(
            snap_ring(poly.exterior.coords),
            [snap_ring(r.coords) for r in poly.interiors],
        )
        if not candidate.is_valid:
            candidate = make_valid(candidate)
        candidates = list(polygon_parts(candidate))
        if not candidates:
            return poly
        best = max(candidates, key=lambda p: p.area)
        if abs(best.area - poly.area) > max(poly.area * 0.25, 2.0):
            return poly
        # Do not silently discard significant components created by make_valid.
        if sum(p.area for p in candidates) < poly.area * 0.995:
            return poly
        return best if len(candidates) == 1 else candidate
    except Exception:
        return poly


def rotated_rectangle(geom):
    return geom.minimum_rotated_rectangle


def rectangle_waste(geom, rect=None) -> float:
    """Fraction of the enclosing rectangle not covered by the geometry."""
    if rect is None:
        rect = geom.minimum_rotated_rectangle
    if rect.is_empty or rect.area <= _EPS:
        return 1.0
    return max(0.0, min(1.0, 1.0 - geom.area / rect.area))


def _dominant_angle(geom) -> float:
    """Angle of the longest edge of the minimum rotated rectangle."""
    rect = geom.minimum_rotated_rectangle
    if rect.is_empty or rect.geom_type != "Polygon":
        return 0.0
    coords = list(rect.exterior.coords)
    best_length = 0.0
    best_angle = 0.0
    for a, b in zip(coords[:-1], coords[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if length > best_length:
            best_length = length
            best_angle = math.degrees(math.atan2(dy, dx))
    return best_angle


def _concave_vertices(poly: Polygon) -> list[tuple[float, float]]:
    """Return reflex vertices from a valid exterior ring."""
    coords = list(poly.exterior.coords)[:-1]
    n = len(coords)
    if n < 4:
        return []

    signed_area2 = sum(
        coords[i][0] * coords[(i + 1) % n][1]
        - coords[(i + 1) % n][0] * coords[i][1]
        for i in range(n)
    )
    orientation = 1.0 if signed_area2 >= 0 else -1.0
    result = []
    for i, curr in enumerate(coords):
        prev, nxt = coords[i - 1], coords[(i + 1) % n]
        ax, ay = curr[0] - prev[0], curr[1] - prev[1]
        bx, by = nxt[0] - curr[0], nxt[1] - curr[1]
        if orientation * (ax * by - ay * bx) < -1e-9:
            result.append(curr)
    return result


def _candidate_cuts(
    geom,
    *,
    max_candidates: int = 24,
    coordinate_precision: int = 7,
) -> list[LineString]:
    """Create a small deterministic set of cuts through reflex-corner axes."""
    minx, miny, maxx, maxy = geom.bounds
    span = max(maxx - minx, maxy - miny, 1.0) * 2.0
    reflex = _concave_vertices(geom) if isinstance(geom, Polygon) else []

    if not reflex:
        return []

    xs = sorted({
        round(x, coordinate_precision)
        for x, _ in reflex
        if minx + 1e-7 < x < maxx - 1e-7
    })
    ys = sorted({
        round(y, coordinate_precision)
        for _, y in reflex
        if miny + 1e-7 < y < maxy - 1e-7
    })

    # A reflex coordinate often lies exactly on the boundary of the notch.
    # A cut placed exactly there can overlap the boundary and fail to split
    # (common for U-shaped footprints). Test tiny offsets on either side too.
    offset = max(maxx - minx, maxy - miny, 1.0) * 1e-6
    x_cuts = sorted({
        value
        for x in xs
        for value in (x - offset, x, x + offset)
        if minx + 1e-7 < value < maxx - 1e-7
    })
    y_cuts = sorted({
        value
        for y in ys
        for value in (y - offset, y, y + offset)
        if miny + 1e-7 < value < maxy - 1e-7
    })
    raw = [
        LineString([(x, miny - span), (x, maxy + span)])
        for x in x_cuts
    ] + [
        LineString([(minx - span, y), (maxx + span, y)])
        for y in y_cuts
    ]

    # Bound work for unusually complex/noisy outlines. Keep cuts distributed
    # over the sorted candidate list rather than taking only one end.
    if len(raw) > max_candidates:
        if max_candidates <= 1:
            raw = raw[:1]
        else:
            indices = [
                round(i * (len(raw) - 1) / (max_candidates - 1))
                for i in range(max_candidates)
            ]
            raw = [raw[i] for i in indices]
    return raw


def _best_split(
    geom,
    min_part_area: float,
    *,
    max_candidates: int = 24,
    split_penalty: float = 0.015,
):
    """Choose a valid concave-corner cut that minimizes summed rectangle waste."""
    current_rect = geom.minimum_rotated_rectangle
    baseline_waste_area = max(0.0, current_rect.area - geom.area)
    best_parts = None
    best_score = baseline_waste_area
    area_tolerance = max(1e-8, geom.area * 1e-6)

    for line in _candidate_cuts(geom, max_candidates=max_candidates):
        try:
            pieces = [
                p for p in polygon_parts(split(geom, line))
                if p.area > _EPS
            ]
        except Exception:
            continue

        if len(pieces) < 2:
            continue

        if sum(p.area for p in pieces) < geom.area * 0.999:
            continue
        if any(p.area < min_part_area for p in pieces):
            continue

        rect_area_sum = sum(p.minimum_rotated_rectangle.area for p in pieces)
        waste_sum = max(0.0, rect_area_sum - sum(p.area for p in pieces))
        score = waste_sum + split_penalty * geom.area * (len(pieces) - 1)

        if score < best_score - area_tolerance:
            best_score = score
            best_parts = pieces

    return best_parts



def _snap_aligned_rectangles(
    rectangles,
    *,
    snap_distance: float,
    snap_angle_deg: float = 8.0,
):
    """Snap close rectangle sides to shared coordinates in the building frame.

    Rectangles are expressed in the dominant-orientation frame. Only rectangles
    whose own orientation is already close to that frame's horizontal/vertical
    axes are eligible. Boundary coordinates from different rectangles are
    clustered when they are within ``snap_distance``. The cluster mean becomes
    the shared coordinate, improving edge alignment without forcing distant
    sides together.

    Returns new aligned-frame rectangles in the same order.
    """
    if snap_distance <= 0 or len(rectangles) < 2:
        return rectangles

    angle_limit = math.radians(snap_angle_deg)
    records = []
    for idx, rect in enumerate(rectangles):
        if rect.is_empty or rect.geom_type != "Polygon":
            records.append(None)
            continue
        coords = list(rect.exterior.coords)
        edge_angles = []
        for a, b in zip(coords[:-1], coords[1:]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            if math.hypot(dx, dy) > _EPS:
                edge_angles.append(math.atan2(dy, dx))
        if not edge_angles:
            records.append(None)
            continue

        # MRRs should be rectangles; use their longest edge as orientation.
        longest = max(
            ((math.hypot(b[0]-a[0], b[1]-a[1]), math.atan2(b[1]-a[1], b[0]-a[0]))
             for a, b in zip(coords[:-1], coords[1:])),
            key=lambda item: item[0],
        )
        angle = longest[1] % (math.pi / 2)
        angle_error = min(angle, math.pi / 2 - angle)
        if angle_error > angle_limit:
            records.append(None)
            continue

        minx, miny, maxx, maxy = rect.bounds
        records.append([minx, miny, maxx, maxy])

    # Each boundary is (coordinate, rectangle index, bound slot).
    # A cluster may contain at most one boundary from a given rectangle,
    # preventing a narrow rectangle from snapping its own opposite sides.
    for axis, slots in ((0, (0, 2)), (1, (1, 3))):
        boundaries = []
        for idx, bounds in enumerate(records):
            if bounds is None:
                continue
            for slot in slots:
                boundaries.append((bounds[slot], idx, slot))
        boundaries.sort(key=lambda item: item[0])

        clusters = []
        for item in boundaries:
            value, rect_idx, slot = item
            placed = False
            for cluster in reversed(clusters):
                if value - cluster[0][0] > snap_distance:
                    break
                if all(existing[1] != rect_idx for existing in cluster):
                    cluster.append(item)
                    placed = True
                    break
            if not placed:
                clusters.append([item])

        for cluster in clusters:
            if len(cluster) < 2:
                continue
            target = sum(item[0] for item in cluster) / len(cluster)
            for _, rect_idx, slot in cluster:
                records[rect_idx][slot] = target

    snapped = []
    for rect, bounds in zip(rectangles, records):
        if bounds is None:
            snapped.append(rect)
            continue
        minx, miny, maxx, maxy = bounds
        if maxx - minx <= _EPS or maxy - miny <= _EPS:
            snapped.append(rect)
            continue
        candidate = box(minx, miny, maxx, maxy)
        if candidate.is_valid and candidate.area > _EPS:
            snapped.append(candidate)
        else:
            snapped.append(rect)
    return snapped

def decompose_building(
    poly,
    waste_limit: float = 0.15,
    max_parts: int = 8,
    min_part_area: float = 10.0,
    *,
    max_candidates: int = 24,
    snap_distance: float = 0.5,
    snap_angle_deg: float = 8.0,
    gap_close_distance: float = 0.5,
):
    """Greedily decompose a footprint into rotated rectangles.

    Returns:
        list[(merged_footprint_polygon, waste_ratio)]. The rectangles are
        dissolved into their union, so internal rectangle edges are removed
        and only the perimeter (plus any genuine holes) remains.
    """
    if poly is None or poly.is_empty or poly.area <= _EPS:
        return []
    if max_parts < 1:
        return []
    if min_part_area < 0:
        raise ValueError("min_part_area must be non-negative")
    if snap_distance < 0:
        raise ValueError("snap_distance must be non-negative")
    if snap_angle_deg < 0 or snap_angle_deg > 45:
        raise ValueError("snap_angle_deg must be between 0 and 45")
    if gap_close_distance < 0:
        raise ValueError("gap_close_distance must be non-negative")
    if not 0.0 <= waste_limit <= 1.0:
        raise ValueError("waste_limit must be between 0 and 1")

    if not isinstance(poly, Polygon):
        result = []
        for component in polygon_parts(poly):
            result.extend(
                decompose_building(
                    component, waste_limit, max_parts, min_part_area,
                    max_candidates=max_candidates,
                    snap_distance=snap_distance,
                    snap_angle_deg=snap_angle_deg,
                    gap_close_distance=gap_close_distance,
                )
            )
        return result

    original_rect = poly.minimum_rotated_rectangle
    original_waste = rectangle_waste(poly, original_rect)
    if original_waste <= waste_limit or poly.area < 2 * min_part_area:
        return [(original_rect, original_waste)]

    angle = _dominant_angle(poly)
    origin = poly.centroid
    aligned = rotate(poly, -angle, origin=origin, use_radians=False)
    aligned_rect = aligned.minimum_rotated_rectangle
    aligned_waste = rectangle_waste(aligned, aligned_rect)

    parts = [{
        "geom": aligned,
        "rect": aligned_rect,
        "waste": aligned_waste,
        "active": True,
    }]

    while len(parts) < max_parts:
        candidates = [
            (item["waste"] * item["geom"].area, idx)
            for idx, item in enumerate(parts)
            if item["active"]
            and item["waste"] > waste_limit
            and item["geom"].area >= 2 * min_part_area
        ]
        if not candidates:
            break

        _, index = max(candidates)
        target = parts[index]
        split_parts = _best_split(
            target["geom"],
            min_part_area,
            max_candidates=max_candidates,
        )
        if not split_parts:
            target["active"] = False
            continue

        parts.pop(index)
        for piece in split_parts:
            piece_rect = piece.minimum_rotated_rectangle
            piece_waste = rectangle_waste(piece, piece_rect)
            parts.append({
                "geom": piece,
                "rect": piece_rect,
                "waste": piece_waste,
                "active": piece_waste > waste_limit,
            })

        if len(parts) > max_parts:
            # A cut may produce more than two components; reject it rather
            # than exceed the caller's requested output bound.
            parts = parts[:index] + [target] + parts[index:]
            break

    aligned_rectangles = [item["rect"] for item in parts]
    snapped_rectangles = _snap_aligned_rectangles(
        aligned_rectangles,
        snap_distance=snap_distance,
        snap_angle_deg=snap_angle_deg,
    )

    rectangle_geometries = []
    total_source_area = sum(item["geom"].area for item in parts)
    total_rectangle_area = 0.0

    for item, aligned_rect in zip(parts, snapped_rectangles):
        rect_map = rotate(
            aligned_rect, angle, origin=origin, use_radians=False
        )
        if rect_map.is_valid and rect_map.area > _EPS:
            rectangle_geometries.append(rect_map)
            total_rectangle_area += rect_map.area

    if not rectangle_geometries:
        return [(original_rect, original_waste)]

    merged = unary_union(rectangle_geometries)

    # Close only small gaps between the decomposed rectangles. A positive
    # buffer joins components whose separation is up to approximately twice
    # the buffer radius; the negative buffer then restores the outer size.
    # Mitre joins preserve the rectilinear character of building corners.
    if gap_close_distance > 0 and not merged.is_empty:
        radius = gap_close_distance / 2.0
        try:
            closed = merged.buffer(radius, join_style=2).buffer(
                -radius, join_style=2
            )
            closed_parts = list(polygon_parts(closed))
            merged_parts = list(polygon_parts(merged))

            if closed_parts:
                # Reject aggressive closing if it changes too much area or
                # collapses a meaningful opening/courtyard.
                before_area = sum(part.area for part in merged_parts)
                after_area = sum(part.area for part in closed_parts)
                relative_change = abs(after_area - before_area) / max(before_area, _EPS)
                before_holes = sum(len(part.interiors) for part in merged_parts)
                after_holes = sum(len(part.interiors) for part in closed_parts)
                if relative_change <= 0.08 and after_holes >= before_holes:
                    merged = closed
        except Exception:
            # Keep the ordinary union if closing fails for an invalid geometry.
            pass

    merged_parts = list(polygon_parts(merged))
    if not merged_parts:
        return [(original_rect, original_waste)]

    waste = max(
        0.0,
        min(1.0, 1.0 - total_source_area / max(total_rectangle_area, _EPS)),
    )
    return [(part, waste) for part in merged_parts]
