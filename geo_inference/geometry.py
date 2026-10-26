"""Polygon cleanup and rectangle decomposition."""
import math
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.validation import make_valid

def polygon_parts(geom):
    if geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, (MultiPolygon, GeometryCollection)):
        for g in geom.geoms:
            yield from polygon_parts(g)

def fill_small_holes(poly, max_area):
    if max_area <= 0:
        return poly
    rings = []
    for ring in poly.interiors:
        hole = Polygon(ring)
        if hole.area >= max_area:
            rings.append(ring.coords)
    return Polygon(poly.exterior.coords, rings)

def orthogonalize(poly, max_angle_deg):
    """Snap near-axis-aligned edges, retaining concave/L-shaped footprints when valid."""
    def snap_ring(coords):
        pts = list(coords)
        if len(pts) < 4:
            return pts
        out = [pts[0]]
        max_angle = math.radians(max_angle_deg)
        for x1, y1 in pts[1:]:
            x0, y0 = out[-1]
            dx, dy = x1-x0, y1-y0
            length = math.hypot(dx, dy)
            if length < 0.5:
                out.append((x1, y1))
                continue
            angle = math.atan2(dy, dx)
            nearest = round(angle / (math.pi/2)) * (math.pi/2)
            delta = abs((angle-nearest+math.pi) % (2*math.pi)-math.pi)
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
        candidate = Polygon(snap_ring(poly.exterior.coords),
                            [snap_ring(r.coords) for r in poly.interiors])
        if not candidate.is_valid:
            candidate = make_valid(candidate)
        parts = list(polygon_parts(candidate))
        # If snapping creates invalid geometry, fall back to original footprint.
        if not parts:
            return poly
        best = max(parts, key=lambda p: p.area)
        if abs(best.area-poly.area) > max(poly.area * 0.25, 2.0):
            return poly
        return best
    except Exception:
        return poly

def rotated_rectangle(geom):
    """Minimum-area rectangle aligned to the dominant orientation of this geometry."""
    return geom.minimum_rotated_rectangle

def rectangle_waste(geom, rect):
    if rect.is_empty or rect.area <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - geom.intersection(rect).area / rect.area))

def _dominant_angle(geom):
    """Return the angle of the longest edge of the minimum rotated rectangle."""
    rect = geom.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)[:-1]
    edges = []
    for i in range(4):
        a, b = coords[i], coords[(i + 1) % 4]
        dx, dy = b[0] - a[0], b[1] - a[1]
        edges.append((math.hypot(dx, dy), math.degrees(math.atan2(dy, dx))))
    return max(edges, key=lambda item: item[0])[1] if edges else 0.0

def _candidate_cuts(geom):
    """Candidate axis-aligned cuts through polygon vertex coordinates."""
    from shapely.geometry import LineString
    minx, miny, maxx, maxy = geom.bounds
    span = max(maxx - minx, maxy - miny, 1.0) * 2.0
    xs, ys = set(), set()
    for poly in polygon_parts(geom):
        for x, y in list(poly.exterior.coords)[:-1]:
            xs.add(x)
            ys.add(y)
    cuts = []
    # Cutting at concave-corner coordinates is especially effective for L/T shapes.
    for x in sorted(xs):
        if minx + 1e-7 < x < maxx - 1e-7:
            cuts.append(LineString([(x, miny - span), (x, maxy + span)]))
    for y in sorted(ys):
        if miny + 1e-7 < y < maxy - 1e-7:
            cuts.append(LineString([(minx - span, y), (maxx + span, y)]))
    return cuts

def _best_split(geom, min_part_area):
    """Find a cut that most reduces rectangle waste without producing tiny slivers."""
    from shapely.ops import split
    current_rect = geom.minimum_rotated_rectangle
    current_waste_area = max(0.0, current_rect.area - geom.area)
    best = None
    best_score = current_waste_area

    for line in _candidate_cuts(geom):
        try:
            pieces = [p for p in polygon_parts(split(geom, line))
                      if p.area >= min_part_area]
        except Exception:
            continue
        if len(pieces) < 2:
            continue
        # Do not silently lose meaningful pieces during a split.
        if sum(p.area for p in pieces) < geom.area * 0.995:
            continue
        rect_area_sum = sum(p.minimum_rotated_rectangle.area for p in pieces)
        waste_sum = max(0.0, rect_area_sum - sum(p.area for p in pieces))
        # Slightly penalize extra pieces, but prefer a substantial waste reduction.
        score = waste_sum + 0.015 * geom.area * (len(pieces) - 1)
        if score < best_score - max(1e-8, geom.area * 1e-6):
            best_score = score
            best = pieces
    return best

def decompose_building(poly, waste_limit, max_parts, min_part_area):
    """
    Greedily decompose a footprint into rotated rectangles.

    The footprint is first rotated to its dominant orientation. Candidate cuts are
    placed at the footprint's vertex coordinates (including concave corners), rather
    than blindly bisecting the bounding box. Each proposed split is scored by the
    total area outside the component rectangles. This makes L-shaped buildings much
    more likely to split along their arms.
    """
    from shapely.affinity import rotate

    angle = _dominant_angle(poly)
    aligned = rotate(poly, -angle, origin="centroid", use_radians=False)

    parts = [aligned]
    while len(parts) < max_parts:
        # Split the part whose rectangle wastes the most area, but only if its waste
        # ratio is above the configured limit.
        candidates = []
        for i, part in enumerate(parts):
            rect = part.minimum_rotated_rectangle
            waste = rectangle_waste(part, rect)
            if waste > waste_limit and part.area >= 2 * min_part_area:
                candidates.append((waste * part.area, i, part))
        if not candidates:
            break

        _, index, target = max(candidates, key=lambda item: item[0])
        split_parts = _best_split(target, min_part_area)
        if not split_parts:
            # If no useful cut exists, leave this part as-is instead of looping.
            parts.pop(index)
            parts.append(target)
            # Mark it as unsplittable for this pass.
            target_area = target.area
            parts = sorted(parts, key=lambda p: p.area, reverse=True)
            # A guard against repeated attempts on an unsplittable shape.
            if len(parts) == 1:
                break
            # Continue only if another part is a plausible split candidate.
            remaining = [p for p in parts if p is not target and p.area > target_area * 1e-9]
            if not remaining:
                break
            # Temporarily stop if all parts are still unsplittable; the loop is bounded.
            if all(rectangle_waste(p, p.minimum_rotated_rectangle) <= waste_limit
                   for p in remaining):
                break
            continue

        parts.pop(index)
        parts.extend(split_parts)
        parts.sort(key=lambda p: p.area, reverse=True)

    # Convert each aligned component to a rectangle, then rotate back to map coordinates.
    result = []
    for part in parts:
        rect_aligned = part.minimum_rotated_rectangle
        rect_map = rotate(rect_aligned, angle, origin=poly.centroid, use_radians=False)
        waste = rectangle_waste(part, rect_aligned)
        if rect_map.area >= min_part_area:
            result.append((rect_map, waste))
    return result or [(poly.minimum_rotated_rectangle,
                       rectangle_waste(poly, poly.minimum_rotated_rectangle))]
