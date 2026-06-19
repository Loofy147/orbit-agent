import re

with open('main.py', 'r') as f:
    content = f.read()

# The regex replace matched more than it should or duplicated
pattern = r'def find_fleet_target_and_arrival\(.*?\):.*?return None, None\n.*?return None, None'
replacement = """def find_fleet_target_and_arrival(fleet, planets, initial_by_id, angular_velocity, comets, step, max_steps=80):
    fx, fy = fleet.x, fleet.y
    speed = fleet_speed(fleet.ships)
    dx = speed * math.cos(fleet.angle)
    dy = speed * math.sin(fleet.angle)

    # Cache radii squared
    p_data = [(p, p.radius**2) for p in planets]

    for t in range(1, max_steps):
        fnx, fny = fx + dx, fy + dy
        # Optimized sun check (no safety for simulation speed)
        if point_to_segment_dist_sq(50.0, 50.0, fx, fy, fnx, fny) < 100.0:
            return None, None

        # Segment bounding box
        min_x, max_x = (fx, fnx) if fx < fnx else (fnx, fx)
        min_y, max_y = (fy, fny) if fy < fny else (fny, fy)

        for p, r_sq in p_data:
            if p.id == fleet.from_planet_id and t <= 2: continue

            pos = predict_position(p.id, p, initial_by_id, angular_velocity, comets, step, t)
            if pos is None: continue
            px, py = pos

            # Rough bounding box filter
            if px < min_x - p.radius or px > max_x + p.radius or py < min_y - p.radius or py > max_y + p.radius:
                continue

            if point_to_segment_dist_sq(px, py, fx, fy, fnx, fny) < r_sq:
                return p.id, t

        if fnx < 0.0 or fnx > 100.0 or fny < 0.0 or fny > 100.0:
            return None, None
        fx, fy = fnx, fny
    return None, None"""

content = re.sub(pattern, replacement, content, flags=re.DOTALL)

with open('main.py', 'w') as f:
    f.write(content)
