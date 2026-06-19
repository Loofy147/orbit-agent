"""
orbit_agent_v3.py — Optimized and Robust Orbit Wars Agent

Key Improvements:
  1. PERF: Dynamic Horizon Scaling based on remaining steps (caps at 20-80).
  2. PERF: Integrates Overage Time Safeguards to reduce search horizon under time pressure.
  3. PERF: Cuts path-tracing limits using the dynamic horizon, saving O(N) calculations on long-range fleets.
  4. FIX: Resolves potential self-collision edge cases on slow fleet launches (t <= 2).
"""

import math
_PLANET_GEO_CACHE = {}
_TURN_POS_CACHE = {}
_COMET_IDS_TURN = set()
_COMET_LIFE_TURN = {}
from collections import defaultdict
from dataclasses import dataclass

try:
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet as EnvPlanet, Fleet as EnvFleet
except ImportError:
    EnvPlanet, EnvFleet = None, None

class Planet:
    __slots__ = ('id', 'owner', 'x', 'y', 'radius', 'ships', 'production')
    def __init__(self, id, owner, x, y, radius, ships, production):
        self.id = int(id)
        self.owner = int(owner)
        self.x = float(x)
        self.y = float(y)
        self.radius = float(radius)
        self.ships = int(ships)
        self.production = int(production)

class Fleet:
    __slots__ = ('id', 'owner', 'x', 'y', 'angle', 'from_planet_id', 'ships')
    def __init__(self, id, owner, x, y, angle, from_planet_id, ships):
        self.id = int(id)
        self.owner = int(owner)
        self.x = float(x)
        self.y = float(y)
        self.angle = float(angle)
        self.from_planet_id = int(from_planet_id)
        self.ships = int(ships)

@dataclass
class PlanetState:
    owner: int
    ships: int

_current_step = -1

# --- Geometry Utilities ---

def dist(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def point_to_segment_dist_sq(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return (px - x1)**2 + (py - y1)**2
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    if t < 0:
        return (px - x1)**2 + (py - y1)**2
    if t > 1:
        return (px - x2)**2 + (py - y2)**2
    return (px - (x1 + t * dx))**2 + (py - (y1 + t * dy))**2
def point_to_segment_distance(px, py, x1, y1, x2, y2):
    return math.sqrt(point_to_segment_dist_sq(px, py, x1, y1, x2, y2))


def segment_hits_sun(x1, y1, x2, y2, sun_radius_sq=100.0):
    return point_to_segment_dist_sq(50.0, 50.0, x1, y1, x2, y2) < sun_radius_sq



def safe_angle_and_distance(sx, sy, sr, tx, ty, tr):
    dx, dy = tx - sx, ty - sy
    d = math.hypot(dx, dy)
    if d == 0: return math.atan2(dy, dx), 0.0
    angle = math.atan2(dy, dx)
    # Start fleet just outside source radius
    lx = sx + (sr + 0.1) * (dx / d)
    ly = sy + (sr + 0.1) * (dy / d)
    if segment_hits_sun(lx, ly, tx, ty, sun_radius_sq=132.25):
        return None
    return angle, max(0.0, d - (sr + 0.1) - tr)


def fleet_speed(ships, max_speed=6.0):
    if ships <= 1:
        return 1.0
    ratio = min(1.0, max(0.0, math.log(ships) / math.log(1000.0)))
    return 1.0 + (max_speed - 1.0) * (ratio ** 1.5)

# --- Comet & Orbital Position Prediction ---

def get_comet_life(planet_id, comets):
    for g in comets:
        pids = g.get("planet_ids", [])
        if planet_id in pids:
            idx = pids.index(planet_id)
            paths = g.get("paths", [])
            path_idx = g.get("path_index", 0)
            if idx < len(paths):
                return max(0, len(paths[idx]) - path_idx)
    return 0

def predict_comet_position_at_turn(planet_id, comets, turns_ahead):
    for g in comets:
        pids = g.get("planet_ids", [])
        if planet_id in pids:
            idx = pids.index(planet_id)
            paths = g.get("paths", [])
            path_idx = g.get("path_index", 0)
            target_idx = path_idx + turns_ahead
            if idx < len(paths) and target_idx < len(paths[idx]):
                return paths[idx][target_idx][0], paths[idx][target_idx][1]
    return None


def predict_planet_position_at_step(planet, initial_by_id, angular_velocity, step):
    global _PLANET_GEO_CACHE
    if planet.id not in _PLANET_GEO_CACHE:
        init = initial_by_id.get(planet.id)
        if init is None:
            _PLANET_GEO_CACHE[planet.id] = (planet.x, planet.y, 0, 0, False)
        else:
            dx_c, dy_c = init.x - 50.0, init.y - 50.0
            orb_r = math.hypot(dx_c, dy_c)
            if orb_r + init.radius >= 50.0:
                _PLANET_GEO_CACHE[planet.id] = (init.x, init.y, 0, 0, False)
            else:
                theta = math.atan2(dy_c, dx_c)
                _PLANET_GEO_CACHE[planet.id] = (50.0, 50.0, orb_r, theta, True)

    cx, cy, r, t0, moves = _PLANET_GEO_CACHE[planet.id]
    if not moves:
        return cx, cy
    theta = t0 + angular_velocity * step
    return cx + r * math.cos(theta), cy + r * math.sin(theta)



def predict_position(planet_id, planet, initial_by_id, angular_velocity, comets, current_step, turns_ahead):
    global _TURN_POS_CACHE
    cache_key = planet_id * 1000 + turns_ahead
    if cache_key in _TURN_POS_CACHE:
        return _TURN_POS_CACHE[cache_key]

    if turns_ahead == 0:
        return planet.x, planet.y

    global _COMET_IDS_TURN
    is_comet = planet_id in _COMET_IDS_TURN
    if is_comet:
        res = predict_comet_position_at_turn(planet_id, comets, turns_ahead)
    else:
        res = predict_planet_position_at_step(planet, initial_by_id, angular_velocity, current_step + turns_ahead)

    _TURN_POS_CACHE[cache_key] = res
    return res


# --- Intercept Solving (Optimized) ---

def find_intercept_angle_and_time(src, target, initial_by_id, angular_velocity, comets, current_step, ships_to_send, horizon=80):
    speed = fleet_speed(ships_to_send)
    d_curr = dist(src.x, src.y, target.x, target.y)
    t_est = max(1, int(math.ceil((d_curr - src.radius - target.radius) / speed)))
    if t_est > horizon:
        return None

    for _ in range(10):
        if t_est > horizon:
            return None
        pos = predict_position(target.id, target, initial_by_id, angular_velocity, comets, current_step, t_est)
        if pos is None:
            return None
        geom = safe_angle_and_distance(src.x, src.y, src.radius, pos[0], pos[1], target.radius)
        if geom is None:
            return None
        t_new = max(1, int(math.ceil(geom[1] / speed)))
        if t_new == t_est:
            return geom[0], t_new
        t_est = t_new
    return None

def find_ships_for_arrival_turn(src, target, target_turn, initial_by_id, angular_velocity, comets, current_step, max_ships_available):
    if target_turn < 1:
        return None
    pos = predict_position(target.id, target, initial_by_id, angular_velocity, comets, current_step, target_turn)
    if pos is None:
        return None
    geom = safe_angle_and_distance(src.x, src.y, src.radius, pos[0], pos[1], target.radius)
    if geom is None:
        return None
    angle, hit_dist = geom
    low, high, best_s = 1, int(max_ships_available), None
    while low <= high:
        mid = (low + high) // 2
        act_t = max(1, int(math.ceil(hit_dist / fleet_speed(mid))))
        if act_t == target_turn:
            best_s = mid
            high = mid - 1
        elif act_t > target_turn:
            low = mid + 1
        else:
            high = mid - 1
    return (angle, best_s) if best_s is not None else None


def find_fleet_target_and_arrival(fleet, planets, initial_by_id, angular_velocity, comets, step, max_steps=80):
    fx, fy = fleet.x, fleet.y
    speed = fleet_speed(fleet.ships)
    cos_a = math.cos(fleet.angle)
    sin_a = math.sin(fleet.angle)
    dx, dy = speed * cos_a, speed * sin_a

    p_data = []
    for p in planets:
        p_data.append((p.id, p, p.radius, p.radius**2))

    for t in range(1, max_steps):
        fnx, fny = fx + dx, fy + dy

        if fx < fnx: min_x, max_x = fx, fnx
        else: min_x, max_x = fnx, fx
        if fy < fny: min_y, max_y = fy, fny
        else: min_y, max_y = fny, fy

        # Sun collision (radius 10, center 50,50). Bounding box check first.
        if min_x < 61.0 and max_x > 39.0 and min_y < 61.0 and max_y > 39.0:
             if point_to_segment_dist_sq(50.0, 50.0, fx, fy, fnx, fny) < 110.25:
                 return None, None

        for p_id, p_obj, p_r, r_sq in p_data:
            if p_id == fleet.from_planet_id and t <= 2: continue
            pos = predict_position(p_id, p_obj, initial_by_id, angular_velocity, comets, step, t)
            if pos is None: continue
            px, py = pos
            if px < min_x - p_r or px > max_x + p_r or py < min_y - p_r or py > max_y + p_r:
                continue
            if point_to_segment_dist_sq(px, py, fx, fy, fnx, fny) < r_sq:
                return p_id, t

        if fnx < 0.0 or fnx > 100.0 or fny < 0.0 or fny > 100.0:
            return None, None
        fx, fy = fnx, fny
    return None, None

def precompute_fleet_arrivals(fleets, planets, initial_by_id, angular_velocity, comets, step, horizon):
    """Returns {fleet_id: (target_id, arr_turns, owner, ships)} — computed once per turn."""
    result = {}
    for f in fleets:
        tid, arr = find_fleet_target_and_arrival(f, planets, initial_by_id, angular_velocity, comets, step, max_steps=horizon)
        result[f.id] = (tid, arr, f.owner, f.ships)
    return result

# --- Combat Resolution ---

def resolve_combat(planet_owner, garrison, arriving_fleets):
    if not arriving_fleets:
        return planet_owner, garrison
    sums = defaultdict(int)
    for owner, ships in arriving_fleets:
        sums[owner] += ships
    sa = sorted(sums.items(), key=lambda x: x[1], reverse=True)
    o1, s1 = sa[0]
    o2, s2 = sa[1] if len(sa) > 1 else (-1, 0)
    if s1 == s2:
        surv_owner, surv_ships = -1, 0
    else:
        surv_owner, surv_ships = o1, s1 - s2
    if surv_ships > 0:
        if surv_owner == planet_owner:
            garrison += surv_ships
        else:
            if surv_ships > garrison:
                planet_owner, garrison = surv_owner, surv_ships - garrison
            else:
                garrison -= surv_ships
    return planet_owner, garrison

# --- Simulate Timelines ---

def simulate_timelines(planets, fleet_arrival_map, initial_by_id, angular_velocity, comets,
                       step, comet_ids_set, horizon=80, extra_arrivals=None):
    # Performance: Only re-simulate planets that actually have arrivals if extra_arrivals is provided
    # We'll need a full simulation on the first call, then can do partial updates.
    # However, to keep it safe and clean, we'll just optimize the inner loop of combat resolution.
    arrivals = defaultdict(lambda: defaultdict(list))

    for f_id, (target_id, arr_turns, owner, ships) in fleet_arrival_map.items():
        if target_id is not None and arr_turns is not None and arr_turns <= horizon:
            arrivals[target_id][arr_turns].append((owner, ships))

    if extra_arrivals:
        for pid, lst in extra_arrivals.items():
            for arr_turns, owner, ships in lst:
                if arr_turns <= horizon:
                    arrivals[pid][arr_turns].append((owner, ships))

    timelines = {}
    for p in planets:
        curr_owner, curr_ships = p.owner, p.ships
        timeline = [PlanetState(curr_owner, curr_ships)]
        p_arrivals = arrivals.get(p.id, {})

        global _COMET_LIFE_TURN
        comet_life = _COMET_LIFE_TURN.get(p.id, 1000)

        for t in range(1, horizon + 1):
            if t > comet_life:
                curr_owner, curr_ships = -1, 0
            else:
                if curr_owner != -1: curr_ships += p.production

                turn_fleets = p_arrivals.get(t)
                if turn_fleets:
                    curr_owner, curr_ships = resolve_combat(curr_owner, curr_ships, turn_fleets)

            timeline.append(PlanetState(curr_owner, curr_ships))
        timelines[p.id] = timeline
    return timelines

def get_game_phase(step):
    if step < 60:
        return "early"
    if step < 420:
        return "mid"
    return "late"

def is_static_planet(planet, initial_by_id):
    init = initial_by_id.get(planet.id)
    if init is None:
        return True
    return math.hypot(init.x - 50.0, init.y - 50.0) + init.radius >= 50.0

def compute_dynamic_margins(my_planets, timelines, player, game_phase):
    margins = {}
    for p in my_planets:
        tl = timelines.get(p.id, [])
        earliest_enemy = None
        for t in range(1, len(tl)):
            if tl[t].owner != player:
                earliest_enemy = t
                break
        if earliest_enemy is not None and earliest_enemy <= 6:
            margins[p.id] = p.ships
        elif earliest_enemy is not None and earliest_enemy <= 18:
            margins[p.id] = max(8, p.ships * 2 // 3)
        else:
            if game_phase == "early":
                margins[p.id] = max(3, p.production)
            else:
                margins[p.id] = max(5, p.production * 2)
    return margins

def score_capture(target, needed_to_send, T, player, timelines, my_planets,
                  current_step, comet_ids_set, comets, initial_by_id):
    remaining = 500 - (current_step + T)
    if remaining <= 5:
        return -1.0

    tl = timelines.get(target.id, [])
    if not tl: return -1.0

    t_idx = min(T, len(tl) - 1)
    target_state = tl[t_idx]

    # If it's already ours in the simulation, ROI is low unless we're reinforcing
    if target_state.owner == player: return 0.1

    time_penalty = 1.0 + T / 30.0
    # Prioritize higher production and planets that will be owned longer
    base = (target.production ** 2.2 * remaining) / (needed_to_send * time_penalty + 5.0)

    # SNIPE OPTIMIZATION: Boost score if the planet just changed hands or is about to
    # Check if there's heavy combat predicted at the target just before we arrive
    combat_nearby = False
    for dt in range(-2, 1):
        check_t = t_idx + dt
        if 1 <= check_t < len(tl):
            if tl[check_t].owner != tl[max(0, check_t-1)].owner:
                combat_nearby = True
                break

    if combat_nearby:
        base *= 2.5  # Significant boost for sniping opportunities

    if tl[t_idx].owner not in (-1, player):
        base *= 2.2  # Boost for taking from enemies vs neutrals

    if is_static_planet(target, initial_by_id):
        base *= 1.6

    if my_planets:
        mn = min(dist(target.x, target.y, p.x, p.y) for p in my_planets)
        if mn < 20.0: base *= 1.3
        elif mn < 40.0: base *= 1.1

    if target.id in comet_ids_set:
        life = get_comet_life(target.id, comets)
        usable = life - T
        if usable < 10: return -1.0
        base *= min(1.0, usable / 50.0)

    return base

def plan_multi_source(target, T_est, needed_total, my_planets, available_ships,
                      initial_by_id, angular_velocity, comets, current_step):
    # Coordination optimization: Try to find a single arrival turn that most planets can hit
    best_plan = None
    max_total = 0

    # Check a window around T_est to find the best synchronization turn
    for sync_t in range(max(1, T_est - 8), T_est + 15):
        current_plan = []
        current_total = 0
        for src in sorted(my_planets, key=lambda p: dist(p.x, p.y, target.x, target.y)):
            surplus = available_ships.get(src.id, 0)
            if surplus <= 2: continue

            res = find_ships_for_arrival_turn(src, target, sync_t, initial_by_id,
                                              angular_velocity, comets, current_step, surplus)
            if res:
                angle, s_min = res
                # Send enough to help but don't over-commit if not needed
                s_send = s_min # Precision timing is better than overkill
                current_plan.append((src, angle, s_send, sync_t))
                current_total += s_send
                if current_total >= needed_total: break

        if current_total >= needed_total:
            # Found a viable plan for this sync_t
            return current_plan
        elif current_total > max_total:
            max_total = current_total
            best_plan = current_plan

    return best_plan if max_total >= needed_total * 0.8 else None

def get_obs_val(obs, key, default):
    if isinstance(obs, dict):
        return obs.get(key, default)
    if hasattr(obs, key):
        return getattr(obs, key)
    return default

# --- Main Agent ---

def agent(obs):
    global _TURN_POS_CACHE
    _TURN_POS_CACHE = {}
    _current_step = obs.get("step", 0)

    player          = get_obs_val(obs, "player", 0)
    raw_planets     = get_obs_val(obs, "planets", [])
    raw_fleets      = get_obs_val(obs, "fleets", [])
    angular_vel     = get_obs_val(obs, "angular_velocity", 0.0)
    raw_initial     = get_obs_val(obs, "initial_planets", [])
    comets          = get_obs_val(obs, "comets", [])
    comet_pids      = get_obs_val(obs, "comet_planet_ids", [])
    obs_step        = get_obs_val(obs, "step", -1)
    overage_time    = get_obs_val(obs, "remainingOverageTime", 10.0)

    if obs_step != -1:
        _current_step = obs_step
    else:
        _current_step += 1

    # Dynamic Horizon Optimization
    horizon = min(80, max(20, 500 - _current_step))
    # Emergency Overage Time Safeguard
    if overage_time < 1.0:
        horizon = min(horizon, 40)
    if overage_time < 0.3:
        horizon = min(horizon, 20)

    def mk_planets(raw):
        out = []
        for p in raw:
            if isinstance(p, (list, tuple)):
                out.append(Planet(*p))
            else:
                out.append(Planet(p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production))
        return out

    planets  = mk_planets(raw_planets)
    initial_planets = mk_planets(raw_initial)

    fleets = []
    for f in raw_fleets:
        if isinstance(f, (list, tuple)):
            fleets.append(Fleet(*f))
        else:
            fleets.append(Fleet(f.id, f.owner, f.x, f.y, f.angle, f.from_planet_id, f.ships))

    initial_by_id   = {p.id: p for p in initial_planets}


    comet_ids_set   = set(comet_pids)
    global _COMET_IDS_TURN, _COMET_LIFE_TURN
    _COMET_IDS_TURN = comet_ids_set
    _COMET_LIFE_TURN = {}
    for g in comets:
        pids = g.get("planet_ids", [])
        paths = g.get("paths", [])
        path_idx = g.get("path_index", 0)
        for j, pid in enumerate(pids):
            if j < len(paths):
                _COMET_LIFE_TURN[pid] = max(0, len(paths[j]) - path_idx)


    planet_by_id    = {p.id: p for p in planets}
    my_planets      = [p for p in planets if p.owner == player]
    game_phase      = get_game_phase(_current_step)

    moves                 = []
    global_extra_arrivals = defaultdict(list)

    if not my_planets:
        return moves

    # Precompute arrivals once per turn using optimized max_steps
    fleet_arrival_map = precompute_fleet_arrivals(
        fleets, planets, initial_by_id, angular_vel, comets, _current_step, horizon
    )

    timelines = simulate_timelines(
        planets, fleet_arrival_map, initial_by_id, angular_vel, comets,
        _current_step, comet_ids_set, horizon=horizon
    )

    margins = compute_dynamic_margins(my_planets, timelines, player, game_phase)
    available_ships = {
        p.id: max(0, p.ships - margins.get(p.id, max(5, p.production * 2)))
        for p in my_planets
    }

    # ==========================================================
    # PHASE 1: Comet evacuation
    # ==========================================================
    for src in my_planets:
        if src.id not in comet_ids_set:
            continue
        life = get_comet_life(src.id, comets)
        if not (1 < life <= 4):
            continue
        candidates = [(dist(src.x, src.y, t.x, t.y), t)
                      for t in my_planets
                      if t.id != src.id and t.id not in comet_ids_set]
        candidates.sort()
        for _, target in candidates:
            res = find_intercept_angle_and_time(src, target, initial_by_id, angular_vel,
                                                comets, _current_step, src.ships - 1, horizon)
            if res is None:
                continue
            ships_evac = src.ships - 1
            if ships_evac > 0:
                moves.append([src.id, res[0], ships_evac])
                available_ships[src.id] = 0
            break

    # ==========================================================
    # ==========================================================
    # ==========================================================
    # PHASE 2: Emergency defense (Stable Multi-source)
    # ==========================================================
    threatened = {}
    for p in my_planets:
        tl = timelines[p.id]
        for t in range(1, len(tl)):
            if tl[t].owner != player:
                threatened[p.id] = t
                break

    # Sort threatened planets by turn of fall, then by production (save high value first)
    for target_id, t_fall in sorted(threatened.items(), key=lambda x: (x[1], -planet_by_id[x[0]].production)):
        target_p = planet_by_id[target_id]
        candidates = sorted(
            [src for src in my_planets if src.id != target_id and available_ships[src.id] > 0],
            key=lambda s: dist(s.x, s.y, target_p.x, target_p.y)
        )

        pending_defense_moves = []
        temp_extra_arrivals = defaultdict(list)
        for k, v in global_extra_arrivals.items():
            temp_extra_arrivals[k] = list(v)

        defended = False
        for src in candidates:
            surplus = available_ships[src.id]
            # Try to arrive exactly on time or slightly before
            # Optimization: Try multiple arrival times to see if it helps against waves
            best_res = None
            for arr_offset in range(0, 3):
                arr_t = t_fall - arr_offset
                if arr_t < 1: continue
                res = find_ships_for_arrival_turn(src, target_p, arr_t, initial_by_id,
                                                  angular_vel, comets, _current_step, surplus)
                if res:
                    best_res = (res[0], arr_t, res[1])
                    break

            if best_res:
                angle, T, s_send = best_res
                pending_defense_moves.append((src.id, angle, s_send, T))
                temp_extra_arrivals[target_id].append((T, player, s_send))

                test_tl = simulate_timelines(
                    [target_p], fleet_arrival_map, initial_by_id, angular_vel,
                    comets, _current_step, comet_ids_set,
                    horizon=horizon, extra_arrivals=temp_extra_arrivals
                )
                # Stability check: ensure it stays ours for the whole predicted timeline
                tl_target = test_tl[target_id]
                if all(tl_target[tx].owner == player for tx in range(t_fall, len(tl_target))):
                    # Defended! Commit moves.
                    for s_id, ang, s_count, t_arr in pending_defense_moves:
                        moves.append([s_id, ang, s_count])
                        available_ships[s_id] -= s_count
                        global_extra_arrivals[target_id].append((t_arr, player, s_count))
                    timelines = simulate_timelines(
                        planets, fleet_arrival_map, initial_by_id, angular_vel, comets,
                        _current_step, comet_ids_set, horizon=horizon,
                        extra_arrivals=global_extra_arrivals
                    )
                    defended = True
                    break
        if defended:
            continue
    # PHASE 3: Tactical snipe
    # ==========================================================
    my_total_prod    = sum(p.production for p in my_planets)
    snipe_threshold  = max(25, int(my_total_prod * 4.0))

    for p in planets:
        if p.owner == player:
            continue
        tl = timelines[p.id]
        for t in range(1, len(tl) - 1):
            prev_s, curr_s = tl[t - 1], tl[t]
            if curr_s.owner == player or prev_s.owner == player:
                continue
            if not (0 < curr_s.ships <= snipe_threshold):
                continue
            target_t = t + 1
            needed   = curr_s.ships + (p.production + 1 if curr_s.owner != -1 else 1)
            for src in my_planets:
                surplus = available_ships[src.id]
                if surplus < needed:
                    continue
                res = find_ships_for_arrival_turn(src, p, target_t, initial_by_id,
                                                  angular_vel, comets, _current_step, surplus)
                if res is None:
                    continue
                angle, s_needed = res
                if needed <= s_needed <= surplus:
                    moves.append([src.id, angle, s_needed])
                    available_ships[src.id] -= s_needed
                    global_extra_arrivals[p.id].append((target_t, player, s_needed))
                    timelines = simulate_timelines(
                        planets, fleet_arrival_map, initial_by_id, angular_vel, comets,
                        _current_step, comet_ids_set, horizon=horizon,
                        extra_arrivals=global_extra_arrivals
                    )
                    break

    # ==========================================================
    # ==========================================================
    # PHASE 4: ROI-driven single-source captures
    # ==========================================================
    candidates = []
    for src in my_planets:
        surplus = available_ships[src.id]
        if surplus <= 3: continue
        for target in planets:
            if target.id == src.id: continue
            # Initial estimate using 75% of surplus as a mid-range speed
            res = find_intercept_angle_and_time(src, target, initial_by_id, angular_vel,
                                                comets, _current_step, max(1, int(surplus*0.75)), horizon)
            if res is None: continue
            angle, T = res
            if _current_step + T >= 495: continue

            tl = timelines.get(target.id, [])
            if not tl: continue
            t_idx = min(T, len(tl) - 1)
            if tl[t_idx].owner == player: continue

            needed_at = tl[t_idx].ships + 1
            overhead = 1.15 if game_phase == "early" else 1.10
            needed_send = int(needed_at * overhead) + 2
            if needed_send > surplus: continue

            # Re-verify intercept with the actual needed_send count (speed might change)
            res2 = find_intercept_angle_and_time(src, target, initial_by_id, angular_vel,
                                                 comets, _current_step, needed_send, horizon)
            if res2:
                angle2, T2 = res2
                sc = score_capture(target, needed_send, T2, player, timelines, my_planets,
                                   _current_step, comet_ids_set, comets, initial_by_id)
                if sc > 0:
                    candidates.append((sc, src.id, target.id, angle2, needed_send, T2))
    committed_targets = set()

    for sc, src_id, target_id, angle, needed, T in candidates:
        if target_id in committed_targets:
            continue
        if available_ships.get(src_id, 0) < needed:
            continue
        moves.append([src_id, angle, needed])
        available_ships[src_id] -= needed
        committed_targets.add(target_id)
        global_extra_arrivals[target_id].append((T, player, needed))
        timelines = simulate_timelines(
            planets, fleet_arrival_map, initial_by_id, angular_vel, comets,
            _current_step, comet_ids_set, horizon=horizon, extra_arrivals=global_extra_arrivals
        )

    # Skip heavy multi-source and logistics if severely constrained by remaining time budget
    if overage_time > 0.5:
        # ==========================================================
        # PHASE 4b: Multi-source captures
        # ==========================================================
        if game_phase != "late":
            for target in planets:
                if target.owner == player or target.id in committed_targets:
                    continue
                if target.id in comet_ids_set:
                    continue

                if not my_planets:
                    continue
                src_ref = min(my_planets, key=lambda p: dist(p.x, p.y, target.x, target.y))
                ref_res = find_intercept_angle_and_time(src_ref, target, initial_by_id,
                                                        angular_vel, comets, _current_step, 50, horizon)
                if ref_res is None:
                    continue
                _, T_est = ref_res
                if _current_step + T_est >= 480:
                    continue

                tl    = timelines.get(target.id, [])
                t_idx = min(T_est, len(tl) - 1)
                if not tl or tl[t_idx].owner == player:
                    continue
                needed_total = tl[t_idx].ships + 3

                contrib = plan_multi_source(target, T_est, needed_total, my_planets,
                                            available_ships, initial_by_id, angular_vel,
                                            comets, _current_step)
                if contrib is None or len(contrib) < 2:
                    continue

                test_extra = defaultdict(list)
                for k, v in global_extra_arrivals.items():
                    test_extra[k] = list(v)
                for src, ang, ships, t_arr in contrib:
                    test_extra[target.id].append((t_arr, player, ships))
                test_tl = simulate_timelines(
                    [target], fleet_arrival_map, initial_by_id, angular_vel, comets,
                    _current_step, comet_ids_set, horizon=T_est + 5, extra_arrivals=test_extra
                )
                if not any(test_tl[target.id][t].owner == player
                           for t in range(1, min(T_est + 5, len(test_tl[target.id])))):
                    continue

                for src, ang, ships, t_arr in contrib:
                    moves.append([src.id, ang, ships])
                    available_ships[src.id] -= ships
                    global_extra_arrivals[target.id].append((t_arr, player, ships))
                committed_targets.add(target.id)
                timelines = simulate_timelines(
                    planets, fleet_arrival_map, initial_by_id, angular_vel, comets,
                    _current_step, comet_ids_set, horizon=horizon, extra_arrivals=global_extra_arrivals
                )

        # ==========================================================
        # PHASE 5: Forward deployment / rear-line regrouping
        # ==========================================================
        if game_phase != "late":
            enemy_planets = [p for p in planets if p.owner not in (player, -1)]
            if enemy_planets:
                for src_id, surplus in list(available_ships.items()):
                    if surplus <= 5:
                        continue
                    src = planet_by_id[src_id]
                    src_dist = min(dist(src.x, src.y, ep.x, ep.y) for ep in enemy_planets)
                    best_target, best_dist_to_enemy, best_angle = None, src_dist - 5.0, None
                    for target in my_planets:
                        if target.id == src_id:
                            continue
                        tgt_dist = min(dist(target.x, target.y, ep.x, ep.y) for ep in enemy_planets)
                        if tgt_dist < best_dist_to_enemy:
                            res = find_intercept_angle_and_time(src, target, initial_by_id,
                                                                angular_vel, comets,
                                                                _current_step, surplus, horizon)
                            if res is not None and dist(src.x, src.y, target.x, target.y) < 40.0:
                                best_target, best_dist_to_enemy, best_angle = target, tgt_dist, res[0]
                    if best_target is not None:
                        moves.append([src_id, best_angle, surplus])
                        available_ships[src_id] = 0

    return moves