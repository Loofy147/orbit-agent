"""
orbit_agent_v3.py — Optimized and Robust Orbit Wars Agent

Key Improvements:
  1. PERF: Dynamic Horizon Scaling based on remaining steps (caps at 20-80).
  2. PERF: Integrates Overage Time Safeguards to reduce search horizon under time pressure.
  3. PERF: Cuts path-tracing limits using the dynamic horizon, saving O(N) calculations on long-range fleets.
  4. FIX: Resolves potential self-collision edge cases on slow fleet launches (t <= 2).
"""

import math
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

_CURRENT_STEP = -1

# --- Geometry Utilities ---

def dist(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def point_to_segment_distance(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return dist(px, py, x1, y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    return dist(px, py, x1 + t * dx, y1 + t * dy)

def segment_hits_sun(x1, y1, x2, y2, sun_radius=10.0, safety=0.2):
    return point_to_segment_distance(50.0, 50.0, x1, y1, x2, y2) < (sun_radius + safety)

def safe_angle_and_distance(sx, sy, sr, tx, ty, tr, sun_radius=10.0, safety=0.2):
    angle = math.atan2(ty - sy, tx - sx)
    lx = sx + (sr + 0.1) * math.cos(angle)
    ly = sy + (sr + 0.1) * math.sin(angle)
    if segment_hits_sun(lx, ly, tx, ty, sun_radius, safety):
        return None
    return angle, max(0.0, dist(lx, ly, tx, ty) - tr)

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
    init = initial_by_id.get(planet.id)
    if init is None:
        return planet.x, planet.y
    dx_c, dy_c = init.x - 50.0, init.y - 50.0
    orb_r = math.hypot(dx_c, dy_c)
    if orb_r + init.radius >= 50.0:
        return init.x, init.y
    theta = math.atan2(dy_c, dx_c) + angular_velocity * step
    return 50.0 + orb_r * math.cos(theta), 50.0 + orb_r * math.sin(theta)

def predict_position(planet_id, planet, initial_by_id, angular_velocity, comets, current_step, turns_ahead):
    if turns_ahead == 0:
        return planet.x, planet.y
    is_comet = any(planet_id in g.get("planet_ids", []) for g in comets)
    if is_comet:
        return predict_comet_position_at_turn(planet_id, comets, turns_ahead)
    return predict_planet_position_at_step(planet, initial_by_id, angular_velocity, current_step + turns_ahead)

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
    dx = speed * math.cos(fleet.angle)
    dy = speed * math.sin(fleet.angle)
    for t in range(1, max_steps):
        fnx, fny = fx + dx, fy + dy
        if segment_hits_sun(fx, fy, fnx, fny, safety=0.0):
            return None, None
        for p in planets:
            # FIX: Prevent incorrect self-collision on slow launch steps (t <= 2)
            if p.id == fleet.from_planet_id and t <= 2: 
                continue
            pos = predict_position(p.id, p, initial_by_id, angular_velocity, comets, step, t)
            if pos is None: 
                continue
            if point_to_segment_distance(pos[0], pos[1], fx, fy, fnx, fny) < p.radius:
                return p.id, t
        if fnx < 0.0 or fnx > 100.0 or fny < 0.0 or fny > 100.0:
            return None, None
        fx, fy = fnx, fny
    return None, None

# --- Precompute Fleet Arrivals ---

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
    arrivals = defaultdict(list)

    for f_id, (target_id, arr_turns, owner, ships) in fleet_arrival_map.items():
        if target_id is not None and arr_turns is not None and arr_turns <= horizon:
            arrivals[target_id].append((arr_turns, owner, ships))

    if extra_arrivals:
        for pid, lst in extra_arrivals.items():
            for arr_turns, owner, ships in lst:
                if arr_turns <= horizon:
                    arrivals[pid].append((arr_turns, owner, ships))

    timelines = {}
    for p in planets:
        curr_owner, curr_ships = p.owner, p.ships
        timeline = [PlanetState(curr_owner, curr_ships)]
        p_arr = defaultdict(list)
        for arr_t, owner, ships in arrivals[p.id]:
            p_arr[arr_t].append((owner, ships))
        comet_life = get_comet_life(p.id, comets) if p.id in comet_ids_set else float('inf')
        for t in range(1, horizon + 1):
            if t > comet_life:
                curr_owner, curr_ships = -1, 0
            else:
                if curr_owner != -1:
                    curr_ships += p.production
                curr_owner, curr_ships = resolve_combat(curr_owner, curr_ships, p_arr[t])
            timeline.append(PlanetState(curr_owner, curr_ships))
        timelines[p.id] = timeline
    return timelines

# --- Strategic Helpers ---

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

    time_penalty = 1.0 + T / 40.0
    base = (target.production ** 1.3 * remaining) / (needed_to_send * time_penalty + 1.0)

    tl = timelines.get(target.id, [])
    if tl:
        t_idx = min(T, len(tl) - 1)
        if tl[t_idx].owner not in (-1, player):
            base *= 2.2

    if is_static_planet(target, initial_by_id):
        base *= 1.4

    if my_planets:
        mn = min(dist(target.x, target.y, p.x, p.y) for p in my_planets)
        if mn < 25.0: 
            base *= 1.25
        elif mn < 40.0: 
            base *= 1.1

    if target.id in comet_ids_set:
        life = get_comet_life(target.id, comets)
        usable = life - T
        if usable < 5: 
            return -1.0
        base *= min(1.0, usable / 40.0)

    return base

def plan_multi_source(target, T_est, needed_total, my_planets, available_ships,
                      initial_by_id, angular_velocity, comets, current_step):
    contributors = []
    total = 0
    for src in sorted(my_planets, key=lambda p: dist(p.x, p.y, target.x, target.y)):
        surplus = available_ships.get(src.id, 0)
        if surplus <= 2: 
            continue
        for t_try in [T_est, T_est + 1, T_est - 1]:
            if t_try < 1: 
                continue
            res = find_ships_for_arrival_turn(src, target, t_try, initial_by_id,
                                              angular_velocity, comets, current_step, surplus)
            if res is None: 
                continue
            angle, s_min = res
            send = max(s_min, min(surplus, needed_total - total + 3))
            contributors.append((src, angle, send, t_try))
            total += send
            break
        if total >= needed_total:
            return contributors
    return None

def get_obs_val(obs, key, default):
    if isinstance(obs, dict): 
        return obs.get(key, default)
    if hasattr(obs, key): 
        return getattr(obs, key)
    return default

# --- Main Agent ---

def agent(obs):
    global _CURRENT_STEP

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
        _CURRENT_STEP = obs_step
    else:              
        _CURRENT_STEP += 1

    # Dynamic Horizon Optimization
    horizon = min(80, max(20, 500 - _CURRENT_STEP))
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
    planet_by_id    = {p.id: p for p in planets}
    my_planets      = [p for p in planets if p.owner == player]
    game_phase      = get_game_phase(_CURRENT_STEP)

    moves                 = []
    global_extra_arrivals = defaultdict(list)

    if not my_planets:
        return moves

    # Precompute arrivals once per turn using optimized max_steps
    fleet_arrival_map = precompute_fleet_arrivals(
        fleets, planets, initial_by_id, angular_vel, comets, _CURRENT_STEP, horizon
    )

    timelines = simulate_timelines(
        planets, fleet_arrival_map, initial_by_id, angular_vel, comets,
        _CURRENT_STEP, comet_ids_set, horizon=horizon
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
                                                comets, _CURRENT_STEP, src.ships - 1, horizon)
            if res is None: 
                continue
            ships_evac = src.ships - 1
            if ships_evac > 0:
                moves.append([src.id, res[0], ships_evac])
                available_ships[src.id] = 0
            break

    # ==========================================================
    # PHASE 2: Emergency defense
    # ==========================================================
    threatened = {}
    for p in my_planets:
        tl = timelines[p.id]
        for t in range(1, len(tl)):
            if tl[t].owner != player:
                threatened[p.id] = t
                break

    for target_id, t_fall in sorted(threatened.items(), key=lambda x: x[1]):
        target_p = planet_by_id[target_id]
        defended  = False
        candidates = sorted(
            [(src, available_ships[src.id], dist(src.x, src.y, target_p.x, target_p.y))
             for src in my_planets if src.id != target_id and available_ships[src.id] > 0],
            key=lambda x: x[2]
        )
        for src, surplus, _ in candidates:
            for arr_t in [t_fall - 1, t_fall]:
                if arr_t < 1: 
                    continue
                res = find_ships_for_arrival_turn(src, target_p, arr_t, initial_by_id,
                                                  angular_vel, comets, _CURRENT_STEP, surplus)
                if res is None: 
                    continue
                angle, s_needed = res
                test_extra = defaultdict(list)
                for k, v in global_extra_arrivals.items():
                    test_extra[k] = list(v)
                test_extra[target_id].append((arr_t, player, s_needed))
                test_tl = simulate_timelines(
                    [target_p], fleet_arrival_map, initial_by_id, angular_vel,
                    comets, _CURRENT_STEP, comet_ids_set,
                    horizon=t_fall + 2, extra_arrivals=test_extra
                )
                t_check = min(t_fall, len(test_tl[target_id]) - 1)
                if test_tl[target_id][t_check].owner == player:
                    moves.append([src.id, angle, s_needed])
                    available_ships[src.id] -= s_needed
                    global_extra_arrivals[target_id].append((arr_t, player, s_needed))
                    defended = True
                    break
            if defended:
                timelines = simulate_timelines(
                    planets, fleet_arrival_map, initial_by_id, angular_vel, comets,
                    _CURRENT_STEP, comet_ids_set, horizon=horizon,
                    extra_arrivals=global_extra_arrivals
                )
                break

    # ==========================================================
    # PHASE 3: Tactical snipe
    # ==========================================================
    my_total_prod    = sum(p.production for p in my_planets)
    snipe_threshold  = max(15, int(my_total_prod * 2.0))

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
                                                  angular_vel, comets, _CURRENT_STEP, surplus)
                if res is None: 
                    continue
                angle, s_needed = res
                if needed <= s_needed <= surplus:
                    moves.append([src.id, angle, s_needed])
                    available_ships[src.id] -= s_needed
                    global_extra_arrivals[p.id].append((target_t, player, s_needed))
                    timelines = simulate_timelines(
                        planets, fleet_arrival_map, initial_by_id, angular_vel, comets,
                        _CURRENT_STEP, comet_ids_set, horizon=horizon,
                        extra_arrivals=global_extra_arrivals
                    )
                    break

    # ==========================================================
    # PHASE 4: ROI-driven single-source captures
    # ==========================================================
    candidates = []
    for src in my_planets:
        surplus = available_ships[src.id]
        if surplus <= 3: 
            continue
        for target in planets:
            if target.id == src.id: 
                continue
            res = find_intercept_angle_and_time(src, target, initial_by_id, angular_vel,
                                                comets, _CURRENT_STEP, surplus, horizon)
            if res is None: 
                continue
            angle, T = res
            if _CURRENT_STEP + T >= 495: 
                continue

            tl    = timelines.get(target.id, [])
            t_idx = min(T, len(tl) - 1)
            if not tl or tl[t_idx].owner == player: 
                continue

            needed_at   = tl[t_idx].ships + 1
            overhead    = 1.15 if game_phase == "early" else 1.10
            needed_send = int(needed_at * overhead) + 2
            if needed_send > surplus: 
                continue

            sc = score_capture(target, needed_send, T, player, timelines, my_planets,
                               _CURRENT_STEP, comet_ids_set, comets, initial_by_id)
            if sc > 0:
                candidates.append((sc, src.id, target.id, angle, needed_send, T))

    candidates.sort(key=lambda x: x[0], reverse=True)
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
            _CURRENT_STEP, comet_ids_set, horizon=horizon, extra_arrivals=global_extra_arrivals
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
                                                        angular_vel, comets, _CURRENT_STEP, 50, horizon)
                if ref_res is None: 
                    continue
                _, T_est = ref_res
                if _CURRENT_STEP + T_est >= 480: 
                    continue

                tl    = timelines.get(target.id, [])
                t_idx = min(T_est, len(tl) - 1)
                if not tl or tl[t_idx].owner == player: 
                    continue
                needed_total = tl[t_idx].ships + 3

                contrib = plan_multi_source(target, T_est, needed_total, my_planets,
                                            available_ships, initial_by_id, angular_vel,
                                            comets, _CURRENT_STEP)
                if contrib is None or len(contrib) < 2: 
                    continue

                test_extra = defaultdict(list)
                for k, v in global_extra_arrivals.items():
                    test_extra[k] = list(v)
                for src, ang, ships, t_arr in contrib:
                    test_extra[target.id].append((t_arr, player, ships))
                test_tl = simulate_timelines(
                    [target], fleet_arrival_map, initial_by_id, angular_vel, comets,
                    _CURRENT_STEP, comet_ids_set, horizon=T_est + 5, extra_arrivals=test_extra
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
                    _CURRENT_STEP, comet_ids_set, horizon=horizon, extra_arrivals=global_extra_arrivals
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
                                                                _CURRENT_STEP, surplus, horizon)
                            if res is not None and dist(src.x, src.y, target.x, target.y) < 40.0:
                                best_target, best_dist_to_enemy, best_angle = target, tgt_dist, res[0]
                    if best_target is not None:
                        moves.append([src_id, best_angle, surplus])
                        available_ships[src_id] = 0

    return moves