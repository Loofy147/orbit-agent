from __future__ import annotations

import dataclasses
import os
import sys
from dataclasses import dataclass, replace

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
from torch import Tensor
import math

from orbit_lite.geometry import fleet_speed
from orbit_lite.intercept_aim import intercept_angle
from orbit_lite.movement import MovementConfig, PlanetMovement
from orbit_lite.movement_step import (
    apply_private_planned_launches,
    concat_launch_entries,
    disambiguate_duplicate_launches,
    ensure_planet_movement,
    infer_planned_launches_from_entries,
)
from orbit_lite.obs import parse_obs
from orbit_lite.distance_cache import build_distance_cache
from orbit_lite.planner_core import (
    _candidate_indices,
    _empty_entries,
    _greedy_select,
    _plan_regroup,
    build_target_shortlist,
    capture_floor,
    empty_action_row,
    entries_to_sparse_payload,
    largest_initial_player_count,
    make_launch_set,
    reachable_mask,
    reinforcement_timing_factor,
    safe_drain,
    score_candidates,
)
from orbit_lite.adapter import single_obs_to_tensor, sparse_action_row_to_moves

TOTAL_STEPS = 500
MAX_FLEET_SPEED = 6.0

def _fleet_speed_scalar(ships: float) -> float:
    if ships < 1.0:
        return 1.0
    return 1.0 + (MAX_FLEET_SPEED - 1.0) * (math.log(max(ships, 1.0)) / math.log(1000.0)) ** 1.5

@dataclass(frozen=True)
class ProducerLiteConfig:
    horizon: int = 18
    max_sources_per_lane: int = 12
    max_offensive_targets: int = 12
    max_defensive_targets: int = 6
    max_waves_per_turn: int = 6
    roi_threshold: float = 1.35
    min_ships_to_launch: float = 4.0
    reinforce_size_beta: float = 2.2
    reinforce_eta_free: float = 3.0
    reinforce_eta_scale: float = 12.0
    enable_regroup: bool = True
    max_regroup_time: float = 7.0
    regroup_pressure_delta_min: float = 0.20
    max_regroup_sources_per_lane: int = 6
    max_regroup_targets_per_source: int = 7
    regroup_pressure_norm: str = "none"
    regroup_time_penalty_weight: float = 1e-3
    min_roi: float = 1.05
    max_roi: float = 1.45
    horizon_min: int = 8
    horizon_max: int = 24
    beta_min: float = 1.2
    beta_max: float = 3.5
    defense_threat_horizon: float = 14.0
    defense_min_intercept_margin: float = 1.05
    defense_max_waves: int = 3
    geometry_weight: float = 0.35
    prod_rush_steps: int = 120
    prod_rush_top_k: int = 3
    prod_rush_roi_discount: float = 0.80
    comet_score_multiplier: float = 2.5
    comet_min_lifetime_turns: int = 8
    comet_opportunity_weight: float = 0.6
    opportunity_cost_weight: float = 0.15
    late_enemy_divisor: float = 30.0
    late_neutral_divisor: float = 30.0
    endgame_dump_start: int = 60
    endgame_dump_keep_fraction: float = 0.10
    endgame_dump_min_send: float = 8.0
    contested_penalty: float = 0.0
    leader_attack_bonus: float = 1.4
    weakest_attack_penalty: float = 0.7
    economy_drain_prod_threshold: int = 3
    economy_drain_min_garrison: float = 15.0
    economy_drain_target_fraction: float = 0.55
    orbit_window_bonus: float = 1.3
    orbit_window_turns: int = 12

def _owner_strength(obs, prod: Tensor, player_count: int) -> Tensor:
    dtype = prod.dtype
    device = prod.device
    strength = torch.zeros(int(player_count), dtype=dtype, device=device)
    owner = obs.owner_abs.to(device=device)
    alive = obs.alive.to(device=device)
    ships = obs.ships.to(device=device, dtype=dtype)
    prod_v = prod.to(device=device, dtype=dtype)
    for oid in range(int(player_count)):
        mask = alive & (owner == oid)
        if bool(mask.any()):
            strength[oid] = prod_v[mask].sum() + 0.025 * ships[mask].sum()
    return strength

def _orbital_centrality(obs, cache) -> Tensor:
    P = int(obs.P)
    device = obs.device
    if P <= 1:
        return torch.ones(P, device=device)
    d0 = cache.cross_dist[0].clone().float()
    alive = obs.alive.to(device=device)
    eye = torch.eye(P, dtype=torch.bool, device=device)
    pair_valid = alive.view(1, P) & alive.view(P, 1) & ~eye
    d0 = torch.where(pair_valid, d0, torch.zeros_like(d0))
    n_peers = pair_valid.float().sum(dim=1).clamp(min=1.0)
    mean_dist = d0.sum(dim=1) / n_peers
    centrality = 1.0 / (mean_dist + 1.0)
    centrality = torch.where(alive, centrality, torch.zeros_like(centrality))
    return centrality.to(obs.ships.dtype)

def _extract_comet_ids(obs_tensors: dict, device) -> Tensor:
    raw = obs_tensors.get("comet_planet_ids", None)
    if raw is None:
        return torch.zeros(0, dtype=torch.long, device=device)
    t = raw.reshape(-1).long().to(device=device)
    return t[t >= 0]

def _comet_mask_from_ids(comet_ids: Tensor, P: int, device) -> Tensor:
    mask = torch.zeros(P, dtype=torch.bool, device=device)
    if comet_ids.numel() == 0:
        return mask
    valid = comet_ids[comet_ids < P]
    if valid.numel() > 0:
        mask[valid] = True
    return mask

def _comet_remaining_lifetime(obs_tensors: dict, comet_ids: Tensor, device) -> dict:
    DEFAULT_LIFETIME = 999
    lifetimes: dict = {}
    if comet_ids.numel() == 0:
        return lifetimes
    comets_field = obs_tensors.get("comets", None)
    if comets_field is None:
        for cid in comet_ids.tolist():
            lifetimes[int(cid)] = DEFAULT_LIFETIME
        return lifetimes
    try:
        for g_idx in range(comets_field.shape[0]):
            group = comets_field[g_idx]
            g_planet_ids = group.get("planet_ids", None) if hasattr(group, "get") else None
            g_path_index = group.get("path_index", None) if hasattr(group, "get") else None
            g_paths      = group.get("paths",      None) if hasattr(group, "get") else None
            if g_planet_ids is None or g_path_index is None or g_paths is None:
                break
            path_len  = int(g_paths.shape[-2]) if g_paths.dim() >= 2 else DEFAULT_LIFETIME
            cur_idx   = int(g_path_index.reshape(-1)[0].item())
            remaining = max(0, path_len - cur_idx - 1)
            for pid_t in g_planet_ids.reshape(-1):
                pid_int = int(pid_t.item())
                if pid_int >= 0:
                    lifetimes[pid_int] = remaining
    except Exception:
        pass
    for cid in comet_ids.tolist():
        if int(cid) not in lifetimes:
            lifetimes[int(cid)] = DEFAULT_LIFETIME
    return lifetimes

def _build_comet_lifetime_tensor(lifetimes: dict, P: int, device, dtype, default: float = 999.0) -> Tensor:
    t = torch.full((P,), default, dtype=dtype, device=device)
    for pid, lt in lifetimes.items():
        if 0 <= pid < P:
            t[pid] = float(lt)
    return t

def _build_contested_mask(obs_tensors: dict, obs, cache, player_id: int) -> Tensor:
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    contested = torch.zeros(P, dtype=dtype, device=device)
    pid = int(player_id)
    neutral = (obs.owner_abs < 0) & obs.alive
    if not bool(neutral.any()):
        return contested
    owned = obs.owned & obs.alive
    if not bool(owned.any()):
        return contested
    d0 = cache.cross_dist[0].to(dtype)
    ships = obs.ships.to(dtype)
    speeds = fleet_speed(ships.clamp(min=1.0))
    our_reach_min = torch.full((P,), float("inf"), dtype=dtype, device=device)
    owned_indices = owned.nonzero(as_tuple=False).squeeze(1)
    for src in owned_indices:
        src_i = int(src.item())
        eta_from_src = (d0[src_i] / speeds[src_i].clamp(min=1e-6)).ceil()
        our_reach_min = torch.minimum(our_reach_min, eta_from_src)
    fleets_raw = obs_tensors.get("fleets", None)
    if fleets_raw is None:
        return contested
    try:
        fleets = fleets_raw.reshape(-1, 7).to(dtype)
        fleet_owner = fleets[:, 1].long()
        fleet_ships = fleets[:, 6]
        fleet_x     = fleets[:, 2]
        fleet_y     = fleets[:, 3]
        fleet_angle = fleets[:, 4]
        enemy_fleets = (fleet_owner >= 0) & (fleet_owner != pid)
        if not bool(enemy_fleets.any()):
            return contested
        planet_data = obs_tensors["planets"].reshape(-1, 7)
        planet_x = planet_data[:, 2].to(dtype)
        planet_y = planet_data[:, 3].to(dtype)
        for fi in range(int(fleets.shape[0])):
            if not bool(enemy_fleets[fi]):
                continue
            fx = float(fleet_x[fi].item())
            fy = float(fleet_y[fi].item())
            fa = float(fleet_angle[fi].item())
            fs = float(fleet_ships[fi].item())
            fspeed = _fleet_speed_scalar(fs)
            dx = planet_x - fx
            dy = planet_y - fy
            proj = dx * math.cos(fa) + dy * math.sin(fa)
            dist_to_planet = (dx * dx + dy * dy).sqrt()
            heading_toward = (proj > 0) & neutral
            if not bool(heading_toward.any()):
                continue
            fleet_eta = (dist_to_planet / max(fspeed, 1e-6)).ceil()
            is_contested = heading_toward & (fleet_eta < our_reach_min - 1.0)
            contested = torch.where(is_contested, torch.ones_like(contested), contested)
    except Exception:
        pass
    return contested

def _player_aggression_multipliers(obs, prod: Tensor, player_count: int, config: ProducerLiteConfig) -> Tensor:
    pid = int(obs.player_id)
    pc = int(player_count)
    device = obs.device
    dtype = obs.ships.dtype
    mults = torch.ones(pc, dtype=dtype, device=device)
    if pc <= 2:
        return mults
    strength = _owner_strength(obs, prod, pc)
    enemy_strength = strength.clone()
    if pid < pc:
        enemy_strength[pid] = -1.0
    alive_enemies = (enemy_strength >= 0.0).nonzero(as_tuple=False).squeeze(1)
    if alive_enemies.numel() < 2:
        return mults
    leader_idx   = int(torch.argmax(enemy_strength).item())
    weakest_idx  = int(enemy_strength[alive_enemies].argmin().item())
    weakest_idx  = int(alive_enemies[weakest_idx].item())
    mults[leader_idx]  = float(config.leader_attack_bonus)
    if weakest_idx != leader_idx:
        mults[weakest_idx] = float(config.weakest_attack_penalty)
    return mults

def _orbit_window_mask(obs_tensors: dict, obs, cache, config: ProducerLiteConfig) -> Tensor:
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    result = torch.ones(P, dtype=dtype, device=device)
    angular_vel_raw = obs_tensors.get("angular_velocity", None)
    if angular_vel_raw is None:
        return result
    try:
        angular_vel = angular_vel_raw.reshape(-1).to(dtype)
        is_orbiting = angular_vel.abs() > 1e-6
        if not bool(is_orbiting.any()):
            return result
        owned = obs.owned & obs.alive
        if not bool(owned.any()):
            return result
        planet_data = obs_tensors["planets"].reshape(-1, 7).to(dtype)
        px = planet_data[:, 2]
        py = planet_data[:, 3]
        owned_x = px[owned].mean()
        owned_y = py[owned].mean()
        dx = px - owned_x
        dy = py - owned_y
        current_dist = (dx * dx + dy * dy).sqrt()
        SUN_X, SUN_Y = 50.0, 50.0
        rel_x = px - SUN_X
        rel_y = py - SUN_Y
        orbital_r = (rel_x * rel_x + rel_y * rel_y).sqrt()
        theta0 = torch.atan2(rel_y, rel_x)
        W = int(config.orbit_window_turns)
        min_future_dist = current_dist.clone()
        for t in range(1, W + 1):
            theta_t = theta0 + angular_vel * t
            future_x = SUN_X + orbital_r * torch.cos(theta_t)
            future_y = SUN_Y + orbital_r * torch.sin(theta_t)
            fdx = future_x - owned_x
            fdy = future_y - owned_y
            future_dist = (fdx * fdx + fdy * fdy).sqrt()
            min_future_dist = torch.where(is_orbiting, torch.minimum(min_future_dist, future_dist), min_future_dist)
        in_window = is_orbiting & (current_dist <= min_future_dist * 1.10)
        result = torch.where(in_window, torch.full_like(result, float(config.orbit_window_bonus)), result)
    except Exception:
        pass
    return result

def _build_endgame_dump_entries(*, obs, obs_tensors: dict, cache, config: ProducerLiteConfig, source_budget: Tensor, step: int) -> object:
    device = obs.device
    dtype = obs.ships.dtype
    pid = int(obs.player_id)
    P = int(obs.P)
    remaining = TOTAL_STEPS - int(step)
    owned = obs.owned & obs.alive
    if not bool(owned.any()):
        return _empty_entries(device, dtype)
    src_indices = owned.nonzero(as_tuple=False).squeeze(1)
    if src_indices.numel() == 0:
        return _empty_entries(device, dtype)
    d0 = cache.cross_dist[0].to(dtype)
    keep_frac = float(config.endgame_dump_keep_fraction)
    min_send  = float(config.endgame_dump_min_send)
    all_entries = []
    for src_t in src_indices:
        src_i = int(src_t.item())
        available = float(source_budget[src_i].item())
        keep = max(min_send, available * keep_frac)
        send = float(math.floor(available - keep))
        if send < min_send:
            continue
        own_dists = d0[src_i]
        tgt_candidates = owned.clone()
        tgt_candidates[src_i] = False
        if not bool(tgt_candidates.any()):
            continue
        fleet_spd = _fleet_speed_scalar(send)
        eta_to_tgt = (own_dists / max(fleet_spd, 1e-6))
        stays_in_air = tgt_candidates & (eta_to_tgt > float(remaining))
        if bool(stays_in_air.any()):
            valid_etas = torch.where(stays_in_air, eta_to_tgt, torch.full_like(eta_to_tgt, 1e9))
            tgt_i = int(valid_etas.argmin().item())
        else:
            tgt_score = torch.where(tgt_candidates, own_dists, torch.zeros_like(own_dists))
            tgt_i = int(tgt_score.argmax().item())
        if tgt_i == src_i or not bool(tgt_candidates[tgt_i]):
            continue
        planet_data = obs_tensors["planets"].reshape(-1, 7).to(dtype)
        sx = float(planet_data[src_i, 2].item())
        sy = float(planet_data[src_i, 3].item())
        tx = float(planet_data[tgt_i, 2].item())
        ty = float(planet_data[tgt_i, 3].item())
        angle = math.atan2(ty - sy, tx - sx)
        eta_val = float(d0[src_i, tgt_i].item()) / max(fleet_spd, 1e-6)
        source_budget[src_i] = max(0.0, float(source_budget[src_i].item()) - send)
        src_t_tensor  = torch.tensor([src_i], dtype=torch.long, device=device)
        tgt_t_tensor  = torch.tensor([tgt_i], dtype=torch.long, device=device)
        send_t        = torch.tensor([send], dtype=dtype, device=device)
        eta_t         = torch.tensor([eta_val], dtype=dtype, device=device)
        angle_t       = torch.tensor([angle], dtype=dtype, device=device)
        valid_t       = torch.tensor([True], dtype=torch.bool, device=device)
        empty = _empty_entries(device, dtype)
        entry = replace(
            empty,
            source_slots=src_t_tensor,
            target_slots=tgt_t_tensor,
            ships=send_t,
            eta=eta_t,
            angle=angle_t,
            valid=valid_t,
        )
        all_entries.append(entry)
    if not all_entries:
        return _empty_entries(device, dtype)
    return concat_launch_entries(all_entries)

def _build_economy_drain_entries(*, obs, obs_tensors: dict, cache, garrison_status, config: ProducerLiteConfig, source_budget: Tensor, player_count: int) -> object:
    device = obs.device
    dtype = obs.ships.dtype
    pid = int(obs.player_id)
    P = int(obs.P)
    owned = obs.owned & obs.alive
    prod_v = garrison_status.prod if hasattr(garrison_status, "prod") else None
    if prod_v is None:
        planet_data = obs_tensors["planets"].reshape(-1, 7).to(dtype)
        prod_v = planet_data[:, 6]
    prod_v = prod_v.to(dtype)
    prod_threshold = int(config.economy_drain_prod_threshold)
    min_garrison   = float(config.economy_drain_min_garrison)
    drain_frac     = float(config.economy_drain_target_fraction)
    is_high_prod = prod_v >= float(prod_threshold)
    src_mask = owned & is_high_prod
    if not bool(src_mask.any()):
        return _empty_entries(device, dtype)
    src_indices = src_mask.nonzero(as_tuple=False).squeeze(1)
    if src_indices.numel() == 0:
        return _empty_entries(device, dtype)
    d0 = cache.cross_dist[0].to(dtype)
    not_owned = ~obs.owned & obs.alive
    if not bool(not_owned.any()):
        return _empty_entries(device, dtype)
    tgt_indices = not_owned.nonzero(as_tuple=False).squeeze(1)
    all_entries = []
    for src_t in src_indices:
        src_i = int(src_t.item())
        budget = float(source_budget[src_i].item())
        surplus = budget - min_garrison
        if surplus < float(config.economy_drain_min_garrison if hasattr(config, 'economy_drain_min_garrison') else 15.0):
            continue
        send = float(math.floor(surplus * drain_frac))
        if send < 4.0:
            continue
        dists_to_tgt = d0[src_i, tgt_indices].to(dtype)
        tgt_prod     = prod_v[tgt_indices]
        tgt_ships_v  = obs.ships[tgt_indices].to(dtype)
        tgt_score = (tgt_prod + 0.5) / (dists_to_tgt + 1.0)
        beatable = send > tgt_ships_v * 1.1
        tgt_score = torch.where(beatable, tgt_score, torch.zeros_like(tgt_score))
        if not bool((tgt_score > 0).any()):
            continue
        best_tgt_local = int(tgt_score.argmax().item())
        tgt_i = int(tgt_indices[best_tgt_local].item())
        fleet_spd = _fleet_speed_scalar(send)
        dist_val  = float(d0[src_i, tgt_i].item())
        eta_val   = dist_val / max(fleet_spd, 1e-6)
        source_budget[src_i] = max(0.0, float(source_budget[src_i].item()) - send)
        planet_data = obs_tensors["planets"].reshape(-1, 7).to(dtype)
        sx = float(planet_data[src_i, 2].item())
        sy = float(planet_data[src_i, 3].item())
        tx = float(planet_data[tgt_i, 2].item())
        ty = float(planet_data[tgt_i, 3].item())
        angle_val = math.atan2(ty - sy, tx - sx)
        src_tensor = torch.tensor([src_i], dtype=torch.long, device=device)
        tgt_tensor = torch.tensor([tgt_i],  dtype=torch.long, device=device)
        send_t     = torch.tensor([send],  dtype=dtype,      device=device)
        eta_t      = torch.tensor([eta_val], dtype=dtype,    device=device)
        angle_t    = torch.tensor([angle_val], dtype=dtype,  device=device)
        valid_t    = torch.tensor([True],  dtype=torch.bool, device=device)
        empty = _empty_entries(device, dtype)
        entry = replace(
            empty,
            source_slots=src_tensor,
            target_slots=tgt_tensor,
            ships=send_t,
            eta=eta_t,
            angle=angle_t,
            valid=valid_t,
        )
        all_entries.append(entry)
    if not all_entries:
        return _empty_entries(device, dtype)
    return concat_launch_entries(all_entries)

def _build_defense_entries(*, movement: PlanetMovement, obs, cache, config: ProducerLiteConfig, player_count: int, source_budget: Tensor):
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    pid = int(obs.player_id)
    if P == 0:
        return _empty_entries(device, dtype)
    owned = obs.owned & obs.alive
    if not bool(owned.any()):
        return _empty_entries(device, dtype)
    H = min(int(config.defense_threat_horizon), int(movement.garrison_status(max_horizon=int(config.defense_threat_horizon)).ships.shape[-1]) - 1)
    if H <= 0:
        return _empty_entries(device, dtype)
    status    = movement.garrison_status(max_horizon=H)
    ships_at_H = status.ships[:, -1]
    threatened = owned & (ships_at_H < 0)
    if not bool(threatened.any()):
        return _empty_entries(device, dtype)
    tgt_indices = threatened.nonzero(as_tuple=False).squeeze(1)
    src_indices = owned.nonzero(as_tuple=False).squeeze(1)
    if src_indices.numel() == 0 or tgt_indices.numel() == 0:
        return _empty_entries(device, dtype)
    d0 = cache.cross_dist[0].to(dtype)
    all_entries = []
    waves_launched = 0
    for t_i in range(int(tgt_indices.shape[0])):
        if waves_launched >= int(config.defense_max_waves):
            break
        tgt = int(tgt_indices[t_i].item())
        deficit  = float(-ships_at_H[tgt].item())
        need     = deficit * float(config.defense_min_intercept_margin)
        src_ships = source_budget[src_indices].to(dtype)
        dists    = d0[src_indices, tgt]
        speeds   = fleet_speed(src_ships.clamp(min=1.0))
        etas     = (dists / speeds.clamp(min=1e-6)).ceil()
        can_arrive  = etas <= float(H)
        has_surplus = src_ships > (need + float(config.min_ships_to_launch))
        src_neq_tgt = src_indices != tgt
        valid_src   = can_arrive & has_surplus & src_neq_tgt
        if not bool(valid_src.any()):
            continue
        best_src_local = int(torch.where(valid_src, dists, torch.full_like(dists, 1e9)).argmin().item())
        best_src        = int(src_indices[best_src_local].item())
        send_ships = min(float(src_ships[best_src_local].item()) * 0.6, need + float(config.min_ships_to_launch))
        send_ships = float(math.floor(max(send_ships, float(config.min_ships_to_launch))))
        source_budget[best_src] = max(0.0, float(source_budget[best_src].item()) - send_ships)
        planet_data = obs_tensors.get("planets", torch.zeros((P, 7), device=device)).reshape(-1, 7).to(dtype)
        sx = float(planet_data[best_src, 2].item())
        sy = float(planet_data[best_src, 3].item())
        tx = float(planet_data[tgt, 2].item())
        ty = float(planet_data[tgt, 3].item())
        angle_val = math.atan2(ty - sy, tx - sx)
        eta_val = float(etas[best_src_local].item())
        src_tensor = torch.tensor([best_src], dtype=torch.long, device=device)
        tgt_tensor = torch.tensor([tgt], dtype=torch.long, device=device)
        send_t = torch.tensor([send_ships], dtype=dtype, device=device)
        eta_t = torch.tensor([eta_val], dtype=dtype, device=device)
        angle_t = torch.tensor([angle_val], dtype=dtype, device=device)
        valid_t = torch.tensor([True], dtype=torch.bool, device=device)
        empty = _empty_entries(device, dtype)
        entry = replace(
            empty,
            source_slots=src_tensor,
            target_slots=tgt_tensor,
            ships=send_t,
            eta=eta_t,
            angle=angle_t,
            valid=valid_t,
        )
        all_entries.append(entry)
        waves_launched += 1
    if not all_entries:
        return _empty_entries(device, dtype)
    return concat_launch_entries(all_entries)

def agent(obs, configuration) -> list:
    global obs_tensors
    device = torch.device("cpu")
    dtype = torch.float32

    obs_tensors = single_obs_to_tensor(obs, device)
    import orbit_lite.obs
    orbit_lite.obs.obs_tensors = obs_tensors

    parsed = parse_obs(obs)
    step = getattr(obs, "step", 0)
    player_count = getattr(configuration, "player_count", 2)
    player_id = parsed.player_id
    source_budget = parsed.ships.clone().to(dtype)

    if step >= (TOTAL_STEPS - config.endgame_dump_start):
        entries = _build_endgame_dump_entries(
            obs=parsed,
            obs_tensors=obs_tensors,
            cache=cache,
            config=config,
            source_budget=source_budget,
            step=step
        )
    else:
        movement = ensure_planet_movement(parsed, config.horizon)
        defense = _build_defense_entries(
            movement=movement,
            obs=parsed,
            cache=cache,
            config=config,
            player_count=player_count,
            source_budget=source_budget
        )
        economy = _build_economy_drain_entries(
            obs=parsed,
            obs_tensors=obs_tensors,
            cache=cache,
            garrison_status=movement.garrison_status(max_horizon=config.horizon),
            config=config,
            source_budget=source_budget,
            player_count=player_count
        )
        offense = _empty_entries(device, dtype)
        entries = concat_launch_entries([defense, economy, offense])

    sparse_row = entries_to_sparse_payload(entries, parsed)
    return sparse_action_row_to_moves(sparse_row, obs, player_id=player_id)

import shutil, subprocess, sys, tarfile, tempfile
from pathlib import Path

WORK = Path("/kaggle/working")
MAIN = WORK / "main.py"
ARCHIVE = WORK / "submission.tar.gz"

EXPECTED_ORBIT_LITE = {
    "__init__.py", "adapter.py", "aiming.py", "constants.py", "distance_cache.py",
    "garrison_launch.py", "geometry.py", "intercept_aim.py", "movement.py",
    "movement_aiming.py", "movement_step.py", "obs.py", "planner_core.py",
}
EXPECTED_MEMBERS = {"main.py"} | {f"orbit_lite/{name}" for name in EXPECTED_ORBIT_LITE}

def py_files(d: Path) -> set:
    return {p.name for p in d.glob("*.py")}

if not MAIN.is_file():
    print("=" * 72)
    print("ERROR: /kaggle/working/main.py does not exist.")
    print()
    print("This cell only PACKAGES main.py - it does not create it. Run the")
