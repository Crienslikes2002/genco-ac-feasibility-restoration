"""Independent AC model extraction and NumPy feasibility audit.

Schema v1: powers MW/MVar, voltage p.u., angles radians, Y coefficients
scaled by the original baseMVA. Thus V*conj(YV) directly gives MW+j*MVar.
Reactive generation is a BUS aggregate, exactly as exposed by GraphKit.
No unobserved per-generator Q limits or capability curves are invented.

Sources: GraphKit datasets/{globals,hetero_preprocess,normalizers}.py;
MATPOWER standard AC OPF/network equations and makeAang conventions.
This module does not call the neural physics decoder or NLP symbolic code.
"""
from __future__ import annotations
import math
import numpy as np


def _np(x, dtype=float):
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


def _list(x):
    return np.asarray(x).tolist()


def from_graph(graph, unit_state='physical', base_mva_orig=100.0):
    """Return ``(case, reference_solution)`` as JSON-serializable dictionaries.

    graph is one HeteroData sample, not a Batch. Accepted unit states:
    physical: AFTER HeteroDataMVANormalizer.inverse_transform: power MW,
              bus angle rad, edge angle limits degrees, Y and shunts scaled.
    raw: unnormalized raw processed graph: power MW, both angles degrees,
         Y and shunts on original p.u. base.
    normalized: training-normalized graph: graph.baseMVA must be present;
                power and Y use learned base, both angles radians, log costs.
    Active branch/generator filtering may already have been applied. Input
    target masking does not affect the physical parameters extracted here.
    """
    if unit_state not in ('physical', 'raw', 'normalized'):
        raise ValueError('unit_state must be physical, raw, or normalized')
    bus = _np(graph['bus'].x).copy()
    gen = _np(graph['gen'].x).copy()
    by = _np(graph['bus'].y).copy()
    gy = _np(graph['gen'].y).reshape(-1).copy()
    ei = _np(graph['bus', 'connects', 'bus'].edge_index, int)
    ea = _np(graph['bus', 'connects', 'bus'].edge_attr).copy()
    gb = _np(graph['gen', 'connected_to', 'bus'].edge_index, int)
    nb, ng0 = len(bus), len(gen)
    gen_bus = np.full(ng0, -1, dtype=int)
    for gi, bi in gb.T:
        if gen_bus[gi] >= 0:
            raise ValueError('Generator has multiple gen-to-bus edges')
        gen_bus[gi] = bi
    if np.any(gen_bus < 0):
        raise ValueError('Generator lacks a bus mapping')
    active_gen = gen[:, 6] == 1 if gen.shape[1] > 6 else np.ones(ng0, bool)
    gen, gy, gen_bus = gen[active_gen], gy[active_gen], gen_bus[active_gen]
    if ea.shape[1] > 10:
        active_edge = ea[:, 10] == 1
        ei, ea = ei[:, active_edge], ea[active_edge]
    if len(ea) % 2:
        raise ValueError('Expected forward/reverse edge pairs')
    nl = len(ea) // 2
    if not np.array_equal(ei[:, :nl], ei[::-1, nl:]):
        raise ValueError('Graph edges are not [forward, corresponding reverse]')
    if not np.allclose(ea[:nl, 9], ea[nl:, 9], rtol=1e-6, atol=1e-6):
        raise ValueError('Forward/reverse thermal limits disagree')
    if unit_state == 'normalized':
        learned_base = float(_np(graph.baseMVA).reshape(-1)[0])
        power_scale, y_scale = learned_base, learned_base
        bus_angle_scale, edge_angle_scale = 1.0, 1.0
        cost = np.sign(gen[:, 3:6]) * np.expm1(np.abs(gen[:, 3:6]))
    else:
        learned_base = None
        power_scale = 1.0
        y_scale = float(base_mva_orig) if unit_state == 'raw' else 1.0
        bus_angle_scale = math.pi / 180 if unit_state == 'raw' else 1.0
        edge_angle_scale = math.pi / 180
        cost = gen[:, 3:6]
    ref = np.flatnonzero(bus[:, 7] == 1)
    if not len(ref):
        raise ValueError('No reference bus in source graph')
    source_angles = by[:, 4] * bus_angle_scale
    # Keep the reference values specified in the original graph, rather than
    # using any model output. OPF masking leaves reference-bus input angles.
    ref_angles = bus[ref, 4] * bus_angle_scale
    branch = {
        'from': _list(ei[0, :nl]), 'to': _list(ei[1, :nl]),
        'Gff': _list(ea[:nl, 2] * y_scale), 'Bff': _list(ea[:nl, 3] * y_scale),
        'Gft': _list(ea[:nl, 4] * y_scale), 'Bft': _list(ea[:nl, 5] * y_scale),
        'Gtt': _list(ea[nl:, 2] * y_scale), 'Btt': _list(ea[nl:, 3] * y_scale),
        'Gtf': _list(ea[nl:, 4] * y_scale), 'Btf': _list(ea[nl:, 5] * y_scale),
        'rate': _list(ea[:nl, 9] * power_scale),
        'angmin': _list(ea[:nl, 7] * edge_angle_scale),
        'angmax': _list(ea[:nl, 8] * edge_angle_scale),
    }
    case = {
        'schema_version': 1, 'nb': nb, 'ng': len(gen), 'nl': nl,
        'base_mva': float(base_mva_orig),
        'units': {'power': 'MW/MVar', 'angle': 'rad', 'voltage': 'p.u.',
                  'admittance': 'original p.u. admittance multiplied by base_mva'},
        'q_model': 'bus-aggregate reactive generation with source bus aggregate limits',
        'source_unit_state': unit_state, 'gen_bus': _list(gen_bus),
        'pd': _list(bus[:, 0] * power_scale), 'qd': _list(bus[:, 1] * power_scale),
        'pmin': _list(gen[:, 1] * power_scale), 'pmax': _list(gen[:, 2] * power_scale),
        'qmin_bus': _list(bus[:, 10] * power_scale),
        'qmax_bus': _list(bus[:, 11] * power_scale),
        'vmin': _list(bus[:, 8]), 'vmax': _list(bus[:, 9]),
        'c0': _list(cost[:, 0]), 'c1': _list(cost[:, 1]), 'c2': _list(cost[:, 2]),
        'gsh': _list(bus[:, 12] * y_scale), 'bsh': _list(bus[:, 13] * y_scale),
        'ref_bus': _list(ref), 'ref_angle': _list(ref_angles), 'branch': branch,
    }
    if hasattr(graph, 'scenario_id'):
        case['scenario_id'] = int(_np(graph.scenario_id).reshape(-1)[0])
    if learned_base is not None:
        case['normalization_base_mva'] = learned_base
    reference = {'vm': _list(by[:, 3]), 'va': _list(source_angles),
                 'pg': _list(gy * power_scale),
                 'qg_bus': _list(by[:, 2] * power_scale)}
    return case, reference


def admittance_matrices(case):
    """Build dense complex Ybus and the physical-line Yf/Yt from edge data."""
    n, b = case['nb'], case['branch']
    f, t = _np(b['from'], int), _np(b['to'], int)
    m = len(f)
    ff = _np(b['Gff']) + 1j * _np(b['Bff'])
    ft = _np(b['Gft']) + 1j * _np(b['Bft'])
    tf = _np(b['Gtf']) + 1j * _np(b['Btf'])
    tt = _np(b['Gtt']) + 1j * _np(b['Btt'])
    yf, yt = np.zeros((m, n), complex), np.zeros((m, n), complex)
    yf[np.arange(m), f] += ff
    yf[np.arange(m), t] += ft
    yt[np.arange(m), f] += tf
    yt[np.arange(m), t] += tt
    ybus = np.diag(_np(case['gsh']) + 1j * _np(case['bsh']))
    np.add.at(ybus, (f, f), ff)
    np.add.at(ybus, (f, t), ft)
    np.add.at(ybus, (t, f), tf)
    np.add.at(ybus, (t, t), tt)
    return ybus, yf, yt


def angle_limit_masks(case):
    """MATPOWER 7.1 makeAang row-selection and +/-360deg conventions.

    Standard [-360,+360] and [0,0] rows are omitted. When a row is active
    because of its other limit, an exact +/-360 endpoint is retained.
    Float32 conversion noise at the sentinel is removed for row selection.
    """
    lo, hi = _np(case['branch']['angmin']).copy(), _np(case['branch']['angmax']).copy()
    lo[np.isclose(lo, -2*math.pi, rtol=0, atol=1e-6)] = -2*math.pi
    hi[np.isclose(hi, 2*math.pi, rtol=0, atol=1e-6)] = 2*math.pi
    nzlo, nzhi = lo != 0, hi != 0
    active = ((nzlo & (lo > -2*math.pi)) | (nzhi & (hi < 2*math.pi))
              | (nzlo & ~nzhi) | (~nzlo & nzhi))
    return active & (lo >= -2*math.pi), active & (hi <= 2*math.pi)


def generation_cost(case, solution):
    p = _np(solution['pg'])
    return float(np.sum(_np(case['c0']) + _np(case['c1']) * p + _np(case['c2']) * p ** 2))


def compute_quantities(case, solution):
    """Complex NumPy power equations, independent of the NLP implementation."""
    vm, va = _np(solution['vm']), _np(solution['va'])
    pg, qg = _np(solution['pg']), _np(solution['qg_bus'])
    n = case['nb']
    if vm.shape != (n,) or va.shape != (n,) or qg.shape != (n,) or pg.shape != (case['ng'],):
        raise ValueError('Solution vector length disagrees with case dimensions')
    v = vm * np.exp(1j * va)
    ybus, yf, yt = admittance_matrices(case)
    inj = v * np.conj(ybus @ v)
    b = case['branch']
    f, t = _np(b['from'], int), _np(b['to'], int)
    sf, st = v[f] * np.conj(yf @ v), v[t] * np.conj(yt @ v)
    pbus = np.zeros(n)
    np.add.at(pbus, _np(case['gen_bus'], int), pg)
    return {'p_injection': inj.real, 'q_injection': inj.imag,
            'p_residual': pbus - _np(case['pd']) - inj.real,
            'q_residual': qg - _np(case['qd']) - inj.imag,
            'sf': sf, 'st': st, 'angle_difference': va[f] - va[t]}


def validate(case, solution, tolerance=1e-5, include_arrays=True):
    """Audit all represented constraints; power tolerance = tolerance*base_mva.

    voltage and angle tolerances are tolerance p.u./rad. Always returns raw
    maximum values and normalized maxima, so feasibility can be re-evaluated
    at another predeclared threshold without rerunning optimization.
    """
    q = compute_quantities(case, solution)
    base = float(case.get('base_mva', 100.0))
    vm, va = _np(solution['vm']), _np(solution['va'])
    pg, qg = _np(solution['pg']), _np(solution['qg_bus'])
    rate = _np(case['branch']['rate'])
    finite = all(np.all(np.isfinite(x)) for x in (vm, va, pg, qg))
    def bounds(x, lower, upper):
        return np.maximum(np.maximum(_np(lower) - x, x - _np(upper)), 0)
    thermal_f = np.where(rate > 0, np.maximum(np.abs(q['sf']) - rate, 0), 0)
    thermal_t = np.where(rate > 0, np.maximum(np.abs(q['st']) - rate, 0), 0)
    lm, um = angle_limit_masks(case)
    alo, ahi = _np(case['branch']['angmin']), _np(case['branch']['angmax'])
    da = q['angle_difference']
    av = np.maximum(np.where(lm, np.maximum(alo - da, 0), 0),
                    np.where(um, np.maximum(da - ahi, 0), 0))
    arrays = {
        'p_balance': np.abs(q['p_residual']), 'q_balance': np.abs(q['q_residual']),
        'p_bounds': bounds(pg, case['pmin'], case['pmax']),
        'q_bounds': bounds(qg, case['qmin_bus'], case['qmax_bus']),
        'voltage_bounds': bounds(vm, case['vmin'], case['vmax']),
        'thermal_from': thermal_f, 'thermal_to': thermal_t,
        'angle_bounds': av,
        'reference_angle': np.abs(va[_np(case['ref_bus'], int)] - _np(case['ref_angle'])),
    }
    power_keys = {'p_balance', 'q_balance', 'p_bounds', 'q_bounds', 'thermal_from', 'thermal_to'}
    raw_max = {key: float(np.max(value, initial=0)) for key, value in arrays.items()}
    norm_max = {key: value / base if key in power_keys else value for key, value in raw_max.items()}
    finite = finite and all(math.isfinite(x) for x in raw_max.values())
    result = {
        'feasible': bool(finite and max(norm_max.values(), default=0) <= tolerance),
        'finite': bool(finite), 'tolerance': float(tolerance),
        'power_tolerance_mw_mvar_mva': tolerance * base,
        'max_violation': raw_max, 'max_violation_normalized': norm_max,
        'overall_max_normalized': max(norm_max.values(), default=0),
        'cost': generation_cost(case, solution),
        'constraint_units': {key: ('MW/MVar/MVA' if key in power_keys else
                                  'p.u.' if key == 'voltage_bounds' else 'rad') for key in arrays},
    }
    if include_arrays:
        result['violations'] = {key: _list(value) for key, value in arrays.items()}
        result['flows'] = {'pf': _list(q['sf'].real), 'qf': _list(q['sf'].imag),
                           'pt': _list(q['st'].real), 'qt': _list(q['st'].imag),
                           'sf': _list(np.abs(q['sf'])), 'st': _list(np.abs(q['st']))}
        result['p_residual'] = _list(q['p_residual'])
        result['q_residual'] = _list(q['q_residual'])
    return result
