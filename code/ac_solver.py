"""CasADi/IPOPT AC repair and OPF; independently checked by ac_physics.py."""
from __future__ import annotations
import time
import numpy as np
import casadi as ca
from ac_physics import angle_limit_masks, validate

SETTINGS = {'seed':42,'tol':1e-8,'max_iter':500,'max_cpu_time':15.0,
            'acceptance_tolerance':1e-5,'angle_distance_scale_rad':1.0,
            'distance_groups':['vm','va','pg','qg_bus']}

def distance(case, solution, target):
    scales = {'vm':np.maximum(np.subtract(case['vmax'],case['vmin']),.01),
              'va':np.ones(case['nb']),
              'pg':np.maximum(np.subtract(case['pmax'],case['pmin']),1.),
              'qg_bus':np.maximum(np.subtract(case['qmax_bus'],case['qmin_bus']),1.)}
    return float(np.mean([np.mean(((np.array(solution[k])-target[k])/s)**2) for k,s in scales.items()]))

def solve(case, prediction, method):
    t0=time.perf_counter()
    n,m,base=case['nb'],case['ng'],case['base_mva']
    a=lambda k:np.asarray(case[k],float)
    va=ca.MX.sym('va',n); vm=ca.MX.sym('vm',n)
    pg=ca.MX.sym('pg',m); qg=ca.MX.sym('qg',n)
    x=ca.vertcat(va,vm,pg,qg)
    b=case['branch']; f=np.array(b['from'],int); t=np.array(b['to'],int)
    d=va[f]-va[t]; vf=vm[f]; vt=vm[t]
    vprod=vf*vt
    pf=ca.DM(b['Gff'])*vf**2+vprod*(ca.DM(b['Gft'])*ca.cos(d)+ca.DM(b['Bft'])*ca.sin(d))
    qf=-ca.DM(b['Bff'])*vf**2+vprod*(ca.DM(b['Gft'])*ca.sin(d)-ca.DM(b['Bft'])*ca.cos(d))
    pt=ca.DM(b['Gtt'])*vt**2+vprod*(ca.DM(b['Gtf'])*ca.cos(d)-ca.DM(b['Btf'])*ca.sin(d))
    qt=-ca.DM(b['Btt'])*vt**2+vprod*(-ca.DM(b['Gtf'])*ca.sin(d)-ca.DM(b['Btf'])*ca.cos(d))
    cg=np.zeros((n,m)); cg[np.array(case['gen_bus'],int),np.arange(m)]=1
    cf=np.zeros((n,len(f))); ct=cf.copy()
    cf[f,np.arange(len(f))]=1; ct[t,np.arange(len(t))]=1
    pb=ca.DM(cg)@pg-a('pd')/base-a('gsh')*vm**2/base-(ca.DM(cf)@pf+ca.DM(ct)@pt)/base
    qb=qg-a('qd')/base+a('bsh')*vm**2/base-(ca.DM(cf)@qf+ca.DM(ct)@qt)/base
    constraints=[pb,qb]; lo=[np.zeros(n),np.zeros(n)]; hi=[np.zeros(n),np.zeros(n)]
    rate=np.array(b['rate']); thermal=np.flatnonzero(rate>0)
    for p,q in ((pf,qf),(pt,qt)):
        constraints.append((p[thermal]**2+q[thermal]**2)/base**2)
        lo.append(np.full(len(thermal),-np.inf)); hi.append((rate[thermal]/base)**2)
    lm,um=angle_limit_masks(case); angles=np.flatnonzero(lm|um)
    if len(angles):
        constraints.append(d[angles]); lo.append(np.where(lm[angles],np.array(b['angmin'])[angles],-np.inf));hi.append(np.where(um[angles],np.array(b['angmax'])[angles],np.inf))
    lb=np.r_[np.full(n,-np.inf),a('vmin'),a('pmin')/base,a('qmin_bus')/base]
    ub=np.r_[np.full(n,np.inf),a('vmax'),a('pmax')/base,a('qmax_bus')/base]
    for i,ang in zip(case['ref_bus'],case['ref_angle']): lb[i]=ub[i]=ang
    target={k:np.asarray(v,float).copy() for k,v in prediction.items()}
    # Predicted angle output already uses the source reference gauge.
    ds=[ca.sumsqr((vm-target['vm'])/np.maximum(a('vmax')-a('vmin'),.01))/n,
        ca.sumsqr(va-target['va'])/n,
        ca.sumsqr((pg*base-target['pg'])/np.maximum(a('pmax')-a('pmin'),1))/m,
        ca.sumsqr((qg*base-target['qg_bus'])/np.maximum(a('qmax_bus')-a('qmin_bus'),1))/n]
    projection=sum(ds)/4
    cost=ca.sum1(a('c0')+a('c1')*pg*base+a('c2')*(pg*base)**2)
    cost_scale=max(1.,float(np.sum(np.abs(a('c0'))+np.abs(a('c1'))*np.abs(a('pmax'))+np.abs(a('c2'))*a('pmax')**2)))
    objective=projection if method=='repair' else cost/cost_scale
    if method=='cold_opf':
        x0=np.r_[np.zeros(n),np.ones(n),(a('pmin')+a('pmax'))/2/base,(a('qmin_bus')+a('qmax_bus'))/2/base]
    else: x0=np.r_[target['va'],target['vm'],target['pg']/base,target['qg_bus']/base]
    x0=np.minimum(np.maximum(x0,lb),ub)
    opts={'print_time':False,'error_on_fail':False,'ipopt.print_level':0,'ipopt.sb':'yes',
          'ipopt.tol':SETTINGS['tol'],'ipopt.constr_viol_tol':SETTINGS['tol'],
          'ipopt.acceptable_tol':1e-7,'ipopt.max_iter':SETTINGS['max_iter'],
          'ipopt.max_cpu_time':SETTINGS['max_cpu_time'],'ipopt.bound_relax_factor':0.,
          'ipopt.fixed_variable_treatment':'make_parameter'}
    solver=ca.nlpsol('ac','ipopt',{'x':x,'f':objective,'g':ca.vertcat(*constraints)},opts)
    built=time.perf_counter()
    answer=solver(x0=x0,lbx=lb,ubx=ub,lbg=np.concatenate(lo),ubg=np.concatenate(hi))
    solved=time.perf_counter(); stats=solver.stats(); vals=np.asarray(answer['x']).ravel()
    solution={'va':vals[:n].tolist(),'vm':vals[n:2*n].tolist(),'pg':(vals[2*n:2*n+m]*base).tolist(),'qg_bus':(vals[2*n+m:]*base).tolist()}
    extracted=time.perf_counter(); check=validate(case,solution,tolerance=SETTINGS['acceptance_tolerance'])
    checked=time.perf_counter()
    its=stats.get('iterations',{})
    return {'solution':solution,'validation':check,'solver_status':stats.get('return_status'),
            'solver_success':bool(stats.get('success',False)),'iterations':int(stats.get('iter_count',-1)),
            'distance_objective':distance(case,solution,target),
            'solver_objective':float(answer['f']),
            'final_inf_pr':float(its.get('inf_pr',[float('nan')])[-1]),
            'final_inf_du':float(its.get('inf_du',[float('nan')])[-1]),
            'build_s':built-t0,'solve_s':solved-built,'extract_s':extracted-solved,
            'validation_s':checked-extracted,'optimization_pipeline_s':checked-t0}
