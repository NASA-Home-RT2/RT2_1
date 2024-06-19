# import packagess
import random
import pandas as pd
import numpy as np
from numpy import genfromtxt
import gurobipy as gp
from gurobipy import GRB
import pickle
import math
import time
import fileinput
import sys
from copy import deepcopy 

def conditional_cdf(z):
    cdf=np.ones((100,400));
    for i in range(100):
        loc = z>i;
        if max(loc)==0:
            break
        x=z[loc]-i;
        total = len(x);
        for t in range(400):
            cdf[i,t]=np.sum(x<=t)/total
    return cdf


## Loading CDFs:
prior_rul_cdf = {}
data_rul_cdf = {}
deg_signals = {}
lifespan = {}
prog_model = ['brownian', 'model']
deg_trend = [['bm'], ['linear', 'paris_law']]

for i in range(2):
    pmodel = prog_model[i]
    for j in range(len(deg_trend[i])):
        trend = deg_trend[i][j]
        if trend == 'linear':
            nfam = 3
        else:
            nfam = 2
        for k in range(nfam):
            prior_rul_cdf[i, j, k] = np.array(pd.read_csv(f"./data/{pmodel}/{trend}/family{k}/prior_cdf.csv", header=None))
            N_s = 10000
            cdf = prior_rul_cdf[i, j, k][199]
            cdf[-1] = 1
            pmf = deepcopy(cdf)
            pmf[1:] = deepcopy(cdf[1:]) - deepcopy(cdf[:-1])
            elements = range(1, 401)
            sampled_scenarios = np.random.choice(elements, N_s, p=pmf)
            complement_prior       = conditional_cdf(sampled_scenarios)
            prior_rul_cdf[i, j, k] =np.concatenate((prior_rul_cdf[i, j, k] , complement_prior), axis=0)
            deg_signals[i, j, k] = np.array( pd.read_csv(f"./data/{pmodel}/{trend}/family{k}/degradation_signals.csv", header=None))
            for n in range(100):
                data_rul_cdf[i, j, k, n] = np.array( pd.read_csv(f"./data/{pmodel}/{trend}/family{k}/data_cdf_{n + 1}.csv", header=None))
                lifespan[i, j, k, n] = data_rul_cdf[i, j, k, n].shape[0]
print("===============================\nDATA HAS BEEN READ CORRECTLY\n===============================")


# Constructing an arbitrary asset:
class Asset:
    def __init__(self, id_number, type_prog, trend, k, fam, nmax):
        self.id_number = id_number
        self.i = type_prog
        self.j = trend
        self.k = k
        self.fam = fam
        self.nmax = nmax
        # Set the initial parameters of the asset:
        self.n = random.randint(0, 99)
        self.idc = (self.i, self.j, self.k, self.n)
        # initiate its age randomly:
        self.lifetime = int(lifespan[self.i, self.j, self.k, self.n])
        self.age = random.randint(1, int(self.lifetime * .2))
        self.status = 1

    def update_age(self):
        self.age += 1
        if self.age >= self.lifetime:
            self.status = 0

    def repair(self):
        tf = self.lifetime
        tr = self.age
        # Repair the asset. It is as good as new now.
        self.n = random.randint(0, 99)
        self.lifetime = int(lifespan[self.i, self.j, self.k, self.n])
        self.idc = (self.i, self.j, self.k, self.n)
        self.age = 1
        self.status = 1
        return tf, tr

    def __str__(self):
        return f" Asset {self.id_number} **** Active: {self.status}"


# Constructing an arbitrary system composed of multiple assets:

class System:
    def __init__(self, components, on_hand_inventory):
        self.components = components

        # Initiate the system:
        self.x_inv = on_hand_inventory
        self.time = 0
        self.asset = []
        self.family = {}
        self.asset_fam = []
        self.F = []
        self.nrepair_f = []
        s = 0
        f = 0
        for r in self.components:
            i, j, k, n, nmax = r[0], r[1], r[2], r[3], r[4]
            self.nrepair_f.append(nmax)
            self.family[f] = []
            for c in range(n):
                self.family[f].append(s)
                self.asset.append(Asset(s, i, j, k, f, nmax))
                self.asset_fam.append(f)
                s += 1
            self.F.append(f)
            f += 1
        self.M = range(s)

    def update(self, maintenances, orders):
        self.time += 1
        # update inventory with new orders:

        for f in self.F:
            self.x_inv[f] += orders[f]
        report = []
        status = []
        # update age of the assets
        for m in self.M:
            f = self.asset_fam[m]
            if ((m,0) in maintenances) and (self.x_inv[f] > 0):
                tf, tr = self.asset[m].repair()
                self.x_inv[f] += -1
                report.append((m, tf, tr, self.time,0))
            elif ((m,1) in maintenances) and (self.x_inv[f] > 0):
                tf, tr = self.asset[m].repair()
                self.x_inv[f] += -1
                report.append((m, tf, tr, self.time,1))
            else:
                self.asset[m].update_age()
            status.append((m, self.asset[m].status, self.time))

        return status, report

    def __str__(self):
        return f" System  composed of {self.n_assets} assets"


def optimization_module(system, optimization_parameters, time_limit,cb, mode):
    tinitial = time.time()
    printt = optimization_parameters['print gurobi optimization results']
    T_max = optimization_parameters['T_max']
    N_s = optimization_parameters['N_s']

    # Orders related parameters
    previous_orders = optimization_parameters['previous_orders']
    fixed_order = optimization_parameters['fixed_order']
    next_time_order = optimization_parameters['next_time_order']
    dt_order = optimization_parameters['dt_order']
    Y_max = optimization_parameters['Y_max']
    V_j = optimization_parameters['V_j']

    # Inventory related parameters:
    on_hand_inventory = optimization_parameters['on_hand_inventory']
    # Penalty costs paramters:
    # lamb = optimization_parameters['lambda']
    ce = optimization_parameters['ce']
    cl = optimization_parameters['cl']
    cp = optimization_parameters['cp']
    cc = optimization_parameters['cc']
    holding_cost = optimization_parameters['holding_cost']
    p_exp = optimization_parameters['p_exp']

    # Chance constraints parameters:

    rho = deepcopy(optimization_parameters['rho'])
    epsilon = deepcopy(optimization_parameters['epsilon'])

    # Maintenance crew
    S_hum = optimization_parameters['S_hum']
    S_rob = optimization_parameters['S_rob']
    rep_cost_hum = optimization_parameters['repair_cost'][:, 0]
    rep_cost_rob = optimization_parameters['repair_cost'][:, 1]
    g_m_hum = optimization_parameters['doable_tasks'][0, :]
    g_m_rob = optimization_parameters['doable_tasks'][1, :]
    A_t_hum = optimization_parameters['crew_availibility'][0, :]
    A_t_rob = optimization_parameters['crew_availibility'][1, :]
    T_m     = optimization_parameters['t_m']
    
    # 0. Set the times of the next orders arrivals:
    order_times = np.arange(next_time_order, T_max, dt_order)
    T_next_orders = range(len(order_times))

    # 1. Index sets:
    F = system.F
    M = system.M
    T = range(1, T_max + 1)
    S = range(N_s)

    # 1.1. Decision variables index sets:
    Lx = []
    Lz = []
    Lzz = []
    Lr = []
    for m in M:
        n_max = system.asset[m].nmax
        for k in range(1, n_max + 1):
            Lzz.append((m, k))
            for t in T:
                Lx.append((m, k, t))
                Lz.append((m, k, t))
        for k in range(n_max + 1):
            for t in range(T_max + 1):
                Lr.append((m, k, t))

    # 3. Optimization program

    # 3.1 LP parameters:

    # 3.2 Initiating optimizer object:
    mod = gp.Model('Planning_module')

    # 3.3. Decision variables

    z_mkt = mod.addVars(Lz, vtype=GRB.BINARY, name="timebetweenrepairs")

    z_mk = mod.addVars(Lzz, vtype=GRB.BINARY, name="install_k")

    x_mkt = mod.addVars(Lx, vtype=GRB.BINARY, name="install_k_at_t")

    x_mkt_b = mod.addVars(Lx, vtype=GRB.BINARY, name="track_repairs")

    r_mkt = mod.addVars(Lr, vtype=GRB.BINARY, name="last_repair_installed")

    x_mkt_rob = mod.addVars(Lx, vtype=GRB.BINARY, name="task_to_robot")

    x_mkt_hum = mod.addVars(Lx, vtype=GRB.BINARY, name="task_to_human")

    x_t_rob = mod.addVars(T, vtype=GRB.BINARY, name="setup_robot")

    x_t_hum = mod.addVars(T, vtype=GRB.BINARY, name="setup_hum")

    q_jt = mod.addVars(F, T, vtype=GRB.CONTINUOUS, name="inventory_level")

    y_jt = mod.addVars(F, T_next_orders, vtype=GRB.INTEGER, name="orders")

    ye_jt = mod.addVars(F, T, vtype=GRB.INTEGER, name="expedited_order")

    # 3.4. Constraints

    for m in M:
        (i, j, r, n) = system.asset[m].idc
        n_max = system.asset[m].nmax
        # Eq. 1:
        mod.addConstrs(z_mk[m, k] + gp.quicksum(z_mkt[m, k, t] for t in T) == 1 for k in range(1, n_max + 1))
        # Eq. 2:
        mod.addConstrs(z_mk[m, k] >= z_mk[m, k - 1] for k in range(2, n_max + 1))
        # Eq. 3:
        mod.addConstrs(
            gp.quicksum((x_mkt_b[m, k, t]) * t for t in T) >= gp.quicksum((x_mkt_b[m, k - 1, t]) * t for t in T) for k
            in range(2, n_max + 1))
        mod.addConstrs(gp.quicksum(x_mkt_b[m, k, t] for t in T) == 1 for k in range(1, n_max + 1))
        # Eq. 4
        mod.addConstrs(gp.quicksum(z_mkt[m, k, t] * t for t in T) == gp.quicksum((x_mkt_b[m, k, t]) * t for t in T) \
                       - gp.quicksum((x_mkt_b[m, k - 1, t]) * t for t in T) for k in range(2, n_max + 1))

        mod.addConstrs(z_mkt[m, 1, t] == x_mkt[m, 1, t] for t in T)
        mod.addConstrs(z_mkt[m, 1, t] <= x_mkt_b[m, 1, t] for t in T)
        # Eq. 5:
        mod.addConstr(r_mkt[m, 0, 1] == z_mk[m, 1])
        mod.addConstrs(r_mkt[m, n_max, t] == x_mkt[m, n_max, t] for t in T)

        mod.addConstrs(r_mkt[m, k, t] <= x_mkt[m, k, t] for t in T for k in range(1, n_max))
        mod.addConstrs(r_mkt[m, k, t] <= z_mk[m, k + 1] for t in T for k in range(1, n_max))
        mod.addConstrs(z_mk[m, k + 1] + x_mkt[m, k, t] - 1 <= r_mkt[m, k, t] for t in T for k in range(1, n_max))

        # Eq 6
        mod.addConstrs(
            x_mkt_b[m, k, t] - z_mk[m, k] >= -2 * (1 - x_mkt[m, k, t]) + 1 for t in T for k in range(2, n_max + 1))
        mod.addConstrs(x_mkt_b[m, k, t] - z_mk[m, k] <= x_mkt[m, k, t] for t in T for k in range(2, n_max + 1))
        # eq7

        # accounting for time availability constraints: ...........................................................
        mod.addConstrs(x_mkt_rob[m, k, t] <= g_m_rob[m]*x_t_rob[t] for t in T for k in range(1, n_max + 1))
        mod.addConstrs(x_mkt_hum[m, k, t] <= g_m_hum[m]*x_t_hum[t] for t in T for k in range(1, n_max + 1)) 
        
        mod.addConstrs( x_mkt_rob[m, k, t] + x_mkt_hum[m, k, t] == x_mkt[m, k, t] for t in T for k in range(1, n_max + 1))
    mod.addConstrs( gp.quicksum(x_mkt_rob[m, k, t]*T_m[1,m] for m in M for k in range(1, system.asset[m].nmax + 1)) <= A_t_rob[t - 1]*x_t_rob[t] for t in T)
    mod.addConstrs(  gp.quicksum(x_mkt_hum[m, k, t]*T_m[0,m] for m in M for k in range(1, system.asset[m].nmax + 1)) <= A_t_hum[t - 1]*x_t_hum[t] for t in T)

    # Inventory Constraints  ..........................................................................................
    for j in F:
        fam = system.family[j]
        n_max = system.nrepair_f[j]
        for t in T:
            if t == 1:
                mod.addConstr(q_jt[j, t] == system.x_inv[j] + ye_jt[j, t] + gp.quicksum(
                    y_jt[j, i] * int(order_times[i] == t) for i in T_next_orders) - gp.quicksum(
                    x_mkt[m, k, t] for m in fam for k in range(1, n_max + 1)))
            else:
                mod.addConstr(q_jt[j, t] == q_jt[j, t - 1] + ye_jt[j, t] + gp.quicksum(
                    y_jt[j, i] * int(order_times[i] == t) for i in T_next_orders) - gp.quicksum(
                    x_mkt[m, k, t] for m in fam for k in range(1, n_max + 1)))
            mod.addConstr(q_jt[j, t] >= 0)

    #  payload constraint:
    mod.addConstrs(gp.quicksum(y_jt[j, t] * V_j[j] for j in F) <= Y_max for t in T_next_orders)
    if fixed_order == 1:
        mod.addConstrs(y_jt[j, 0] == np.round(previous_orders[j, 0], 0) for j in F)
    if min(mode)==0:
        # chance constraints paramaters
        for m in M:
            rho[1] += 1 - system.asset[m].status

        big_M1 = np.zeros((len(M), N_s))
        bigM2 = 0
        for m in M:
            (i, j, r, n) = system.asset[m].idc
            n_max = system.asset[m].nmax
            bigM2 += n_max
            big_M1[m, :] = T_max + 1 - rho[0][m]

        big_M2 = np.ones(N_s) * bigM2 + 1 - rho[1]
    ## generate scenarios:
        ft_s_mk = {}
        for m in system.M:
            (i, j, r, n) = system.asset[m].idc
            n_max = system.asset[m].nmax
            prior_rul = prior_rul_cdf[i, j, r][0]
            pmf_prior = deepcopy(prior_rul)
            pmf_prior[1:] = deepcopy(pmf_prior[1:]) - deepcopy(pmf_prior[:-1])
            for k in range(n_max + 1):
                if k > 0:
                    cdf = data_rul_cdf[i, j, r, n][0]
                    cdf[-1] = 1
                    pmf = deepcopy(cdf)
                    pmf[1:] = deepcopy(cdf[1:]) - deepcopy(cdf[:-1])
                    elements = range(1, 401)
                    sampled_scenarios = np.random.choice(elements, N_s, p=pmf)
                    ft_s_mk[m, k] = sampled_scenarios
                else:
                    if system.asset[m].status == 1:
                        age = system.asset[m].age
                        if cb == 1:
                            cdf = data_rul_cdf[i, j, r, n][age]
                        else:
                            cdf = prior_rul_cdf[i, j, r][age]
                        cdf[-1] = 1
                        pmf = deepcopy(cdf)
                        pmf[1:] = deepcopy(cdf[1:]) - deepcopy(cdf[:-1])
                        elements = range(1, 401)
                        sampled_scenarios = np.random.choice(elements, N_s, p=pmf)
                        ft_s_mk[m, k] = sampled_scenarios
                    else:
                        ft_s_mk[m, k] = np.zeros(N_s)
    # cc 1
    if mode[0] == 0: 
        mu_ms = mod.addVars(M, S, vtype=GRB.BINARY, name="chance_constraint_counter")
        for m in M:
            (i, j, r, n) = system.asset[m].idc
            n_max = system.asset[m].nmax
            for s in S:
                u_ms = gp.quicksum(
                    z_mkt[m, k, t] * max(0, int(t - ft_s_mk[m, k - 1][s])) for k in range(1, n_max + 1) for t in
                    T) + \
                       gp.quicksum(
                           r_mkt[m, k, t] * max(0, int((T_max - t) - ft_s_mk[m, k][s])) for k in range(n_max + 1)
                           for t
                           in range(T_max + 1))
                mod.addConstr(rho[0][m] - u_ms >= 1 - big_M1[m, s] * mu_ms[m, s])
            mod.addConstr(gp.quicksum(mu_ms[m, s] for s in S) <= int(epsilon[0][m] * len(S)))
    elif mode[0] == 1:
        for m in M:
            (i, j, r, n) = system.asset[m].idc
            n_max = system.asset[m].nmax
            fam = system.asset[m].fam
            CC = 0
            for k in range(1, n_max + 1):
                if k == 1:
                    if system.asset[m].status == 1:
                        age = system.asset[m].age
                        if cb == 1:
                            cdf = data_rul_cdf[i, j, r, n][age]
                        else:
                            cdf = prior_rul_cdf[i, j, r][age]
                        cdf[-1] = 1
                        pmf = deepcopy(cdf)
                        pmf[1:] = deepcopy(cdf[1:]) - deepcopy(cdf[:-1])
                        for t in T:
                            late1 = np.array([max(0, t - t_mkt) for t_mkt in range(1, 401)])
                            a1   = np.round(np.sum(np.multiply(late1, pmf)),3)
                            late2 = np.array([max(0, (T_max-t) - t_mkt) for t_mkt in range(1, 401)])
                            a2   = np.round(np.sum(np.multiply(late2, pmf)) ,3)  
                            CC += z_mkt[m, 1, t]*a1 + a2*r_mkt[m,0,t]
                    else:
                        for t in T:
                            CC += z_mkt[m, 1, t] *t
                else:
                    cdf = prior_rul_cdf[i, j, r][0]
                    cdf[-1] = 1
                    pmf = deepcopy(cdf)
                    pmf[1:] = deepcopy(cdf[1:]) - deepcopy(cdf[:-1])
                    for t in T:
                        late1 = np.array([max(0, t - t_mkt) for t_mkt in range(1, 401)])
                        a1   = np.round(np.sum(np.multiply(late1, pmf)),3) 
                        late2 = np.array([max(0, (T_max-t) - t_mkt) for t_mkt in range(1, 401)])
                        a2   = np.round(np.sum(np.multiply(late2, pmf)) ,3) 
                        CC += z_mkt[m, k, t]*a1 + a2*r_mkt[m,k-1,t]
            cdf = prior_rul_cdf[i, j, r][0]
            cdf[-1] = 1
            pmf = deepcopy(cdf)
            pmf[1:] = deepcopy(cdf[1:]) - deepcopy(cdf[:-1])
            for t in T: 
                late2 = np.array([max(0, (T_max-t) - t_mkt) for t_mkt in range(1, 401)])
                a2   = np.round(np.sum(np.multiply(late2, pmf)) ,3) 
                CC += a2*r_mkt[m,n_max,t]
            mod.addConstr(CC<=epsilon[0][m]*rho[0][m])

    if mode[1]==0:
        l_s = mod.addVars(S, vtype=GRB.BINARY, name="chance_constraint_counter2")
        # Chance constraint 2:
        # cc2:
        for s in S:
            corr = 0
            total = 0
            for m in M:
                (i, j, r, n) = system.asset[m].idc
                n_max = system.asset[m].nmax
                corr += gp.quicksum(
                    z_mkt[m, k, t] * int((t - ft_s_mk[m, k - 1][s]) > 0) for k in range(1, n_max + 1) for t in T)
            mod.addConstr(rho[1] - corr >= 1 - big_M2[s] * l_s[s])
        mod.addConstr(gp.quicksum(l_s[s] for s in S) <= int(epsilon[1] * len(S)))
    elif mode[1] == 1:
        bigM2       = 0
        for m in M:
            bigM2  += system.asset[m].nmax
         
        # Computing the bound of safe approx:
        f1=-1e+10
        da=0.001
        df = 1
        alpha = 0.0
        N_tot = bigM2 
        while df>=0:
            alpha+=da
            f2    = (((epsilon[1]*np.exp(alpha*rho[1]) )**(1/N_tot)) - 1)*N_tot/(np.exp(alpha)-1)
            df    = f2-f1
            f1    = f2
        alpha -=da
        f2     = (((epsilon[1]*np.exp(alpha*rho[1]) )**(1/N_tot)) - 1)*N_tot/(np.exp(alpha)-1)
        Theta  = max(rho[1]*epsilon[1],f2)
        # corrective repairs :
        CC = 0
        sure_corr = 0
        for m in M:
            (i, j, r, n) = system.asset[m].idc
            n_max = system.asset[m].nmax
            fam = system.asset[m].fam
            for k in range(1, n_max + 1):
                if k == 1:
                    if system.asset[m].status == 1:
                        age = system.asset[m].age
                        if cb == 1:
                            cdf = data_rul_cdf[i, j, r, n][age]
                        else:
                            cdf = prior_rul_cdf[i, j, r][age]
                        for t in T: 
                            CC += z_mkt[m, 1, t]*np.round(cdf[t],3) 
                    else:
                        sure_corr+=1
                        for t in T:
                            CC += z_mkt[m, 1, t] *1
                else:
                    cdf = prior_rul_cdf[i, j, r][0] 
                    for t in T: 
                        CC += z_mkt[m, k, t]*np.round(cdf[t],3) 
        mod.addConstr( CC<=(Theta+sure_corr))
        
    if mode[0]==2 and mode[1]==2:
        mod.addConstrs( gp.quicksum((x_mkt[m,1, t]) for t in T)==1 for m in M)
    # 3.5. Objective function

    # holding cost (dont depend on the scenarios)
    F_holding_cost = gp.quicksum(q_jt[j, t] * holding_cost[j] for j in F for t in T)

    # Dynamic costs:
    F_costs = 0

    for m in M:
        (i, j, r, n) = system.asset[m].idc
        n_max = system.asset[m].nmax
        fam = system.asset[m].fam
        for k in range(1, n_max + 1):
            if k == 1:
                if system.asset[m].status == 1:
                    age = system.asset[m].age
                    if cb == 1:
                        cdf = data_rul_cdf[i, j, r, n][age]
                    else:
                        cdf = prior_rul_cdf[i, j, r][age]
                    cdf[-1] = 1
                    pmf = deepcopy(cdf)
                    pmf[1:] = deepcopy(cdf[1:]) - deepcopy(cdf[:-1])
                    for t in T:
                        late = np.array([max(0, t - t_mkt) for t_mkt in range(1, 401)])
                        prem = np.array([max(0, t_mkt - t) for t_mkt in range(1, 401)])
                        alpha_mkt = ce[fam] * np.sum(np.multiply(prem, pmf)) + cl[fam] * np.sum(np.multiply(late, pmf)) + \
                                    cp[fam] + (cc[fam] - cp[fam]) * cdf[t]
                        F_costs += z_mkt[m, 1, t] * alpha_mkt
                else:
                    for t in T:
                        F_costs += z_mkt[m, 1, t] * (cc[fam] + cl[fam] * t)
            else:
                cdf = prior_rul_cdf[i, j, r][0]
                cdf[-1] = 1
                pmf = deepcopy(cdf)
                pmf[1:] = deepcopy(cdf[1:]) - deepcopy(cdf[:-1])
                for t in T:
                    late = np.array([max(0, t - t_mkt) for t_mkt in range(1, 401)])
                    prem = np.array([max(0, t_mkt - t) for t_mkt in range(1, 401)])
                    alpha_mkt = ce[fam] * np.sum(np.multiply(prem, pmf)) + cl[fam] * np.sum(np.multiply(late, pmf)) + \
                                cp[fam] + (cc[fam] - cp[fam]) * cdf[t]
                    F_costs += z_mkt[m, k, t] * alpha_mkt

    # expedited order:
    F_expedited = gp.quicksum(ye_jt[j, t] * p_exp for j in F for t in T)

    # Repair cost:
    F_rep = 0
    for m in M:
        (i, j, r, n) = system.asset[m].idc
        n_max = system.asset[m].nmax
        F_rep += gp.quicksum(rep_cost_rob[m] * x_mkt_rob[m, k, t] + rep_cost_hum[m] * x_mkt_hum[m, k, t] for k in range(1, n_max + 1) for t in T)

    F_setup = gp.quicksum(x_t_rob[t] * S_rob + x_t_hum[t] * S_hum for t in T)

    # TOTAL COSTS;
    Ftot = -(F_holding_cost + F_expedited + F_costs + F_rep + F_setup)
    mod.update()

    # 4. OPTIMIZING:
    mod.Params.LogToConsole = 0
    start_time = time.time()
    mod.setParam('TimeLimit', time_limit)
    mod.setObjective(Ftot, GRB.MAXIMIZE)
    mod.setParam('MIPGap',0.01)
    mod.update()
    mod.optimize()
    mod.update()
    runningtime = (time.time() - start_time)
    runningtime1 = (time.time() - tinitial)

    try:
        w = y_jt[0, 0].x
    except AttributeError:
        print('LP unfeasible due to inventory shortage')
        raise KeyboardInterrupt
    # Returning optimization outcomes:
    calendar = []
    calendar_rob = []
    calendar_hum = []
    exp_orders = {}
    orders = {}
    con = 0
    for t in range(len(order_times)):
        orde = []
        for j in F:
            orders[j, t] = y_jt[j, t].x
    for t in T:
        for j in F:
            exp_orders[j, t] = ye_jt[j, t].x

    for t in T:
        maint = [] 
        for m in M:
            (i, j, r, n) = system.asset[m].idc
            n_max = system.asset[m].nmax
            for k in range(1, n_max + 1):
                if x_mkt[m, k, t].x > 0.95:
                    if x_mkt_rob[m, k, t].x > 0.95:
                        maint.append((m,1))
                    else:
                        maint.append((m,0))
        calendar.append(maint)
    return calendar, orders, runningtime, runningtime1, exp_orders, mod.ObjVal

