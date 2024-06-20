import pandas as pd
from datetime import datetime
import time
from optimization_functions import*
import json
# Opening JSON file
f = open('server_data/data/responseServer.json')
# returns JSON object as a dictionary
update_trigger = json.load(f)['date']
time_scale=pd.period_range(start='2022-07-01', end='2030-01-01', freq="3D")

##  MAIN SIMULATION LOOP:

mode  = [1,1]
cb    = 1
Ns  = 50
N_cpf = 4   # N_cpf: number of components per family
N_upd = 50   # N_upd: number of updates of rolling horizon

# Defining the system
dt_orders = 90
# ["Adsorbent bed ","Pump ","Valve ","Battery ","Actuator ","Fan "]
components=[[1,1,0,1,2],[0,0,1,N_cpf+1,1],[1,1,1,N_cpf,1],[1,1,1,N_cpf,1],[0,0,0,N_cpf+2,2],[1,1,1,N_cpf,1]]

on_hand_inventory= [1,2,2,2,2,2]
system     = System(components,on_hand_inventory)
Ntot = len(system.M)


# SET SIMULATION AND OPTIMIZATION PARAMETERS
par = [1.5,1.5]


optimization_parameters  = {}
optimization_parameters['T_max']  = 120
optimization_parameters['N_s']    = Ns 

# Orders related parameters
optimization_parameters['previous_orders'] = []
optimization_parameters['fixed_order']     = 0
optimization_parameters['next_time_order'] = 0
optimization_parameters['dt_order'] = dt_orders
optimization_parameters['freeze_per_orders'] = 30


optimization_parameters['Y_max']    = Ntot*2*2
optimization_parameters['V_j']      = [1,2,1,2,1,2] 
# Inventory related parameters:
optimization_parameters['on_hand_inventory'] = on_hand_inventory
# Penalty costs paramters:

optimization_parameters['ce'] = np.array([1,1,1,1,1,1])*.5
optimization_parameters['cl'] = optimization_parameters['ce']*par[0]
optimization_parameters['cp'] = np.array([1,1,1,1,1,1])*8
optimization_parameters['cc'] = optimization_parameters['cp'] *par[1]

optimization_parameters['p_exp']           = 10 
optimization_parameters['holding_cost']        = np.array([1,1,1,1,1,1])*0.05
# Chance constraints parameters:
rho1 = np.ones(Ntot)*6
eps1=np.ones(Ntot)*.1 
optimization_parameters['rho']     = [rho1,4]
optimization_parameters['epsilon'] = [eps1,0.1] 
# others 
optimization_parameters['crew_availibility'] = np.ones((2,optimization_parameters['T_max']))*5
optimization_parameters['t_m']               = np.ones((2,Ntot))
m = 0
for r in components:
    nmax = r[4]
    nfam = r[3] 
    for j in range(nfam):
        optimization_parameters['t_m'][0,m] = (2/nmax)
        optimization_parameters['t_m'][1,m] = (2/nmax)*1.2
        m+=1 
optimization_parameters['t_m'][0,0] = 1
optimization_parameters['t_m'][1,0] = 1
for m in [14,15,16,17,18,19]:
        optimization_parameters['t_m'][0,m] = 1.5
        optimization_parameters['t_m'][1,m] = 1.5

optimization_parameters['doable_tasks']      = np.ones((2,Ntot))
M_no_robot= [0,1,2,5,6,7,11,14,15,16,17,18,19,22,23]

for m in M_no_robot:
    optimization_parameters['doable_tasks'][1,m]=0
         


optimization_parameters['S_rob'] = 5
optimization_parameters['S_hum'] = 20

optimization_parameters['repair_cost']      = np.ones((Ntot,2))
optimization_parameters['repair_cost'][:,1]      = optimization_parameters['repair_cost'][:,1]*.7

dt_orders  =optimization_parameters['dt_order']

# set optimizer type
# 1 : mro with 1 chance-constraint for unavailability represented by scenarios
# 2 : mro with 2 chance-constraints for unavailability and corr maintenances represented by scenarios
# 3 : mro with 2 chance-constraints for unavailability (scenarios) and corr maintenances (safe approx)
optimization_parameters['print gurobi optimization results']  = 0

# ===================================================================================================== 
# simulation starts:
T_orders      = [(dt_orders*x) for x in range(100)]
freeze_period = 10
abs_time      = 51
summary       = []
orders_record = []
inventory_records = np.zeros(((N_upd+2)*freeze_period+abs_time,len(system.F)))
orders            = [0,0,0,0,0,0]
exped_orders      = []
states            = []
y_ord             = [0,0,0,0,0,0]
optimization_parameters0 = deepcopy(optimization_parameters)
inventory_records[abs_time]= system.x_inv
cal0 = []
op0  = []
op1  = []
run_times00 =[]
run_times10 =[]
run_times01 =[]
run_times11 =[]
tlim = 50
M=range(24)
F=range(6)
dist =[[] for m in M]
stat_dist =[[] for m in M]

print("\n ============================ \n SIMULATION STARTS \n ============================")

for n_rolling in range(40):  
    #clear_output(wait=True)
    print('Update #:  ', n_rolling, ", Update frequency [days]:", freeze_period*3, ', Date:', time_scale[abs_time])
    print("--------------------------------------------------------------------------")
    # Planning Module:    
    # Solve stochastic MILP
    for s in range(100): 
        if (abs_time + s) in T_orders:
            fixed_order = (s<=optimization_parameters['freeze_per_orders'])
            next_time_order = int(s)
            break
    optimization_parameters['previous_orders']   = orders 
    optimization_parameters['fixed_order']       = fixed_order
    optimization_parameters['next_time_order']   = next_time_order
    optimization_parameters['on_hand_inventory'] = system.x_inv
    # waiting to run....
    if abs_time>200:
        # Opening JSON file
        f = open('server_data/data/responseServer.json')
        # returns JSON object as a dictionary
        update_trigger_new = json.load(f)['date']
        while update_trigger==update_trigger_new:
            time.sleep(2)
            # Opening JSON file
            f = open('server_data/data/responseServer.json')
            # returns JSON object as a dictionary
            update_trigger_new = json.load(f)['date']
        update_trigger = update_trigger_new
    currentdate= abs_time-51 
    calendar,orders,runningtime00,runningtime10,exp_orders,opt_v0      = optimization_module(system,optimization_parameters,tlim,cb,mode) 
    time.sleep(10)
    print("Optimization has finished!")
    
    ## +++++++++++++++++++++++++++++++++++ start
    if abs_time>100:
        F = system.F
        M = system.M 
        des=[" An absorbent bed is a component of the ECLSS CDRA subsystem that removes CO2 from spacecraft air by chemically reacting with it.",
             "A pump is a device that moves fluids (liquids or gases) by mechanical action, typically converted from electrical energy into hydraulic energy.",
             "A valve is a device or natural object that regulates, directs or controls the flow of a fluid", 
             "A battery is a source of electric power consisting of one or more electrochemical cells with external connections for powering electrical devices.",
             "Robot agents performing specific tasks in SmartHab. ",
             "An apparatus with rotating blades that creates a current of air for cooling or ventilation."
            ]
        name = ["Adsorbent bed ","Pump ","Valve ","Battery ","Robot ","Fan "]
        

        ID= ['A401', 'B020', 'C102', 'D201', 'E091', 'F021']
        
        subSystemsNames = ["ECLSS (CO2 Rem.)","Robotics","EPS","Environment"]
        # components=[[0,0,0,1,2],[0,0,1,N_cpf+1,1],[1,0,0,N_cpf+2,3],[1,0,1,N_cpf,2],[1,1,0,N_cpf,2],[1,1,1,N_cpf,1]]
        # [0, [1,2,3,4,5], [6,7,8,9], [10,11,12,13], [14,15,16,17,18,19],[20,21,22,23]]
        subSystemsComp  = [[0,1,2,3,6,7,8], [14,15,16,17,18,19], [11,12,13,20,21],[4,5,9,10,22,23]]
        progModelName   = ["Particle Filter","Brownian Motion",\
                          "Kalman Filter","Kalman Filter",
                          "Brownian Motion","Particle Filter"]

        desc = []
        count= np.zeros(6)

        data_comp = []

        for m in M:
            f =system.asset[m].fam
            (i, j, r, n) = system.asset[m].idc
            if system.asset[m].status == 1:
                age = system.asset[m].age
            else:
                age = 1
            if cb == 1:
                cdf = data_rul_cdf[i, j, r, n][age]
            else:
                cdf = prior_rul_cdf[i, j, r][age]
            cdf[-1] = 1
            pmf = deepcopy(cdf)
            pmf[1:] = deepcopy(cdf[1:]) - deepcopy(cdf[:-1])
            elements = np.arange(1, 401)*3
            sampled_scenarios = np.random.choice(elements, 1000, p=pmf)
            ss = sampled_scenarios*system.asset[m].status
            mu = np.round(np.mean(ss)*system.asset[m].status,2)
            med = np.round(np.median(ss)*system.asset[m].status,2)
            st = np.round(np.std(ss)*system.asset[m].status,2)  
            D = np.zeros((400,3), dtype=object)
            pmf[0]=0
            pmf[-1]=0
            D[:,0]=elements
            D[:,1]=pmf
            D[:,2]= str(time_scale[abs_time-51])

            if len(dist[m])==10:
                dist[m].pop(0)
                dist[m].append(D)
                stat_dist[m].pop(0)
                stat_dist[m].append([mu,med,st,str(time_scale[abs_time-51])])
            else:
                dist[m].append(D)
                stat_dist[m].append([mu,med,st,str(time_scale[abs_time-51])])
            tot_dist = dist[m][0]
            for l in range(1,len(dist[m])):
                tot_dist=np.concatenate((tot_dist,dist[m][l]))
            dist_m = tot_dist.tolist()
            stat_m = stat_dist[m]  
            per = np.percentile(sampled_scenarios, 20, axis=0)
            if per <15:
                state = 3
            elif per <30:
                state = 2
            else:
                state = 1
            '''
            if (mu-2*st) <15:
                state = 3
            elif (mu-2*st) <30:
                state = 2
            else:
                state = 1
            '''
            d=0
            next_repair = '2028-05-25'
            for c in calendar:
                if (m,0) in c or (m,1) in c:
                    next_repair = str(time_scale[abs_time-51+d])
                    break
                d+=1

            Dcom_m = {}
            Dcom_m["id"] =  "omp."+str(m)
            Dcom_m["name"] =  name[f]+" " + str(int(count[f]))
            Dcom_m["next_repair"] = next_repair
            Dcom_m["stat_dist"] = stat_m
            Dcom_m["status"]   =  state
            Dcom_m["family"]   =  name[f]+ " " +ID[f] 
            Dcom_m["alert3"]= "Alert 1 - Potential failure within the next month "
            Dcom_m["alert2"]= "Alert 2 - Potential failure within the next 2 months "
            Dcom_m["algorithm"]= progModelName[system.asset[m].fam]
            mm=0
            for nam in subSystemsComp:
                if m in nam:
                    break
                mm+=1
            Dcom_m["subsystem"]=subSystemsNames[mm]
            Dcom_m["description"] = des[f]
            Dcom_m["dist"] = dist_m
            
            data_comp.append(Dcom_m)
            count[f]+=1
        with open('server_data/data/data.json', 'w') as f:
            json.dump(data_comp, f)

        d = 0
        dataCalendar = []
        for c in calendar:
            for m in M:
                if (m,0) in c or (m,1) in c:
                    next_repair = str(time_scale[abs_time-51+d])
                    rep ={}
                    rep["date"] = next_repair
                    rep["components"] = m+1
                    rep["component_name"] = data_comp[m]['name']
                    rep["type"] = int((m,1) in c)
                    rep["duration"] = np.round(optimization_parameters['t_m'][0,m],2) #np.round(np.random.uniform(.5,2.5),2)
                    dataCalendar.append(rep)
            d+=1
        with open('server_data/data/dataCalendar.json', 'w') as f:
            json.dump(dataCalendar, f)


        inventory = []
        for f in F:
            inv_f = {}
            inv_f["fam_id"] = ID[f]
            inv_f["fam_name"] = name[f]
            inv_f["Q"] = int(system.x_inv[f])
            inv_f["nextOrder"] = int(orders[(f,0)])
            inv_f["dateNextOrder"] = str(time_scale[abs_time-51+next_time_order])  
            inv_f["description"] = des[f]
            inventory.append(inv_f)
            with open('server_data/data/inventory.json', 'w') as f:
                json.dump(inventory, f)
        response = {}
        response["date"] = str(time_scale[currentdate])
        
        with open('server_data/data/responseClient.json', 'w') as f:
            json.dump(response, f)
        # end ================================================
    
    cal0.append(calendar)
    run_times00.append(runningtime00)
    run_times10.append(runningtime10)
    op0.append(opt_v0)


    # Execution module:
    time_freezep = 0
    for j in system.F:
        y_ord[j]+=exp_orders[j,1]
        if exp_orders[j,1]>=1:
            exped_orders.append((abs_time,exp_orders[j,time_freezep+1],j))
    for t in range(freeze_period):
        time_freezep+=1
        abs_time += 1
        #update status:
        status, report = system.update(calendar[t],y_ord) 
        # save relevant data:
        summary.append(report)  
        states.append(status)
        #exped_orders.
        #update orders for the next period
        if abs_time+1 in T_orders:
            y_ord  = [orders[j,0] for j in range(len(system.F))]
            orders_record.append((abs_time,y_ord)) 
        else:
            y_ord         = [0,0,0,0,0,0]
        for j in range(len(system.F)):
            y_ord[j]+=exp_orders[j,time_freezep+1]
            if exp_orders[j,time_freezep+1]>=1:
                exped_orders.append((abs_time,exp_orders[j,time_freezep+1],j))
        inventory_records[abs_time]= system.x_inv





