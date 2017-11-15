#!/usr/bin/env pyomo_python
import sqlite3, sys, getopt, os
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from IPython import embed as IP

# Including all future techs
tech_map = {
    'ENGACC05':     'NGA',
    'ENGACT05':     'NGA',    
    'ENGAACC':      'NGA',
    'ENGAACT':      'NGA',
    'ENGACCCCS':    'NGA',
    'ENGACCR':      'NGA',
    'ENGACTR':      'NGA',
    'ENGASTMR':     'NGA',
    'ECOALSTM':     'COA',    
    'ECOALIGCC':    'COA',
    'ECOALIGCCS':   'COA',
    'ECOALOXYCS':   'COA',
    'ECOASTMR':     'COA',
    'ECOALSTM_b':   'COA',    
    'ECOALIGCC_b':  'COA',
    'ECOALIGCCS_b': 'COA',
    'ECOASTMR_b':   'COA',
    'ECOALSTMCCS':  'COA',
    'EDSLCCR':      'OIL',
    'EDSLCTR':      'OIL',
    'ERFLSTMR':     'OIL',
    'EURNALWR':     'NUC',
    'EURNALWR15':   'NUC',
    'EBIOIGCC':     'BIO',
    'EBIOSTMR':     'BIO',
    'EGEOBCFS':     'GEO',
    'EGEOR':        'GEO',
    'ESOLPVCEN':    'SOL',
    'ESOLSTCEN':    'SOL',
    'ESOLTHR':      'SOL',
    'ESOLPVR':      'SOL',
    'ESOLPVDIS':    'SOL',
    'EWNDR':        'WND',
    'EWNDON':       'WND',     
    'EWNDOFS':      'WND',
    'EWNDOFD':      'WND',
    'EHYDCONR':     'HYD',
    'EHYDREVR':     'PUM', # Pumped hydro
    'EMSWSTMR':     'BIO',
    'ELFGICER':     'BIO',
    'ELFGGTR':      'BIO',
    'EHYDGS':       'GSR',
    'EE':           'EE'
    }

emis_map = {
    'E_FGD_COABH_N':                'SO2 control',
    'E_FGD_COABH_R':                'SO2 control',
    'E_FGD_COABM_N':                'SO2 control',
    'E_FGD_COABM_R':                'SO2 control',
    'E_FGD_COABL_N':                'SO2 control',
    'E_FGD_COABL_R':                'SO2 control',
    'E_LNBSNCR_COAB_R':             'NOx control',
    'E_LNBSNCR_COAB_N':             'NOx control',
    'E_LNBSCR_COAB_R':              'NOx control',
    'E_LNBSCR_COAB_N':              'NOx control',
    'E_LNB_COAB_R':                 'NOx control',
    'E_LNB_COAB_N':                 'NOx control',
    'E_SCR_COAB_R':                 'NOx control',
    'E_SCR_COAB_N':                 'NOx control',
    'E_SNCR_COAB_R':                'NOx control',
    'E_SNCR_COAB_N':                'NOx control',
    'E_CCR_COAB':                   'CO2 control',
    'E_CCR_COALIGCC_N':             'CO2 control',
    'E_CCR_COALSTM_N':              'CO2 control',
    'E_CCR_NGAACC_N':               'CO2 control',
    }

# http://www.rapidtables.com/web/color/RGB_Color.htm
color_map = {
    'NGA':         [0.7, 0.7, 0.7],
    'COA':         [0.0, 0.0, 0.0],
    'OIL':         [1.0, 0.0, 0.0],
    'NUC':         [0.6, 0.0, 0.8],
    'BIO':         [0.0, 1.0, 0.0],
    'GEO':         [1.0, 0.5, 0.3],
    'SOL':         [1.0, 1.0, 0.0],
    'WND':         [0.0, 0.0, 1.0],
    'HYD':         [0.4, 0.6, 0.9],
    'PUM':         [0.4, 0.6, 0.9],
    'GSR':         [1.0, 0.0, 0.0],
    'CO2 control': 'black',
    'NOx control': [0.5, 0.0, 0.0],
    'SO2 control': 'green',
    'EE':          'white',
    'other':       [1.0, 1.0, 1.0]
    }

edge_map = {
    'NGA':         [0.7, 0.7, 0.7],
    'COA':         [0.0, 0.0, 0.0],
    'OIL':         [1.0, 0.0, 0.0],
    'NUC':         [0.6, 0.0, 0.8],
    'BIO':         [0.0, 1.0, 0.0],
    'GEO':         [1.0, 0.5, 0.3],
    'SOL':         [1.0, 1.0, 0.0],
    'WND':         [0.0, 0.0, 1.0],
    'HYD':         [0.4, 0.6, 0.9],
    'PUM':         [0.4, 0.6, 0.9],
    'GSR':         [1.0, 0.0, 0.0],
    'CO2 control': 'black',
    'NOx control': [0.5, 0.0, 0.0],
    'SO2 control': 'green',
    'EE':          'black',
    'other':       [1.0, 1.0, 1.0]
    }

hatch_map = {
    'NGA':   None,
    'COA':   None,
    'OIL':   None,
    'NUC':   None,
    'BIO':   None,
    'GEO':   None,
    'SOL':   None,
    'WND':   None,
    'HYD':   None,
    'PUM':   '++',
    'GSR':   None,
    'EE':    '//',
    'other': '++'
    }

class TemoaNCResult():
    def __init__(self, db, scenario):
        con = sqlite3.connect(db)
        cur = con.cursor()

        qry = "SELECT * FROM Output_Objective"
        qry += " WHERE scenario='" + scenario + "'"
        cur.execute(qry)
        self.Output_Objective = cur.fetchall()

        qry = "SELECT * FROM Output_CapacityByPeriodAndTech"
        qry += " WHERE scenario='" + scenario + "'"
        cur.execute(qry)
        self.Output_CapacityByPeriodAndTech = cur.fetchall()
        
        qry = "SELECT * FROM Output_VFlow_Out"
        qry += " WHERE scenario='" + scenario + "'"
        cur.execute(qry)
        self.Output_Activity = cur.fetchall()
        
        qry = "SELECT * FROM Output_Emissions"
        qry += " WHERE scenario='" + scenario + "'"
        cur.execute(qry)
        self.Output_Emissions = cur.fetchall()
        
        qry = "SELECT * FROM time_season"
        cur.execute(qry)
        self.seasons = cur.fetchall()
        self.seasons = [str(i[0]) for i in self.seasons]
        
        qry = "SELECT * FROM time_of_day"
        cur.execute(qry)
        self.tods = cur.fetchall()
        self.tods = [str(i[0]) for i in self.tods]
        
        qry = "SELECT * FROM Demand"
        cur.execute(qry)
        self.Demands = cur.fetchall()
        self.Demands = [i[2] for i in self.Demands]

        qry = "SELECT * FROM DemandSpecificDistribution"
        cur.execute(qry)
        self.DSD = cur.fetchall()
        self.DSD = [i[3] for i in self.DSD]

        qry = "SELECT * FROM SegFrac"
        cur.execute(qry)
        self.SegFrac = cur.fetchall()
        self.SegFrac = [i[2] for i in self.SegFrac]

        qry = "SELECT * FROM Efficiency"

        con.close()

        self.TotalCost   = self.Output_Objective[0][2] # We know there is only one objective function

        self.periods     = set()
        self.techs       = set() # This set include the broader tech catagory: NGA, BIO, etc.
        self.e_ctrls     = set() # Emission being controlled, CO2, SO2, etc.
        
        for row in self.Output_CapacityByPeriodAndTech:
            scenario, sector, p, t, value = row
            self.periods.add(p)
            # periods.append(p)
            if t in tech_map:
                self.techs.add(tech_map[t])
            if t in emis_map:
                self.e_ctrls.add(emis_map[t])
        
        self.periods = list(self.periods)
        self.periods.sort()
        self.techs = list(self.techs)
        if 'EE' in self.techs:
            self.techs.remove('EE')
            self.techs.append('EE') # EE always on top of all color

        self.chemicals = set()

        for row in self.Output_Emissions:
            # Scenario, sector, period, emission, tech, vintage, value
            scenario, sector, p, e, t, v, value = row
            self.chemicals.add(e)

        # chemicals = list(chemicals)

        self.capacities = dict()
        self.cap_e_ctrl = dict()
        self.activities = dict()
        self.act_e_ctrl = dict()
        self.emissions  = dict()
        self.emissions1 = dict() # emissions1 is emission by source
        self.emis_redct = dict()

        self.p_t = dict() # power output in each time slice by technology

        for tech in self.techs:
            self.capacities[tech] = [0]*len(self.periods)
            self.activities[tech] = [0]*len(self.periods)

        # for t in tech_map.keys():
        #     p_t[t] = dict()
        #     for p in periods:
        #         p_t[t][str(p)] = [0]*len(seasons)*len(tods)
        for t in tech_map.keys():
            self.p_t[t] = list()
            self.p_t[tech_map[t]] = list()
            for p in self.periods:
                self.p_t[t].append( [0]*len(self.seasons)*len(self.tods) )
                self.p_t[tech_map[t]].append( [0]*len(self.seasons)*len(self.tods) )

        for tech in self.e_ctrls:
            self.cap_e_ctrl[tech] = [0]*len(self.periods)
            self.act_e_ctrl[tech] = [0]*len(self.periods)

        for row in self.Output_CapacityByPeriodAndTech:
            scenario, sector, p, t, value = row
            if t in tech_map:
                self.capacities[tech_map[t]][self.periods.index(p)] += value
            if t in emis_map:
                self.cap_e_ctrl[emis_map[t]][self.periods.index(p)] += value

        for row in self.Output_Activity:
            scenario, sector, p, s, d, i, t, v, o, value = row
            # Scenario, sector, period, season, tod, input, tech, vintage, output, value
            if (t in tech_map and o in ['ELC', 'ELCRNWB', 'ELCSOL']) or (t == 'EE'):
                # o == 'ELC': Common EGU; o == 'ELCRNWB': renewable EGU;
                pindex = self.periods.index(p) # period index
                hindex = len(self.tods)*self.seasons.index(s) + self.tods.index(d) # 'hour' index
                self.activities[tech_map[t]][pindex] += value
                self.p_t[t][pindex][hindex] += value
                if t != 'EE':
                    self.p_t[tech_map[t]][pindex][hindex] += value

            if t in emis_map:
                self.act_e_ctrl[emis_map[t]][self.periods.index(p)] += value

        for chemical in self.chemicals:
            temp = {
                    'NGA':   [0]*len(self.periods),
                    'COA':   [0]*len(self.periods),
                    'BIO':   [0]*len(self.periods),
                    'OIL':   [0]*len(self.periods),
                    'other': [0]*len(self.periods)
            }
            self.emissions1[chemical] = temp
            self.emissions[chemical]  = [0]*len(self.periods)
            self.emis_redct[chemical] = [0]*len(self.periods)
        
        for row in self.Output_Emissions:
            # Scenario, sector, period, emission, tech, vintage, value
            scenario, sector, p, e, t, v, value = row # To increase readability
            self.emissions[e][self.periods.index(p)] += value
            if value < 0:
                self.emis_redct[e][self.periods.index(p)] += -value

            if value < 0:
                catagory = 'COA'
            elif t == 'E_EA_COAB':
                catagory = 'COA'
            elif t in tech_map:
                catagory = tech_map[t]
            else:
                catagory = 'other'

            self.emissions1[e][catagory][self.periods.index(p)] += value

def plot_stochastic_var(options):
    techs       = options.techs
    directory   = options.directory
    run         = options.run
    scenarios   = options.scenarios
    linestyle_s = options.linestyle_s
    color_r     = options.color_r
    alpha       = options.alpha
    db_name     = options.db_name
    filenames   = [os.path.sep.join( [directory, i, db_name] ) for i in run]

    capacities_rs = dict() # Capacities in GW, all runs, all scenarios
    activities_rs = dict() # Activities in PJ, all runs, all scenarios
    for r in run:
        f = filenames[ run.index(r) ]
        capacities_s = dict()
        activities_s = dict()
        for s in scenarios:
            instance   = TemoaNCResult(f, s)
            periods    = instance.periods
            capacities = instance.capacities
            activities = instance.activities
            capacities_s[s] = capacities
            activities_s[s] = activities
        capacities_rs[r] = capacities_s
        activities_rs[r] = activities_s

    for t in techs:
        plt.figure( techs.index(t) )
        handles = list()
        for i in range(0, len(filenames)):
            c = color_r[i]
            f = filenames[i]
            a_s = activities_rs[run[i]]
            for j in range(0, len(scenarios)):
                s = scenarios[j]
                l = linestyle_s[j]
                plt.plot(
                    periods,
                    a_s[s][t],
                    color=c,
                    linestyle=l,
                    # alpha=alpha,
                    linewidth=2,
                )
            h = plt.fill_between(
                periods, 
                a_s[scenarios[0]][t], 
                a_s[scenarios[-1]][t], 
                color=c, 
                alpha=alpha,
            )
            handles.append(h)
        plt.legend(handles, run, loc='upper left')
        plt.title(t)
    plt.show()

def plot_stochastic_obj(l_scale, directory, run, scenarios, db_name):
    x_cross = list()
    y_cross = list()
    plt.figure(0)
    for scale in l_scale:
        filenames = [os.path.sep.join( [directory, str(scale), i, db_name] ) for i in run]
        obj_r = dict() # Stochastic objective value for each run
        for r in run:
            f = filenames[ run.index(r) ]
            obj = list()
            for s in scenarios:
                instance   = TemoaNCResult(f, s)
                obj.append(instance.TotalCost)
            obj_r[r] = sum(obj)/3.0 # We know the probability for each scenario is 1/3
        prob = [0, 1]
        plt.plot(prob, [ obj_r[ run[2] ], obj_r[ run[0] ] ], '-bs')
        plt.plot(prob, [ obj_r[ run[3] ], obj_r[ run[1] ] ], '-rs')

        x = -( 
            obj_r[ run[2] ] -
            obj_r[ run[3] ]
        )/( 
            ( 
                obj_r[ run[0] ] -
                obj_r[ run[2] ]
            ) 
            - (
                obj_r[ run[1] ] -
                obj_r[ run[3] ]
            )
        )
        y = obj_r[ run[2] ] + x*(obj_r[ run[0] ] - obj_r[ run[2] ])
        x_cross.append(x)
        y_cross.append(y)
    plt.plot(x_cross, y_cross, 'k*')

    plt.figure(1)
    plt.plot(l_scale, x_cross, 'k*')
    plt.show()

def plot_result(fname, s_chosen):

    instance = TemoaNCResult(fname, s_chosen)
    Demands    = instance.Demands
    DSD        = instance.DSD
    periods    = instance.periods
    techs      = instance.techs
    e_ctrls    = instance.e_ctrls
    seasons    = instance.seasons
    tods       = instance.tods
    SegFrac    = instance.SegFrac
    chemicals  = instance.chemicals
    capacities = instance.capacities
    cap_e_ctrl = instance.cap_e_ctrl
    activities = instance.activities
    act_e_ctrl = instance.act_e_ctrl
    emissions  = instance.emissions
    emissions1 = instance.emissions1
    emis_redct = instance.emis_redct
    p_t        = instance.p_t
    
    print '\nNOx'    
    print 'COA_red', 'NOx_elc', [ -i for i in emis_redct['nox_ELC'] ]
    for s in ['COA', 'NGA', 'other', 'BIO', 'OIL']:
        print s, 'NOx_elc', emissions1['nox_ELC'][s]

    print '\nSO2'
    print 'COA_red', 'SO2_elc', [ -i for i in emis_redct['so2_ELC'] ]
    for s in ['COA', 'NGA', 'other', 'BIO', 'OIL']:
        print s, 'SO2_elc', emissions1['so2_ELC'][s]

    print '\nCO2'
    print 'COA_red', 'CO2', [ -i for i in emis_redct['co2_ELC'] ]
    for s in ['COA', 'NGA', 'other', 'BIO', 'OIL']:
        print s, 'CO2', emissions1['co2_ELC'][s]

    width = 2
    findex = 0
    # Start plotting cpacities: electricity generating capacity and 
    # emission control technologies
    plt.figure(findex)
    findex += 1
    handles = list()
    b = [0]*len(periods)
    #ax = plt.subplot(121)
    ax = plt.subplot(111)
    for tech in techs:
        if hatch_map[tech]:
            h = ax.bar(periods, 
                        capacities[tech], 
                        width, 
                        bottom = b,
                        color = color_map[tech], 
                        hatch = hatch_map[tech]
                        )
        else:
            h = ax.bar(periods, 
                        capacities[tech], 
                        width, 
                        bottom = b,
                        color = color_map[tech], 
                        edgecolor = edge_map[tech]
                        )
        handles.append(h)
        b = [b[i] + capacities[tech][i] for i in range(0, len(b))]
    plt.ylabel('Capacity (GW)')
    plt.xticks([i + width*0.5 for i in periods], [str(i) for i in periods])
    plt.title('Electricity Generating Capacity')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.9, box.height])
    ax.legend([h[0] for h in handles], 
                techs, 
                bbox_to_anchor = (1.01, 0.5), 
                loc='center left')
    ax.yaxis.grid(True)

    #ax = plt.subplot(122)
    plt.figure(findex)
    findex += 1
    ax = plt.subplot(111)
    handles = list()
    l = periods
    for tech in e_ctrls:
        h = ax.bar(l, 
                   cap_e_ctrl[tech], 
                   width*.5, 
                   color = color_map[tech], 
                   edgecolor = edge_map[tech]
                   )
        handles.append(h)
        l = [l[i] + width*.5 for i in range(0, len(l))]
    plt.ylabel('Capacity (PJ)')
    plt.xticks([i + width*0.25*len(cap_e_ctrl) for i in periods], 
                [str(i) for i in periods]
                )
    plt.title('Capacity of technologies defined as emission control')
    plt.xlim((min(periods), max(periods) + len(cap_e_ctrl)*width))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.9, box.height])
    ax.legend([h[0] for h in handles], e_ctrls, 
                bbox_to_anchor = (1.01, 0.5), 
                loc='center left'
                )
    ax.yaxis.grid(True)

    # Second figure: activites, inclulding electricity generating and emission 
    # control.
    plt.figure(findex)
    findex += 1
    handles = list()
    b = [0]*len(periods)
    #ax = plt.subplot(121)
    ax = plt.subplot(111)
    for tech in techs:
        if hatch_map[tech]:
            h = ax.bar(periods, 
                        [i/3.6 for i in activities[tech]],  # PJ -> TWh
                        width, 
                        bottom = b,
                        color = color_map[tech], 
                        hatch = hatch_map[tech]
                        )
        else:
            h = ax.bar(periods, 
                        [i/3.6 for i in activities[tech]],  # PJ -> TWh
                        width, 
                        bottom = b,
                        color = color_map[tech], 
                        edgecolor = edge_map[tech]
                        )
        handles.append(h)
        b = [b[i] + activities[tech][i]/3.6 for i in range(0, len(b))]
    plt.ylabel('Generation (TWh)')
    plt.xticks([i + width*0.5 for i in periods], [str(i) for i in periods])
    plt.title('Electricity Generation')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.9, box.height])
    ax.legend([h[0] for h in handles], techs, 
                bbox_to_anchor = (1.01, 0.5), 
                loc='center left'
                )
    ax.yaxis.grid(True)

    #ax = plt.subplot(122)
    plt.figure(findex)
    findex += 1
    ax = plt.subplot(111)
    handles = list()
    l = periods
    for tech in e_ctrls:
        h = ax.bar(l, 
                   act_e_ctrl[tech], 
                   width*.5, 
                   color = color_map[tech], 
                   edgecolor = edge_map[tech]
                   )
        handles.append(h)
        l = [l[i] + width*.5 for i in range(0, len(l))]
    plt.ylabel('Activity (PJ)')
    plt.xticks([i + width*0.25*len(act_e_ctrl) for i in periods], 
                [str(i) for i in periods]
                )
    plt.title('Activity of technologies defined as emission control')
    plt.xlim((min(periods), max(periods) + len(act_e_ctrl)*width))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.9, box.height])
    ax.legend([h[0] for h in handles], e_ctrls, 
                bbox_to_anchor = (1.01, 0.5), 
                loc='center left'
                )
    ax.yaxis.grid(True)

    # Figure 3: emission activities
    plt.figure(findex)
    findex += 1
    i = 1
    for chemical in chemicals:
        ax = plt.subplot(2, 3, i)
        handles = list()

        # h = ax.bar(periods,
        #            emissions[chemical],
        #            width,
        #            color = 'k'
        #            )
        # handles.append(h)

        # h = ax.bar(periods,
        #            emis_redct[chemical],
        #            bottom = emissions[chemical],
        #            width = width,
        #            color = 'white'
        #            )
        # handles.append(h)

        order = ['OIL', 'BIO', 'other', 'NGA', 'COA']
        bottom = np.array([0]*len(periods))
        for source in order:
            h = ax.bar(periods,
                       emissions1[chemical][source],
                       width,
                       bottom = bottom,
                       color = color_map[source],
                       hatch = hatch_map[source]
                       )
            handles.append(h)
            bottom = bottom + np.array(emissions1[chemical][source])

        h = ax.bar(periods,
                   emis_redct[chemical],
                   bottom = bottom,
                   width = width,
                   color = 'white'
                   )
        handles.append(h)

        ax.yaxis.grid(True)
        plt.xticks([j + width/2. for j in periods], 
                    [str(j) for j in periods], 
                    rotation='vertical'
                    )
        plt.title(chemical)
        plt.ylabel('emissions (kton)')
        i += 1
    ax.legend([h[0] for h in handles], 
                ['OIL', 'BIO', 'other', 'NGA', 'COA', 'Emission Reductios'], 
                bbox_to_anchor = (1.1, 0.5), 
                loc='center left'
                )
    # this_figure.legend([h[0] for h in handles], 
    #                     ['Emissions', 'Emission Reductios'], 'lower right')

    for p in periods:
        plt.figure(findex)
        findex += 1
        ys = list()
        catagories = list( set(tech_map.values()) )
        catagories.remove('NUC')
        catagories.remove('COA')
        catagories.remove('NGA')
        catagories.insert(0, 'NGA')
        catagories.insert(0, 'COA')
        catagories.insert(0, 'NUC')
        catagories.remove('EE')
        catagories.append('EE')
    
        # Remove those technologies with output = 0
        catagories_on_fig = list()
        for t in catagories:
            if abs( sum(p_t[t][periods.index(p)]) ) > 1E-3:
                catagories_on_fig.append(t)
    
        for t in catagories_on_fig:
            # Convert electricity production in each slice in PJ to the average
            # power output in the same slice in GW
            y = np.array(p_t[t][periods.index(p)])/3.6/(8760*np.array(SegFrac))*1000
            if t != 'EE':
                y *= 0.97
            ys.append(y)
        ys = np.array(ys) # Convert unit to GWh
        slices = range(0, len(ys[0]))
        ystack = np.cumsum(ys, axis=0)
    
        area_handles = list()
        line_handles = list()
        ax = plt.subplot(111)
        for i in range(0, len(catagories_on_fig)):
            t = catagories_on_fig[i]
            if i==0:
                h = ax.fill_between(slices, 
                                    0, ystack[i, ], 
                                    facecolor=color_map[t],
                                    # edgecolor=color_map[t],
                                    # alpha=.7,
                                    hatch=hatch_map[t])
                area_handles.append(h)
                # Plot a second time, since the color hatch line and edge have to be 
                # the same.
                ax.fill_between(slices, 
                                0, ystack[i, ], 
                                facecolor='none',
                                edgecolor=color_map[t])
            else:
                h = ax.fill_between(slices, 
                                    ystack[i-1, ], ystack[i, ], 
                                    facecolor=color_map[t],
                                    # edgecolor=color_map[t],
                                    # alpha=.7,
                                    hatch=hatch_map[t])
                area_handles.append(h)
                ax.fill_between(slices, 
                                ystack[i-1, ], ystack[i, ], 
                                facecolor='none',
                                edgecolor=color_map[t])
        demand_real = Demands[periods.index(p)]*np.array(DSD)/3.6*1000/(
                        8760*np.array(SegFrac) )
        h = ax.plot(slices, demand_real, 'r-', label='Demand')
        line_handles.append(h)
        
        sigma_cap = np.array([0]*len(periods))
        for c in capacities:
            sigma_cap = sigma_cap + np.array(capacities[c])
        h = ax.plot(slices, [sigma_cap[periods.index(p)]]*len(slices), 
                    'k-', label='Installed cap')
        line_handles.append(h)
    
        plt.xlim([ slices[0], slices[-1] ])
        plt.axvline(x=23, color=[.5, 0, 0], linestyle='--')
        plt.axvline(x=47, color=[.5, 0, 0], linestyle='--')
        plt.axvline(x=71, color=[.5, 0, 0], linestyle='--')
        ax.yaxis.grid(True)
        major_ticks = range(0, 95, 20)
        minor_ticks = range(0, 95, 5)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.xaxis.grid(which='minor', alpha=0.8) 
    
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.9, box.height])
        leg1 = ax.legend(area_handles, 
                        catagories_on_fig, 
                        bbox_to_anchor = (1.01, 0.5), 
                        loc='center left')
        plt.gca().add_artist(leg1)
        ax.legend(line_handles, loc='upper right')
        ax.legend()
    
        plt.xlabel('Slice index #')
        plt.ylabel('Average power (GW)')
        plt.title(str(p) + ' load time sequence')

    plt.show()

def plot_NCdemand_all():
    # Area plots of the history demand and future demands in each scenarios

    year_his = range(1990, 2015, 5)

    # Activities in MWh
    ACT_year  = [2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007, 2006, 2005, 2004, 2003, 2002, 2001, 2000, 1999, 1998, 1997, 1996, 1995, 1994, 1993, 1992, 1991, 1990]
    ACT_COA   = [39922168, 49238197, 47072210, 50932180, 59758357, 71951214, 65082782, 75814787, 79983038, 75487005, 78435700, 75546115, 74776231, 75187741, 72957769, 75945463, 72255507, 72949464, 74072041, 69331347, 61484998, 58124391, 63572654, 58274018, 50767188, 49821595]
    ACT_HYD   = [4742004, 4756083, 6900533, 3727938, 3893396, 4756549, 5171257, 3033642, 2984159, 3839012, 5396502, 5435199, 7200943, 3492048, 2595708, 3137815, 3684186, 5738207, 5625884, 5952176, 5520810, 7192276, 4987071, 5768453, 5850120, 6819427]
    ACT_NGA   = [36544596, 28737608, 27982509, 19302008, 11155211, 8447237, 4851885, 4177342, 4456643, 3195563, 3159377, 2558654, 1580366, 3560871, 1534355, 1130253, 1166663, 1284357, 699937, 470138, 771688, 310242, 394751, 361308, 385907, 203466]
    ACT_NUC   = [42096761, 40967020, 40241737, 39385592, 40526834, 40739529, 40847711, 39776280, 40044705, 39963184, 39981739, 40090623, 40906900, 39626849, 37775025, 39126881, 37523504, 38778211, 32453074, 33718182, 35910195, 32346007, 23758927, 22753813, 30312425, 25905319]
    ACT_other = [685283, 631153, 566884, 451865, 492937, 407440, 220428, 315642, 341852, 319235, 302737, 277403, 239177, 196202, 224973, 226443, 252784, 249356, 236051, 249410, 259130, 260777, 234982, 225149, 226926, 203406]
    ACT_BIO   = [544024, 519931, 410294, 302342, 375213, 195310, 131491, 120482, 86845, 91740, 98442, 112166, 114053, 103451, 108959, 109376, 92654, 88423, 78570, 55908, 54779, 45025, 42505, 30506, 44128, 26270]
    ACT_gas   = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 863, 990, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # Other gas
    ACT_OIL   = [434587, 459687, 217571, 178261, 217636, 293373, 296859, 320221, 495689, 451138, 518869, 610909, 783712, 592026, 655745, 825831, 662070, 691313, 496510, 547830, 479619, 452234, 407255, 335046, 275785, 328025]
    ACT_PUM   = [94, 78009, 0, 0, 0, 0, 43077, -121064, 136996, 131342, 146505, 78251, 119273, 25175, 0, 108102, 175397, 65672, 253805, 340406, 167317, 380819, 246302, 109885, 216432, 180046]
    ACT_SOL   = [1373579, 729130, 344663, 139491, 17381, 11340, 4563, 1801, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ACT_wood  = [2045348, 2026770, 2199893, 2262087, 1952891, 1876492, 1757350, 1799930, 1585374, 1736565, 1708706, 1620636, 1861663, 1682804, 1642330, 1664192, 1544851, 1533822, 1578818, 1532342, 1646120, 1716892, 1669998, 1677076, 1296373, 1438215]

    ACT_BIO = [ACT_BIO[i] + ACT_wood[i] for i in range(0, len(ACT_BIO))]
    
    ys    = np.array([ACT_BIO, ACT_OIL, ACT_HYD, ACT_NGA, 
                      ACT_PUM, ACT_SOL, ACT_COA, ACT_NUC]
                      )*1E-6 
    # ys    = np.array([ACT_BIO, ACT_HYD, ACT_NGA, 
    #                   ACT_SOL, ACT_COA, ACT_NUC]
    #                   )*1E-6 # TWh 
    tech  = ['BIO', 'OIL', 'HYD', 'NGA', 'PUM', 'SOL', 'COA', 'NUC']
    # tech  = ['BIO', 'HYD', 'NGA', 'SOL', 'COA', 'NUC']
    tech.reverse()
    b     = [0]*len(year_his)
    handles = list()

    ystack = np.cumsum(ys, axis=0)

    fig = plt.figure(0)
    ax = plt.subplot(111)
    for i in range(0, len(ystack[: ,1])):
        if i==0:
            t = tech.pop()
            h = ax.fill_between(ACT_year, 
                            0, ystack[i, ], 
                            facecolor=color_map[t],
                            edgecolor=None,
                            # alpha=.7,
                            hatch=hatch_map[t])
            handles.append(h)
        else:
            t = tech.pop()
            h = ax.fill_between(ACT_year, 
                                ystack[i-1, ], ystack[i, ], 
                                facecolor=color_map[t],
                                edgecolor=None,
                                # alpha=.7,
                                hatch=hatch_map[t])
            handles.append(h)

    # Future demand, from reference.sql
    DMD_year = range(2015, 2055, 5)
    # DMD_R    = np.array([453, 474, 495, 517, 540, 564, 590, 616])/3.6 # TWh
    # DMD_LF   = np.array([453, 480, 508, 539, 570, 604, 640, 677])/3.6 # TWh
    # DMD_HF   = np.array([453, 475, 497, 521, 546, 572, 599, 627])/3.6 # TWh
    
    # DMD_R      = np.array([453, 456, 481, 500, 525, 551, 589, 618])/3.6 # TWh
    # DMD_LF     = np.array([453, 451, 472, 523, 580, 568, 601, 632])/3.6 # TWh
    # DMD_HF     = np.array([453, 445, 479, 499, 523, 549, 577, 609])/3.6 # TWh
    # DMD_CPP    = np.array([453, 452, 507, 507, 526, 555, 588, 605])/3.6 # TWh
    # DMD_CPPLF  = np.array([453, 447, 497, 530, 582, 572, 600, 619])/3.6 # TWh
    # DMD_CPPHF  = np.array([453, 441, 505, 506, 525, 553, 576, 596])/3.6 # TWh

    DMD        = np.array([462, 491, 521, 553, 587, 621, 661, 702])/3.6 # TWh

    # ax.fill_between(DMD_year, 0, DMD_R,  facecolor='grey', alpha=0.7)
    # ax.fill_between(DMD_year, 0, DMD_LF, facecolor='grey', alpha=0.7)
    # ax.fill_between(DMD_year, 0, DMD_HF, facecolor='grey', alpha=0.7)

    # h = ax.plot(DMD_year, DMD_R,  '-k')
    # h = ax.plot(DMD_year, DMD_LF, '--k')
    # h = ax.plot(DMD_year, DMD_HF, '-.k')
    # h = ax.plot(DMD_year, DMD_CPP,  '-g')
    # h = ax.plot(DMD_year, DMD_CPPLF, '--g')
    # h = ax.plot(DMD_year, DMD_CPPHF, '-.g')

    h = ax.plot(DMD_year, DMD, '-k')


    plt.ylabel('Electricity Generation (TWh)')
    plt.xlabel('Year')
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.9, box.height])
    ax.legend(handles, 
                ['BIO', 'OIL', 'HYD', 'NGA', 'PUM', 'SOL', 'COA', 'NUC'], 
                # ['BIO', 'HYD', 'NGA', 'SOL', 'COA', 'NUC'], 
                bbox_to_anchor = (1.01, 0.5), 
                loc='center left')
    ax.annotate('2015', xy=(2015, 200), xytext=(2013, 201), color='r')
    # ax.annotate('Reference', 
    #             xy=(2050, 616/3.6), xytext=(2053, 610/3.6), 
    #             color='k',
    #             arrowprops=dict(facecolor='black', 
    #                             width=.5, 
    #                             headwidth=3, 
    #                             headlength=5
    #                             )
    #             )
    # ax.annotate('LF', 
    #             xy=(2050, 677/3.6), xytext=(2053, 670/3.6), 
    #             color='k',
    #             arrowprops=dict(facecolor='black', 
    #                             width=.5, 
    #                             headwidth=3, 
    #                             headlength=5
    #                             )
    #             )
    # ax.annotate('HF', 
    #             xy=(2050, 627/3.6), xytext=(2053, 640/3.6), 
    #             color='k',
    #             arrowprops=dict(facecolor='black', 
    #                             width=.5, 
    #                             headwidth=3, 
    #                             headlength=5
    #                             )
    #             )

    ax.axvline(x=2015, color='r')
    plt.show()

def plot_emis_all(*arg):
    # Area plots of the history emissions and future emission caps for NOx, CO2 
    # and SO2.
    # Data from Electric Powre Monthly.
    
    # Future emission caps
    LIM_year = range(2015, 2055, 5)
    # LIM_NOx  = np.array([52.3, 29.9, 33.4, 34.7, 34.7, 
    #                      34.7, 34.7, 34.7, 34.7]) # kt
    # LIM_SO2  = np.array([135.2, 30.1, 33.5, 34.9, 34.9, 
    #                      34.9, 34.9, 34.9, 34.9]) # kt
    # LIM_CO2  = np.array([0, 0, 0, 51843, 46508, 
    #                      46508, 46508, 46508, 46508]) # kt

    LIM_NOx  = np.array([51.0, 49.0, 41.5, 41.5, 41.5, 
                         41.5, 41.5, 41.5]) # kt
    LIM_SO2  = np.array([52.0, 68.0, 57.6, 57.6, 57.6, 
                         57.6, 57.6, 57.6]) # kt
    LIM_CO2  = np.array(
        [
        [0, 0, 51843, 46508, 46508, 46508, 46508, 46508],
        [0, 0, 47114, 40404, 33694, 26984, 20274, 13564],
        ]
    ) # kt

    # NOx emissions in metric tons
    ############################################################################
    NOx_year  = [1990, 1991, 1992, 1993, 1994, 
                 1995, 1996, 1997, 1998, 1999, 
                 2000, 2001, 2002, 2003, 2004, 
                 2005, 2006, 2007, 2008, 2009, 
                 2010, 2011, 2012, 2013, 2014]

    NOx_COA   = [197050, 200392, 226735, 246162, 220491, 
                 253394, 270469, 270411, 238762, 195076, 
                 159338, 143831, 144080, 128733, 112585, 
                 102783, 96728,  56664,  54641,  38686, 
                 48840,  40003,  42225,  39754,  34518]

    NOx_NGA   = [517,  650,  708,  716,  548, 
                 1388, 914,  1292, 2045, 1934, 
                 1634, 1171, 1550, 571,  624, 
                 957,  924,  1477, 794,  1142, 
                 1752, 1681, 3135, 8672, 16066]

    NOx_other = [3918, 2481, 3665, 3592, 3566, 
                 422,  358,  360,  351,  350, 
                 296,  263,  534,  338,  356, 
                 370,  401,  366,  279,  237, 
                 214,  287,  351,  496,  664]

    NOx_BIO   = [5082, 4916, 5173, 5006, 4836, 
                 5130, 4557, 3354, 3803, 1494, 
                 1568, 1372, 1496, 3198, 3632, 
                 5972, 6962, 7179, 7341, 8426]

    NOx_OIL   = [1235, 671,  1202, 1493, 1467, 
                 2589, 3303, 3236, 3757, 3725, 
                 3926, 2673, 2230, 3027, 886, 
                 1014, 618,  639,  675,  550, 
                 629,  1082, 302,  351,  1586]
    NOx_BIO = [0]*(len(NOx_year) - len(NOx_BIO)) + NOx_BIO
    
    ys    = np.array([NOx_COA, 
                      NOx_NGA, 
                      NOx_other, 
                      NOx_BIO, 
                      NOx_OIL])/1.E3
    tech  = ['COA', 'NGA', 'other', 'BIO', 'OIL']
    tech.reverse()
    handles = list()
    ystack = np.cumsum(ys, axis=0)

    fig = plt.figure(0)
    ax = plt.subplot(111)

    for i in range(0, len(ystack[: ,1])):
        if i==0:
            t = tech.pop()
            h = ax.fill_between(NOx_year, 
                            0, ystack[i, ], 
                            facecolor=color_map[t],
                            edgecolor=None,
                            # alpha=.7,
                            hatch=hatch_map[t])
            handles.append(h)
        else:
            t = tech.pop()
            h = ax.fill_between(NOx_year, 
                                ystack[i-1, ], ystack[i, ], 
                                facecolor=color_map[t],
                                edgecolor=None,
                                # alpha=.7,
                                hatch=hatch_map[t])
            handles.append(h)
    
    h = ax.plot(LIM_year, LIM_NOx,  '-rs')
    h = mlines.Line2D([], [], color='red', marker='s')
    # h = ax.fill_between(LIM_year, 
    #                     0, LIM_NOx, 
    #                     facecolor='r', 
    #                     edgecolor=None,
    #                     alpha=.3,
    #                     hatch=None)
    handles.append(h)

    plt.ylabel('Emissions (kilotonnes)')
    plt.xlabel('Year')
    plt.title(r'(b) $\mathregular{NO}_\mathregular{X}$ Emissions')
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width*0.9, box.height])
    ax.legend(handles, 
                ['Coal', 'Natural gas', 'Other', 'Bioenergy', 'Oil', 'Limit'], 
                # bbox_to_anchor = (1.01, 0.5), 
                loc='upper right')
    ax.annotate('2015', xy=(2015, 300), xytext=(2014, 250), 
                color='r', 
                rotation=90)
    ax.axvline(x=2015, color='r')

    # SO2 emissions in metric tons
    ############################################################################
    SO2_year  = [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014]
    SO2_COA   = [343263, 340177, 385118, 428869, 377139, 399139, 473738, 505103, 483523, 451434, 450907, 429693, 437194, 436618, 442897, 468976, 438135, 356472, 223498, 112227, 115861, 73883, 57685, 46983, 39859]
    SO2_NGA   = [1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 2, 4, 7, 7, 4, 6, 4, 7, 7, 8, 19, 30, 40, 50, 56]
    SO2_other = [15845, 10746, 16141, 14918, 15422, 122, 101, 102, 97, 98, 162, 4, 945, 611, 599, 979, 772, 1077, 962, 1355, 624, 1752, 2138, 3307, 4652]
    SO2_BIO   = [14261, 13979, 14055, 14217, 14884, 13890, 11536, 8441, 9476, 10141, 10100, 6706, 6131, 10210, 11779, 13344, 13225, 13352, 13925, 12901]
    SO2_OIL   = [4719, 3534, 310, 566, 1060, 6142, 7992, 8219, 11024, 9921, 7807, 4141, 3264, 3688, 2049, 2140, 1706, 1554, 1097, 803, 825, 2407, 353, 407, 748]
    SO2_BIO = [0]*(len(SO2_year) - len(SO2_BIO)) + SO2_BIO
    
    ys    = np.array([SO2_COA, 
                      SO2_NGA, 
                      SO2_other, 
                      SO2_BIO, 
                      SO2_OIL])/1.E3
    tech  = ['COA', 'NGA', 'other', 'BIO', 'OIL']
    tech.reverse()
    handles = list()
    ystack = np.cumsum(ys, axis=0)

    fig = plt.figure(1)
    ax = plt.subplot(111)

    for i in range(0, len(ystack[: ,1])):
        if i==0:
            t = tech.pop()
            h = ax.fill_between(SO2_year, 
                            0, ystack[i, ], 
                            facecolor=color_map[t],
                            edgecolor=None,
                            # alpha=.7,
                            hatch=hatch_map[t])
            handles.append(h)
        else:
            t = tech.pop()
            h = ax.fill_between(SO2_year, 
                                ystack[i-1, ], ystack[i, ], 
                                facecolor=color_map[t],
                                edgecolor=None,
                                # alpha=.7,
                                hatch=hatch_map[t])
            handles.append(h)
    
    h = ax.plot(LIM_year, LIM_SO2,  '-rs')
    h = mlines.Line2D([], [], color='red', marker='s')
    # h = ax.fill_between(LIM_year, 
    #                     0, LIM_SO2, 
    #                     facecolor='r', 
    #                     edgecolor=None,
    #                     alpha=.3,
    #                     hatch=None)
    handles.append(h)

    plt.ylabel('Emissions (kilotonnes)')
    plt.xlabel('Year')
    plt.title(r'(a) $\mathregular{SO}_2$ Emissions')
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width*0.9, box.height])
    ax.legend(handles, 
                ['Coal', 'Natural gas', 'Other', 'Bioenergy', 'Oil', 'Limit'], 
                # bbox_to_anchor = (1.01, 0.5), 
                loc='upper right')
    ax.annotate('2015', 
                xy=(2015, 600), xytext=(2014, 550), 
                color='r', 
                rotation=90)
    ax.axvline(x=2015, color='r')

    # CO2 emissions in metric tons
    ############################################################################
    CO2_year  = [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014]
    CO2_COA   = [48009174, 48798954, 55926125, 60469989, 55276926, 58094135, 65815061, 69150035, 68490646, 67709643, 71760129, 68956306, 70512977, 70564701, 71052843, 74265341, 71434913, 76239880, 72735826, 62208579, 68780238, 57423992, 49260274, 45374628, 46327216]
    CO2_NGA   = [221111, 293706, 325262, 332719, 278960, 615312, 388795, 579749, 1021037, 919829, 747840, 940758, 1750226, 793318, 1178029, 1484947, 1568101, 2225863, 1967927, 2158813, 4028363, 4946960, 8270796, 11022108, 11359219]
    CO2_other = [10130, 32752, 32752, 28599, 28599, 42657, 44334, 49560, 47645, 47534, 39393, 84664, 61823, 83234, 106052, 125382, 140090, 135346, 101005, 59245, 61568, 171384, 205928, 337075, 466382]
    CO2_BIO   = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    CO2_OIL   = [759462, 498727, 880758, 1094102, 1155916, 1196745, 1412981, 1376365, 1553466, 1637966, 1618922, 1277449, 1081080, 1272118, 966678, 872314, 820706, 815460, 539034, 418411, 370659, 255078, 186829, 205705, 425216]
    CO2_BIO = [0]*(len(CO2_year) - len(CO2_BIO)) + CO2_BIO

    ys    = np.array([CO2_COA, 
                      CO2_NGA, 
                      CO2_other, 
                      CO2_BIO, 
                      CO2_OIL])/1.E6
    tech  = ['COA', 'NGA', 'other', 'BIO', 'OIL']
    tech.reverse()
    handles = list()
    ystack = np.cumsum(ys, axis=0)

    fig = plt.figure(2)
    ax = plt.subplot(111)
    
    for i in range(0, len(ystack[: ,1])):
        if i==0:
            t = tech.pop()
            h = ax.fill_between(CO2_year, 
                            0, ystack[i, ], 
                            facecolor=color_map[t],
                            edgecolor=None,
                            # alpha=.7,
                            hatch=hatch_map[t])
            handles.append(h)
        else:
            t = tech.pop()
            h = ax.fill_between(CO2_year, 
                                ystack[i-1, ], ystack[i, ], 
                                facecolor=color_map[t],
                                edgecolor=None,
                                # alpha=.7,
                                hatch=hatch_map[t])
            handles.append(h)

    LIM_year = np.array(LIM_year)
    line_CO2 = ['--', '-']
    for i in range(0, len(LIM_CO2)):
        LIM_CO2  = np.array(LIM_CO2)
        x = LIM_year[np.nonzero( LIM_CO2[i] )]
        y = LIM_CO2[i][np.nonzero(LIM_CO2[i])]/1E3
        h = ax.plot(x, y,  color='red', marker='s', linestyle = line_CO2[i])
        h = mlines.Line2D([], [], color='red', marker='s', linestyle = line_CO2[i])
        # h = ax.fill_between(LIM_year[3:], 
        #                     0, LIM_CO2[3:], 
        #                     facecolor='r', 
        #                     edgecolor=None,
        #                     alpha=.3,
        #                     hatch=None)
        handles.append(h)

    plt.ylabel('Emissions (megatonnes)')
    plt.xlabel('Year')
    plt.title(r'$\mathregular{CO}_2$ Emissions')
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width*0.9, box.height])
    ax.legend(handles, 
                ['Coal', 'Natural gas', 'Other', 'Bioenergy', 'Oil', 'CPP', 'CP'], 
                # bbox_to_anchor = (1.01, 0.5), 
                loc='upper right')
    ax.annotate('2015', xy=(2015, 80000), xytext=(2014, 75000), 
                color='r', 
                rotation=90)
    ax.axvline(x=2015, color='r')

    # if len(arg) > 0:
    #     fname = arg[0]
    #     con = sqlite3.connect(fname)
    #     cur = con.cursor()
    #     qry = "SELECT * FROM Output_Emissions"
    #     cur.execute(qry)
    #     Output_Emissions = cur.fetchall()
    #     con.close()

    #     chemicals   = set()
    #     emissions  = dict(); emissions1  = dict()
    #     periods     = set()
    #     techs       = set()
    #     e_ctrls     = set()
    #     for row in Output_Emissions:
    #         s, p, e, t, v, value = row # To increase readability
    #         periods.add(p)
    #         if row[2] in tech_map:
    #             techs.add(tech_map[row[2]])
    #         if row[2] in emis_map:
    #             e_ctrls.add(emis_map[row[2]])
    #     periods = list(periods)
    #     periods.sort()


    #     for row in Output_Emissions:
    #         chemicals.add(row[2])
        
    #     for chemical in chemicals:
    #         temp = {
    #                 'NGA':   [0]*len(periods),
    #                 'COA':   [0]*len(periods),
    #                 'BIO':   [0]*len(periods),
    #                 'OIL':   [0]*len(periods),
    #                 'other': [0]*len(periods)
    #         }
    #         emissions1[chemical] = temp
    #         emissions[chemical]  = [0]*len(periods)
    #         emis_redct[chemical] = [0]*len(periods)
    #     for row in Output_Emissions:
    #         s, p, e, t, v, value = row # To increase readability
    #         emissions[e][periods.index(p)] += value
    #         if value < 0:
    #             emis_redct[e][periods.index(p)] += -value
    
    #         if value < 0:
    #             catagory = 'COA'
    #         elif t == 'E_EA_COAB':
    #             catagory = 'COA'
    #         elif t in tech_map:
    #             catagory = tech_map[t]
    #         else:
    #             catagory = 'other'
    
    #         emissions1[e][catagory][periods.index(p)] += value

    plt.show()

def plot_LC_compare():
    # width: 0.4, gap: 0.2
    w=0.2 # bar width
    x1 = range(0, 17)
    x2 = [i+w for i in x1]
    xticks = ['ENGACC05', 'ENGACT05', 'ENGAACC', 'ENGAACT', 'ENGACCCCS', 'ECOALSTM', 'ECOALIGCC', 'ECOALIGCCS', 'ECOALOXYCS', 'EURNALWR15', 'EBIOIGCC', 'EGEOBCFS', 'ESOLPVCEN', 'ESOLSTCEN', 'EWNDCL4', 'EWNDCL5', 'EWNDOFD']
    # Before accounting for T&D cost and inflation
    #y1b = [0.041918163, 0.061661346, 0.039752484, 0.053710534, 0.055225886, 0.050209656, 0.054849954, 0.074625307, 0.081597943, 0.038638264, 0.092045747, 0.025558711, 0.139658884, 0.062748259, 0.04566736, 0.059435332, 0.05747296]
    #y1t = [0.038317028, 0.058671079, 0.035190733, 0.052788694, 0.041512287, 0.008566557, 0.010196089, 0.009782192, 0.007637653, 0.001549599, 0.000952229, 0.000822821, 0.0001, 0.03827165, 0.001054504, 0.005551127, 0.005399723]
    
    #After accounting for T&D cost and inflation
    y1b = [0.084634166, 0.106893294, 0.082341356, 0.097667441, 0.10030487, 0.095717919, 0.103072857, 0.12593516, 0.133376426, 0.088481204, 0.14078029, 0.07184225, 0.157113353, 0.095341277, 0.096450102, 0.112946181, 0.110572093]
    y1t = [0.041967993, 0.057676809, 0.039564248, 0.051736615, 0.045478453, 0.0129411, 0.010866192, 0.012150004, 0.013144274, 0.003852888, 0.004575589, 0.004732148, 0.005606621, 0.026337069, 0.004451857, 0.005700534, 0.005517366]
    
    y2b = [0.084490738, 0.110999391, 0.08129096, 0.099901715, 0.098447315, 0.09435225, 0.101430307, 0.117370963, 0.117370963, 0.078190221, 0.140895246, 0.09013313, 0.194027645, 0.094436067, 0.097553663, 0.091310588, 0.08724533]
    y2t = [0.03840803, 0.058485785, 0.035039097, 0.053079191, 0.040815188, 0.008084994, 0.007994256, 0.007637653, 0.007637653, 0.0001, 0.0001, 0.0001, 0.0001, 0.024918495, 0.000762179, 0.000681972, 0.000629743]
    
    ax = plt.subplot(111)
    ax.bar(x1, y1b, width=w, color=[1, 1, 1], edgecolor=[1, 1, 1])
    h1 = ax.bar(x1, y1t, width=w, bottom=y1b, color='b', edgecolor='b')
    ax.bar(x2, y2b, width=w, color=[1, 1, 1], edgecolor=[1, 1, 1])
    h2 = ax.bar(x2, y2t, width=w, bottom=y2b, color='r', edgecolor='r')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0+box.height*0.1, box.width, box.height*0.9])
    ax.xaxis.grid(True)
    ax.legend([h1, h2], ['MARKAL2014', 'NUSTD'], loc='lower left')
    plt.xticks([i+w for i in x1], xticks, rotation='vertical')
    plt.ylabel('LC ($/kWh)')
    plt.title('T&D cost, inflation accounted for')
    plt.show()

def plot_NC_his():
    # Data from EIA. North Carolina Electricity Profile (2013). 
    # http://www.eia.gov/electricity/state/northcarolina/

    T = None  # None indicates show all years
    width = 0.5

    # Capacities in MW
    year  = [2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007, 2006, 2005, 2004, 2003, 2002, 2001, 2000, 1999, 1998, 1997, 1996, 1995, 1994, 1993, 1992, 1991, 1990]
    COA   = [10832.8, 10795, 12105, 12251, 12766, 12952, 13069, 13068, 13113, 13196, 13226, 13268, 13268, 13322, 13365, 13264, 13266, 13352, 13401, 13405, 13356, 13185, 13169, 13078, 13049]
    HYD   = [1999.1, 1997, 1964, 1964, 1956, 1952, 1952, 1960, 1954, 1945, 1951, 1939, 1914, 1867, 1860, 1860, 1990, 1927, 1916, 1890, 1818, 1815, 1815, 1827, 1827]
    NGA   = [10770.2, 10742, 10086, 8026, 6742, 6718, 6679, 6616, 5997, 5997, 6310, 6299, 5723, 4789, 3260, 1782, 1571, 1576, 1579, 469, 428, 425, 357, 262, 307]
    NUC   = [5094.1, 5076, 4998, 4970, 4958, 4958, 4958, 4975, 4975, 4938, 4938, 4783, 4731, 4731, 4691, 4691, 4691, 4749, 4639, 4639, 4639, 4639, 4639, 4639, 4698]
    other = [54.0, 54, 54, 50, 50, 50, 47, 37, 37, 37, 37, 37, 37, 180, 133, 133, 133, 133, 133, 37, 37, 37, 133, 117, 117]
    BIO   = [81.7, 60, 54, 41, 27, 20, 20, 18, 14, 14, 14, 14, 19, 16, 20, 14, 13, 13, 30, 10, 15, 15, 15, 5, 0]
    OIL   = [402.4, 402, 447, 544, 573, 560, 558, 564, 563, 594, 595, 885, 887, 839, 886, 843, 883, 879, 864, 1758, 868, 854, 826, 854, 816]
    PUM   = [86.0, 86, 86, 86, 86, 86, 90, 84, 84, 95, 95, 94, 94, 94, 94, 94, 0, 0, 0, 0, 68, 68, 68, 68, 68]
    SOL   = [676.0, 333, 115, 45, 35, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    wood  = [501.7, 502, 481, 481, 481, 318, 318, 324, 324, 291, 256, 250, 247, 239, 171, 294, 294, 208, 210, 186, 179, 179, 202, 277, 277]

    BIO = [BIO[i] + wood[i] for i in range(0, len(BIO))]

    # Convert MW into GJ
    if T is None:
        COA  = [i*1E-3 for i in COA]
        HYD  = [i*1E-3 for i in HYD]
        NGA  = [i*1E-3 for i in NGA]
        NUC  = [i*1E-3 for i in NUC]
        BIO  = [i*1E-3 for i in BIO]
        OIL  = [i*1E-3 for i in OIL]
        PUM  = [i*1E-3 for i in PUM]
        SOL  = [i*1E-3 for i in SOL]
    else:
        year = year[0: T]
        COA  = [i*1E-3 for i in COA[0: T]]
        HYD  = [i*1E-3 for i in HYD[0: T]]
        NGA  = [i*1E-3 for i in NGA[0: T]]
        NUC  = [i*1E-3 for i in NUC[0: T]]
        BIO  = [i*1E-3 for i in BIO[0: T]]
        OIL  = [i*1E-3 for i in OIL[0: T]]
        PUM  = [i*1E-3 for i in PUM[0: T]]
        SOL  = [i*1E-3 for i in SOL[0: T]]

    ys    = [BIO, OIL, HYD, NGA, PUM, SOL, COA, NUC]
    tech  = ['BIO', 'OIL', 'HYD', 'NGA', 'PUM', 'SOL', 'COA', 'NUC']
    b     = [0]*len(year)

    plt.figure(0) 
    handles = list()
    totals  = [0]*len(year) # Total annual generation by year
    ax = plt.subplot(111)
    for i in range(0, len(ys)):
        totals = [totals[j]+ys[i][j] for j in range(0, len(totals))]
        h = ax.bar(year,
                    ys[i],
                    width,
                    bottom=b,
                    color=color_map[tech[i]],
                    hatch=hatch_map[tech[i]]
                    )
        handles.append(h)
        b = [b[j]+ys[i][j] for j in range(0, len(b))]
        # b = [b[j] + max(ys[i]) for j in range(0, len(b))]
    plt.xticks([i + width/2. for i in year], [str(i) for i in year])
    plt.ylabel('Capacity (GW)')
    plt.title('Electric power capacity by primary energy source in North Carolina\nSource: NC Electricity Profile (EIA 2014)')
    for i in range(0, len(year)):
        height = totals[i] 
        ax.text(year[i]+width/2., 
                1.01*height, 
                '%d'%int(height), 
                ha='center', 
                va='bottom')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.9, box.height])
    ax.legend([h[0] for h in handles], 
                tech, 
                bbox_to_anchor = (1.01, 0.5), 
                loc='center left')
    ax.yaxis.grid(True)

    # Activities in MWh
    year  = [2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007, 2006, 2005, 2004, 2003, 2002, 2001, 2000, 1999, 1998, 1997, 1996, 1995, 1994, 1993, 1992, 1991, 1990]
    COA   = [49238197, 47072210, 50932180, 59758357, 71951214, 65082782, 75814787, 79983038, 75487005, 78435700, 75546115, 74776231, 75187741, 72957769, 75945463, 72255507, 72949464, 74072041, 69331347, 61484998, 58124391, 63572654, 58274018, 50767188, 49821595]
    HYD   = [4756083, 6900533, 3727938, 3893396, 4756549, 5171257, 3033642, 2984159, 3839012, 5396502, 5435199, 7200943, 3492048, 2595708, 3137815, 3684186, 5738207, 5625884, 5952176, 5520810, 7192276, 4987071, 5768453, 5850120, 6819427]
    NGA   = [28737608, 27982509, 19302008, 11155211, 8447237, 4851885, 4177342, 4456643, 3195563, 3159377, 2558654, 1580366, 3560871, 1534355, 1130253, 1166663, 1284357, 699937, 470138, 771688, 310242, 394751, 361308, 385907, 203466]
    NUC   = [40967020, 40241737, 39385592, 40526834, 40739529, 40847711, 39776280, 40044705, 39963184, 39981739, 40090623, 40906900, 39626849, 37775025, 39126881, 37523504, 38778211, 32453074, 33718182, 35910195, 32346007, 23758927, 22753813, 30312425, 25905319]
    other = [631153, 566884, 451865, 492937, 407440, 220428, 315642, 341852, 319235, 302737, 277403, 239177, 196202, 224973, 226443, 252784, 249356, 236051, 249410, 259130, 260777, 234982, 225149, 226926, 203406]
    BIO   = [519931, 410294, 302342, 375213, 195310, 131491, 120482, 86845, 91740, 98442, 112166, 114053, 103451, 108959, 109376, 92654, 88423, 78570, 55908, 54779, 45025, 42505, 30506, 44128, 26270]
    gas   = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 863, 990, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # Other gas
    OIL   = [459687, 217571, 178261, 217636, 293373, 296859, 320221, 495689, 451138, 518869, 610909, 783712, 592026, 655745, 825831, 662070, 691313, 496510, 547830, 479619, 452234, 407255, 335046, 275785, 328025]
    PUM   = [78009, 0, 0, 0, 0, 43077, -121064, 136996, 131342, 146505, 78251, 119273, 25175, 0, 108102, 175397, 65672, 253805, 340406, 167317, 380819, 246302, 109885, 216432, 180046]
    SOL   = [729130, 344663, 139491, 17381, 11340, 4563, 1801, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    wood  = [2026770, 2199893, 2262087, 1952891, 1876492, 1757350, 1799930, 1585374, 1736565, 1708706, 1620636, 1861663, 1682804, 1642330, 1664192, 1544851, 1533822, 1578818, 1532342, 1646120, 1716892, 1669998, 1677076, 1296373, 1438215]

    BIO = [BIO[i] + wood[i] for i in range(0, len(BIO))]

    # Convert MWh into PJ
    if T is None:
        COA  = [i*3.6E-6 for i in COA]
        HYD  = [i*3.6E-6 for i in HYD]
        NGA  = [i*3.6E-6 for i in NGA]
        NUC  = [i*3.6E-6 for i in NUC]
        BIO  = [i*3.6E-6 for i in BIO]
        OIL  = [i*3.6E-6 for i in OIL]
        PUM  = [i*3.6E-6 for i in PUM]
        SOL  = [i*3.6E-6 for i in SOL]
    else:
        year = year[0: T]
        COA  = [i*3.6E-6 for i in COA[0: T]]
        HYD  = [i*3.6E-6 for i in HYD[0: T]]
        NGA  = [i*3.6E-6 for i in NGA[0: T]]
        NUC  = [i*3.6E-6 for i in NUC[0: T]]
        BIO  = [i*3.6E-6 for i in BIO[0: T]]
        OIL  = [i*3.6E-6 for i in OIL[0: T]]
        PUM  = [i*3.6E-6 for i in PUM[0: T]]
        SOL  = [i*3.6E-6 for i in SOL[0: T]]

    ys    = [BIO, OIL, HYD, NGA, PUM, SOL, COA, NUC]
    tech  = ['BIO', 'OIL', 'HYD', 'NGA', 'PUM', 'SOL', 'COA', 'NUC']
    b     = [0]*len(year)

    plt.figure(1) 
    handles = list()
    totals  = [0]*len(year) # Total annual generation by year
    ax = plt.subplot(111)
    for i in range(0, len(ys)):
        totals = [totals[j]+ys[i][j] for j in range(0, len(totals))]
        h = ax.bar(year,
                    ys[i],
                    width,
                    bottom=b,
                    color=color_map[tech[i]],
                    hatch=hatch_map[tech[i]]
                    )
        handles.append(h)
        b = [b[j]+ys[i][j] for j in range(0, len(b))]
        # b = [b[j] + max(ys[i]) for j in range(0, len(b))]
    plt.xticks([i + width/2. for i in year], [str(i) for i in year])
    plt.ylabel('Activity (PJ)')
    plt.title('Electric power generation by primary energy source in North Carolina\nSource: NC Electricity Profile (EIA 2014)')
    for i in range(0, len(year)):
        height = totals[i] 
        ax.text(year[i]+width/2., 
                1.01*height, 
                '%d'%int(height), 
                ha='center', 
                va='bottom')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.9, box.height])
    ax.legend([h[0] for h in handles], 
                tech, 
                bbox_to_anchor = (1.01, 0.5), 
                loc='center left')
    ax.yaxis.grid(True)

    plt.show()

def plot_emis_his():
    # This function produces bar plots showing historic emissions of NOx, SO2 
    # and CO2.
    # Data from EIA's Electric Power Annual 2014, Bio and wood have already been
    # Combined into BIO.

    width = 0.5


    # NOx emissions in metric tons
    NOx_year  = [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014]
    NOx_COA   = [197050, 200392, 226735, 246162, 220491, 253394, 270469, 270411, 238762, 195076, 159338, 143831, 144080, 128733, 112585, 102783, 96728, 56664, 54641, 38686, 48840, 40003, 42225, 39754, 34518]
    NOx_NGA   = [517, 650, 708, 716, 548, 1388, 914, 1292, 2045, 1934, 1634, 1171, 1550, 571, 624, 957, 924, 1477, 794, 1142, 1752, 1681, 3135, 8672, 16066]
    NOx_other = [3918, 2481, 3665, 3592, 3566, 422, 358, 360, 351, 350, 296, 263, 534, 338, 356, 370, 401, 366, 279, 237, 214, 287, 351, 496, 664]
    NOx_BIO   = [5082, 4916, 5173, 5006, 4836, 5130, 4557, 3354, 3803, 1494, 1568, 1372, 1496, 3198, 3632, 5972, 6962, 7179, 7341, 8426]
    NOx_OIL   = [1235, 671, 1202, 1493, 1467, 2589, 3303, 3236, 3757, 3725, 3926, 2673, 2230, 3027, 886, 1014, 618, 639, 675, 550, 629, 1082, 302, 351, 1586]
    # NOx_wood  = []

    # NOx_BIO = [BIO[i] + wood[i] for i in range(0, len(BIO))]

    # SO2 emissions in metric tons
    SO2_year  = [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014]
    SO2_COA   = [343263, 340177, 385118, 428869, 377139, 399139, 473738, 505103, 483523, 451434, 450907, 429693, 437194, 436618, 442897, 468976, 438135, 356472, 223498, 112227, 115861, 73883, 57685, 46983, 39859]
    SO2_NGA   = [1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 2, 4, 7, 7, 4, 6, 4, 7, 7, 8, 19, 30, 40, 50, 56]
    SO2_other = [15845, 10746, 16141, 14918, 15422, 122, 101, 102, 97, 98, 162, 4, 945, 611, 599, 979, 772, 1077, 962, 1355, 624, 1752, 2138, 3307, 4652]
    SO2_BIO   = [14261, 13979, 14055, 14217, 14884, 13890, 11536, 8441, 9476, 10141, 10100, 6706, 6131, 10210, 11779, 13344, 13225, 13352, 13925, 12901]
    SO2_OIL   = [4719, 3534, 310, 566, 1060, 6142, 7992, 8219, 11024, 9921, 7807, 4141, 3264, 3688, 2049, 2140, 1706, 1554, 1097, 803, 825, 2407, 353, 407, 748]
    # SO2_wood  = []

    # SO2_BIO = [BIO[i] + wood[i] for i in range(0, len(BIO))]

    # CO2 emissions in metric tons
    CO2_year  = [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014]
    CO2_COA   = [48009174, 48798954, 55926125, 60469989, 55276926, 58094135, 65815061, 69150035, 68490646, 67709643, 71760129, 68956306, 70512977, 70564701, 71052843, 74265341, 71434913, 76239880, 72735826, 62208579, 68780238, 57423992, 49260274, 45374628, 46327216]
    CO2_NGA   = [221111, 293706, 325262, 332719, 278960, 615312, 388795, 579749, 1021037, 919829, 747840, 940758, 1750226, 793318, 1178029, 1484947, 1568101, 2225863, 1967927, 2158813, 4028363, 4946960, 8270796, 11022108, 11359219]
    CO2_other = [10130, 32752, 32752, 28599, 28599, 42657, 44334, 49560, 47645, 47534, 39393, 84664, 61823, 83234, 106052, 125382, 140090, 135346, 101005, 59245, 61568, 171384, 205928, 337075, 466382]
    CO2_BIO   = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    CO2_OIL   = [759462, 498727, 880758, 1094102, 1155916, 1196745, 1412981, 1376365, 1553466, 1637966, 1618922, 1277449, 1081080, 1272118, 966678, 872314, 820706, 815460, 539034, 418411, 370659, 255078, 186829, 205705, 425216]
    # CO2_wood  = []

    # NOx_BIO = [BIO[i] + wood[i] for i in range(0, len(BIO))]

    NOx_BIO = [0]*(len(NOx_year) - len(NOx_BIO)) + NOx_BIO
    SO2_BIO = [0]*(len(SO2_year) - len(SO2_BIO)) + SO2_BIO
    CO2_BIO = [0]*(len(CO2_year) - len(CO2_BIO)) + CO2_BIO

    NOx_year.reverse()
    NOx_COA.reverse()
    NOx_NGA.reverse()
    NOx_BIO.reverse()
    NOx_OIL.reverse()
    NOx_other.reverse()

    NOx_year  = np.array( NOx_year )
    NOx_COA   = np.array( NOx_COA )
    NOx_NGA   = np.array( NOx_NGA )
    NOx_BIO   = np.array( NOx_BIO )
    NOx_OIL   = np.array( NOx_OIL )
    NOx_Other = np.array( NOx_other )

    SO2_year.reverse()
    SO2_COA.reverse()
    SO2_NGA.reverse()
    SO2_BIO.reverse()
    SO2_OIL.reverse()
    SO2_other.reverse()

    SO2_year  = np.array( SO2_year )
    SO2_COA   = np.array( SO2_COA )
    SO2_NGA   = np.array( SO2_NGA )
    SO2_BIO   = np.array( SO2_BIO )
    SO2_OIL   = np.array( SO2_OIL )
    SO2_Other = np.array( SO2_other )

    CO2_year.reverse()
    CO2_COA.reverse()
    CO2_NGA.reverse()
    CO2_BIO.reverse()
    CO2_OIL.reverse()
    CO2_other.reverse()

    CO2_year  = np.array( CO2_year )
    CO2_COA   = np.array( CO2_COA )
    CO2_NGA   = np.array( CO2_NGA )
    CO2_BIO   = np.array( CO2_BIO )
    CO2_OIL   = np.array( CO2_OIL )
    CO2_Other = np.array( CO2_other )

    # NOx emissions
    ys    = np.array([NOx_BIO, NOx_OIL, NOx_NGA, NOx_COA, NOx_other])/1.E3
    tech  = ['BIO', 'OIL', 'NGA', 'COA', 'other']
    b     = [0]*len(NOx_year)

    plt.figure(2) 
    handles = list()
    totals  = [0]*len(NOx_year) # Total annual generation by year
    ax = plt.subplot(111)
    for i in range(0, len(ys)):
        totals = [totals[j]+ys[i][j] for j in range(0, len(totals))]
        h = ax.bar(NOx_year,
                    ys[i],
                    width,
                    bottom=b,
                    color=color_map[tech[i]],
                    hatch=hatch_map[tech[i]]
                    )
        handles.append(h)
        b = [b[j]+ys[i][j] for j in range(0, len(b))]
        # b = [b[j] + max(ys[i]) for j in range(0, len(b))]
    plt.xticks([i + width/2. for i in NOx_year], [str(i) for i in NOx_year])
    plt.ylabel('Emissions (thousand metric tons)')
    plt.title('NOx emissions by primary energy source in North Carolina\nSource: NC Electricity Profile (EIA 2014)')
    for i in range(0, len(NOx_year)):
        height = totals[i] 
        ax.text(NOx_year[i]+width/2., 1.01*height, 
                '%d'%int(height), 
                ha='center', 
                va='bottom')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.9, box.height])
    ax.legend([h[0] for h in handles], 
                tech, 
                bbox_to_anchor = (1.01, 0.5), 
                loc='center left')
    ax.yaxis.grid(True)

    
    # SO2 emission
    ys    = np.array([SO2_BIO, SO2_OIL, SO2_NGA, SO2_COA, SO2_other])/1.E3
    tech  = ['BIO', 'OIL', 'NGA', 'COA', 'other']
    b     = [0]*len(SO2_year)

    plt.figure(3) 
    handles = list()
    totals  = [0]*len(SO2_year) # Total annual generation by year
    ax = plt.subplot(111)
    for i in range(0, len(ys)):
        totals = [totals[j]+ys[i][j] for j in range(0, len(totals))]
        h = ax.bar(SO2_year,
                    ys[i],
                    width,
                    bottom=b,
                    color=color_map[tech[i]],
                    hatch=hatch_map[tech[i]]
                    )
        handles.append(h)
        b = [b[j]+ys[i][j] for j in range(0, len(b))]
        # b = [b[j] + max(ys[i]) for j in range(0, len(b))]
    plt.xticks([i + width/2. for i in SO2_year], [str(i) for i in SO2_year])
    plt.ylabel('Emissions (thousand metric tons)')
    plt.title('SO2 emissions by primary energy source in North Carolina\nSource: NC Electricity Profile (EIA 2014)')
    for i in range(0, len(SO2_year)):
        height = totals[i] 
        ax.text(SO2_year[i]+width/2., 1.01*height, 
                '%d'%int(height), 
                ha='center', 
                va='bottom')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.9, box.height])
    ax.legend([h[0] for h in handles], 
                tech, 
                bbox_to_anchor = (1.01, 0.5), 
                loc='center left')
    ax.yaxis.grid(True)

    # CO2 emissions
    ys    = np.array([CO2_BIO, CO2_OIL, CO2_NGA, CO2_COA, CO2_other])/1.E3
    tech  = ['BIO', 'OIL', 'NGA', 'COA', 'other']
    b     = [0]*len(CO2_year)

    plt.figure(4) 
    handles = list()
    totals  = [0]*len(CO2_year) # Total annual generation by year
    ax = plt.subplot(111)
    for i in range(0, len(ys)):
        totals = [totals[j]+ys[i][j] for j in range(0, len(totals))]
        h = ax.bar(CO2_year,
                    ys[i],
                    width,
                    bottom=b,
                    color=color_map[tech[i]],
                    hatch=hatch_map[tech[i]]
                    )
        handles.append(h)
        b = [b[j]+ys[i][j] for j in range(0, len(b))]
        # b = [b[j] + max(ys[i]) for j in range(0, len(b))]
    plt.xticks([i + width/2. for i in CO2_year], [str(i) for i in CO2_year])
    plt.ylabel('Emissions (thousand metric tons)')
    plt.title('CO2 emissions by primary energy source in North Carolina\nSource: NC Electricity Profile (EIA 2014)')
    for i in range(0, len(CO2_year)):
        height = totals[i] 
        ax.text(CO2_year[i]+width/2., 1.01*height, 
                '%d'%int(height), 
                ha='center', 
                va='bottom')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.9, box.height])
    ax.legend([h[0] for h in handles], 
                tech, 
                bbox_to_anchor = (1.01, 0.5), 
                loc='center left')
    ax.yaxis.grid(True)

    plt.show()

def plot_LCOE():
    # This function makas a figure that displays the LCOEs during the optimized
    # periods from each of the 6 scenarios.
    bar_width = 0.7
    left_position = np.array(range(2015, 2055, 5))
    scenarios = ['LF', 'R', 'HF', 'CPPLF', 'CPP', 'CPPHF']
    colors = {
        'LF':    [1, 1, 1],
        'R':     [1, 1, 1],
        'HF':    [1, 1, 1],
        'CPPLF': [0, 1, 0],
        'CPP':   [0, 1, 0],
        'CPPHF': [0, 1, 0]
    }
    hatchs = {
        'LF':    '//',
        'R':     None,
        'HF':    '\\\\',
        'CPPLF': '//',
        'CPP':   None,
        'CPPHF': '\\\\'
    }
    handles = list()
    
    LCOI = {
        'CPP': [
            4.202117448949164,
            0.8010075330553514,
            3.846303172911718,
            3.4726578122012395,
            2.163803539094672,
            1.9051492079654992,
            2.8924263964390504,
            1.7795921397533334],
        'CPPHF': [
            4.202117448949164,
            0.804261520333801,
            9.465665357455723,
            8.924038141303848,
            3.6426852909409213,
            4.322081668304149,
            4.626344974764035,
            3.044841482407985],
        'CPPLF': [
            3.5973572156772406,
            0.0964365334569726,
            3.9380605132824336,
            3.3010858755398087,
            2.07482041064344,
            1.4648327061126192,
            1.6304587362162046,
            2.0446234919589137],
        'HF': [
            4.198474865334963,
            1.0995917434549431,
            4.688946969176352,
            2.6590859711642123,
            2.8491194229259347,
            2.2319528578871246,
            1.1306483638438936,
            0.884015741145288],
        'LF': [
            3.5973572156772398,
            0.06239382673089822,
            2.9662110441661014,
            0.8911761130683306,
            0.3196556589555822,
            0.30904521073959085,
            0.016405147341945397,
            0.8994568504441772],
        'R': [
            4.198474865334963,
            1.238237531699957,
            2.998745385912655,
            1.5182492237292675,
            0.9549271146417107,
            0.76831468589389,
            0.0,
            0.011033096308838933]}
    
    LCOF = {
        'CPP': [
            7.537845460299389,
            7.217149837021663,
            7.222647806229216,
            7.3847298728021045,
            7.370704566768394,
            7.313799181342872,
            7.04992445322512,
            6.962480122011038],
        'CPPHF': [
            7.537845460299389,
            7.202393631134783,
            7.467113041165452,
            7.974639620493996,
            8.360955907850526,
            8.897885859302148,
            8.865519332029445,
            8.952907714637043],
        'CPPLF': [
            7.453862380730347,
            6.946426067475656,
            6.882822080470269,
            6.922425612542608,
            6.825033245383387,
            6.651466342200055,
            6.218410160984788,
            6.176086452567049],
        'HF': [
            7.531015475445532,
            7.224133868112752,
            7.276761473516486,
            7.256296465078654,
            7.268357368103723,
            7.2024075985460865,
            6.740550990532929,
            6.53587122104711],
        'LF': [7.453862380730346,
            6.939720114265965,
            6.7315319094696555,
            6.453307667972902,
            6.1320689769972,
            5.814780725259551,
            5.207752676758742,
            5.026288368446478],
        'R': [
            7.531015475445532,
            7.263549792509345,
            7.1273387765302845,
            7.007541507642699,
            6.829353703524555,
            6.634572210917631,
            6.04836826498253,
            5.747801240376743]}
    
    LCOV = {
        'CPP': [
            60.38916319521857,
            63.217633458349155,
            64.62367562891558,
            66.6006056783636,
            69.43982027933686,
            73.84001605480002,
            78.64869204843862,
            82.62676674775724],
        'CPPHF': [
            60.7479702935177,
            63.320156029791974,
            65.13764233453568,
            65.07377757299128,
            66.60051537689525,
            68.71610171638403,
            72.4635337587177,
            75.64657637390093],
        'CPPLF': [
            60.14674818405497,
            60.415895180334175,
            60.97504924422911,
            62.31740572920421,
            64.29704049353819,
            66.56510187579207,
            70.65638285310672,
            72.63166101521188],
        'HF': [
            60.79316985761363,
            63.23479998422471,
            66.8238773352666,
            68.45886494216394,
            70.11128809989232,
            73.40209808652737,
            79.82272342884045,
            84.51582103862827],
        'LF': [
            60.146748184054964,
            60.41172258424919,
            61.08366628322324,
            62.40527321805806,
            64.40999760174817,
            66.47236481508429,
            69.88443244130912,
            71.35259146462872],
        'R': [
            60.423122373158876,
            62.959776933464795,
            64.6809777237004,
            65.57883687992626,
            68.24461174933248,
            72.26856154779492,
            78.20357821467529,
            82.64343272876769]}

    plt.figure(0)
    ax = plt.subplot(111)
    for s in scenarios:
        h = plt.bar(left_position, np.array(LCOV[s]), 
                    bar_width, 
                    # alpha=0.3, 
                    color=[0.9*i for i in colors[s]],
                    hatch=hatchs[s])
        handles.append(h)
        h = plt.bar(left_position, np.array(LCOF[s]),
                    bar_width, 
                    bottom=np.array(LCOV[s]),
                    # alpha=0.5, 
                    color=[0.5*i for i in colors[s]],
                    hatch=hatchs[s])
        handles.append(h)
        h = plt.bar(left_position, np.array(LCOI[s]),
                    bar_width, 
                    bottom=np.array(LCOV[s])+np.array(LCOF[s]),
                    # alpha=1, 
                    color=[0.2*i for i in colors[s]],
                    hatch=hatchs[s])
        handles.append(h)
        left_position = left_position + bar_width
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height*0.9])
    # ax.legend([h[0] for h in handles], 
    #             techs, 
    #             bbox_to_anchor = (1.01, 0.5), 
    #             loc='center left')
    plt.xlabel('Periods')
    plt.ylabel('LCOE ($/MWh)')
    plt.xticks(left_position-3*bar_width, [str(p) for p in range(2015, 2055, 5)])
    ax.legend( handles, 
                ['LF var',    'LF fixed',    'LF invest', 
                 'R var',     'R fixed',     'R invest',
                 'HF var',    'HF fixed',    'HF invest',
                 'CPPLF var', 'CPPLF fixed', 'CPPLF invest',
                 'CPP var',   'CPP fixed',   'CPP invest',
                 'CPPHF var', 'CPPHF fixed', 'CPPHF invest'],
                bbox_to_anchor=(0.5, 1.2),
                ncol=6,
                loc=9)
    ax.yaxis.grid(True)
    plt.show()

def plot_LCOE_clean_center():
    # This function makas a figure that displays the LCOEs during the optimized
    # periods from each of the 6 scenarios.
    bar_width = 0.7
    left_position = np.array(range(2015, 2055, 5))
    scenarios = ['LF', 'R', 'HF', 'CPPLF', 'CPP', 'CPPHF']
    colors = {
        'LF':    [.3, .3, .3],
        'R':     'black',
        'HF':    [.9, .9, .9],
        'CPPLF': 'red',
        'CPP':   'green',
        'CPPHF': [6.0/255,50.0/255,208.0/255]
    }
    hatchs = {
        'LF':    '/',
        'R':     None,
        'HF':    '\\',
        'CPPLF': '/',
        'CPP':   None,
        'CPPHF': '\\'
    }
    handles = list()
    
    LCOI = {
        'CPP': [
            4.202117448949164,
            0.8010075330553514,
            3.846303172911718,
            3.4726578122012395,
            2.163803539094672,
            1.9051492079654992,
            2.8924263964390504,
            1.7795921397533334],
        'CPPHF': [
            4.202117448949164,
            0.804261520333801,
            9.465665357455723,
            8.924038141303848,
            3.6426852909409213,
            4.322081668304149,
            4.626344974764035,
            3.044841482407985],
        'CPPLF': [
            3.5973572156772406,
            0.0964365334569726,
            3.9380605132824336,
            3.3010858755398087,
            2.07482041064344,
            1.4648327061126192,
            1.6304587362162046,
            2.0446234919589137],
        'HF': [
            4.198474865334963,
            1.0995917434549431,
            4.688946969176352,
            2.6590859711642123,
            2.8491194229259347,
            2.2319528578871246,
            1.1306483638438936,
            0.884015741145288],
        'LF': [
            3.5973572156772398,
            0.06239382673089822,
            2.9662110441661014,
            0.8911761130683306,
            0.3196556589555822,
            0.30904521073959085,
            0.016405147341945397,
            0.8994568504441772],
        'R': [
            4.198474865334963,
            1.238237531699957,
            2.998745385912655,
            1.5182492237292675,
            0.9549271146417107,
            0.76831468589389,
            0.0,
            0.011033096308838933]}
    
    LCOF = {
        'CPP': [
            7.537845460299389,
            7.217149837021663,
            7.222647806229216,
            7.3847298728021045,
            7.370704566768394,
            7.313799181342872,
            7.04992445322512,
            6.962480122011038],
        'CPPHF': [
            7.537845460299389,
            7.202393631134783,
            7.467113041165452,
            7.974639620493996,
            8.360955907850526,
            8.897885859302148,
            8.865519332029445,
            8.952907714637043],
        'CPPLF': [
            7.453862380730347,
            6.946426067475656,
            6.882822080470269,
            6.922425612542608,
            6.825033245383387,
            6.651466342200055,
            6.218410160984788,
            6.176086452567049],
        'HF': [
            7.531015475445532,
            7.224133868112752,
            7.276761473516486,
            7.256296465078654,
            7.268357368103723,
            7.2024075985460865,
            6.740550990532929,
            6.53587122104711],
        'LF': [7.453862380730346,
            6.939720114265965,
            6.7315319094696555,
            6.453307667972902,
            6.1320689769972,
            5.814780725259551,
            5.207752676758742,
            5.026288368446478],
        'R': [
            7.531015475445532,
            7.263549792509345,
            7.1273387765302845,
            7.007541507642699,
            6.829353703524555,
            6.634572210917631,
            6.04836826498253,
            5.747801240376743]}
    
    LCOV = {
        'CPP': [
            60.38916319521857,
            63.217633458349155,
            64.62367562891558,
            66.6006056783636,
            69.43982027933686,
            73.84001605480002,
            78.64869204843862,
            82.62676674775724],
        'CPPHF': [
            60.7479702935177,
            63.320156029791974,
            65.13764233453568,
            65.07377757299128,
            66.60051537689525,
            68.71610171638403,
            72.4635337587177,
            75.64657637390093],
        'CPPLF': [
            60.14674818405497,
            60.415895180334175,
            60.97504924422911,
            62.31740572920421,
            64.29704049353819,
            66.56510187579207,
            70.65638285310672,
            72.63166101521188],
        'HF': [
            60.79316985761363,
            63.23479998422471,
            66.8238773352666,
            68.45886494216394,
            70.11128809989232,
            73.40209808652737,
            79.82272342884045,
            84.51582103862827],
        'LF': [
            60.146748184054964,
            60.41172258424919,
            61.08366628322324,
            62.40527321805806,
            64.40999760174817,
            66.47236481508429,
            69.88443244130912,
            71.35259146462872],
        'R': [
            60.423122373158876,
            62.959776933464795,
            64.6809777237004,
            65.57883687992626,
            68.24461174933248,
            72.26856154779492,
            78.20357821467529,
            82.64343272876769]}

    LCOE = {
        'CPP':   None,
        'CPPHF': None,
        'CPPLF': None,
        'HF':    None,
        'LF':    None,
        'R':     None
    }

    for k in LCOE:
        LCOE[k] = np.array(LCOI[k]) + np.array(LCOF[k]) + np.array(LCOV[k])

    plt.figure(0)
    ax = plt.subplot(111)
    scenarios.remove('LF')
    scenarios.remove('HF')
    for s in scenarios:
        # h = plt.bar(left_position, np.array(LCOV[s]), 
        #             bar_width, 
        #             # alpha=0.3, 
        #             color=[0.9*i for i in colors[s]],
        #             hatch=hatchs[s])
        # handles.append(h)
        # h = plt.bar(left_position, np.array(LCOF[s]),
        #             bar_width, 
        #             bottom=np.array(LCOV[s]),
        #             # alpha=0.5, 
        #             color=[0.5*i for i in colors[s]],
        #             hatch=hatchs[s])
        # handles.append(h)
        h = plt.bar(left_position, np.array(LCOE[s]),
                    bar_width, 
                    # bottom=np.array(LCOV[s])+np.array(LCOF[s]),
                    alpha=.8, 
                    color=colors[s])
        handles.append(h)
        left_position = left_position + bar_width
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height*0.9])
    # ax.legend([h[0] for h in handles], 
    #             techs, 
    #             bbox_to_anchor = (1.01, 0.5), 
    #             loc='center left')
    plt.xlabel('Periods')
    plt.ylabel('LCOE ($/MWh)')
    plt.xticks(left_position-3*bar_width, [str(p) for p in range(2015, 2055, 5)])
    ax.legend( handles, 
                ['R',
                 'CPPLF',
                 'CPP',   
                 'CPPHF'],
                bbox_to_anchor=(0.5, 1.2),
                ncol=6,
                loc=9)
    ax.yaxis.grid(True)
    plt.show()

def do_plot(inputs):

    fname = None
    scenario = None
    
    if inputs is None:
        raise "no arguments found"
        
    for opt, arg in inputs.iteritems():
        if opt in ("-f", "--fname"):
            fname = arg
        elif opt in ("-s", "--scenario"):
            scenario = arg
        elif opt in ("-h", "--help") :
            print "Use as :\n    python DB_to_Excel.py -f <fname> -s <scenario>    Use -h for help."                          
            sys.exit()
    print 'Read database: {}'.format(fname)
    print 'Plot out scenario: {}'.format(scenario)
    plot_result(fname, scenario)

if __name__ == "__main__":
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, "hf:s:", ["help", "fname=", "scenario="])
    print opts
    do_plot( dict(opts) )


    # plot_NCdemand_all()
    # plot_emis_all()    
    # plot_LC_compare()
    # plot_NC_his()
    # plot_emis_his()
    # plot_LCOE()
    # plot_LCOE_clean_center()

    # IP()
