from pyomo.environ import *
from pyomo.core import Constraint
from pyomo.opt import SolverFactory
import sys, xlsxwriter
from matplotlib import pyplot as plt
import numpy as np
from collections import OrderedDict, defaultdict
from time import time
from IPython import embed as IP

sys.path.append("/afs/unity.ncsu.edu/users/b/bli6/temoa/temoa_model")

def return_c_vector(block, unfixed):
    # Note that this function is adapted function collect_linear_terms defined
    # in pyomo/repn/collect.py.
    from pyutilib.misc import Bunch
    from pyomo.core.base import  Var, Constraint, Objective, maximize, minimize
    from pyomo.repn import generate_canonical_repn
    #
    # Variables are constraints of block
    # Constraints are unfixed variables of block and the parent model.
    #
    vnames = set()
    for (name, data) in block.component_map(Constraint, active=True).items():
        vnames.add((name, data.is_indexed()))
    cnames = set(unfixed)
    for (name, data) in block.component_map(Var, active=True).items():
        cnames.add((name, data.is_indexed()))
    #
    A = {}
    b_coef = {}
    c_rhs = {}
    c_sense = {}
    d_sense = None
    v_domain = {}
    #
    # Collect objective
    #
    for (oname, odata) in block.component_map(Objective, active=True).items():
        for ndx in odata:
            if odata[ndx].sense == maximize:
                o_terms = generate_canonical_repn(-1*odata[ndx].expr, compute_values=False)
                d_sense = minimize
            else:
                o_terms = generate_canonical_repn(odata[ndx].expr, compute_values=False)
                d_sense = maximize
            for i in range(len(o_terms.variables)):
                c_rhs[ o_terms.variables[i].parent_component().local_name, o_terms.variables[i].index() ] = o_terms.linear[i]
        # Stop after the first objective
        break
    return c_rhs

def sensitivity(dat, techs):
    from temoa_model import temoa_create_model
    model = temoa_create_model()
    
    model.dual  = Suffix(direction=Suffix.IMPORT)
    model.rc    = Suffix(direction=Suffix.IMPORT)
    model.slack = Suffix(direction=Suffix.IMPORT)
    model.lrc   = Suffix(direction=Suffix.IMPORT)
    model.urc   = Suffix(direction=Suffix.IMPORT)

    data = DataPortal(model = model)
    for d in dat:
        data.load(filename=d)
    instance = model.create_instance(data)
    optimizer = SolverFactory('cplex')
    optimizer.options['lpmethod'] = 1 # Use primal simplex
    results = optimizer.solve(instance, suffixes=['dual', 'urc', 'slack', 'lrc'])
    instance.solutions.load_from(results)

    coef_CAP = dict()
    scal_CAP = dict()
    # Break-even investment cost for this scenario, indexed by technology
    years    = list()
    bic_s    = dict()
    ic_s     = dict() # Raw investment costs for this scenario, indexed by tech
    cap_s    = dict()
    for t in techs:
        vintages = instance.vintage_optimize
        P_0 = min( instance.time_optimize )
        GDR = value( instance.GlobalDiscountRate )
        MLL = instance.ModelLoanLife
        MPL = instance.ModelProcessLife
        LLN = instance.LifetimeLoanProcess
        x   = 1 + GDR    # convenience variable, nothing more.

        bic_s[t] = list()
        ic_s[t]  = list()
        cap_s[t] = list()
        years = vintages.value
        for v in vintages:
            period_available = set()
            for p in instance.time_future:
                if (p, t, v) in instance.CostFixed.keys():
                    period_available.add(p)
            c_i = ( 
                    instance.CostInvest[t, v] 
                    * instance.LoanAnnualize[t, v] 
                    * ( LLN[t, v] if not GDR else 
                        (x**(P_0 - v + 1) 
                        * ( 1 - x **( -value(LLN[t, v]) ) ) 
                        / GDR) 
                      ) 
            )

            c_s = (-1)*(
                value( instance.CostInvest[t, v] )
                * value( instance.SalvageRate[t, v] )
                / ( 1 if not GDR else 
                    (1 + GDR)**( 
                        instance.time_future.last() 
                        - instance.time_future.first()
                        - 1
                        ) 
                    )
                )

            c_f = sum( 
                instance.CostFixed[p, t, v]
                * ( MPL[p, t, v] if not GDR else
                    (x**(P_0 - p + 1)
                    * ( 1 - x**( -value(MPL[p, t, v]) ) )
                    / GDR ) 
                  )
                for p in period_available 
            ) 

            c = c_i + c_s + c_f
            s = (c - instance.lrc[instance.V_Capacity[t, v]])/c
            coef_CAP[t, v] = c
            scal_CAP[t, v] = s # Must reduce TO this percentage
            bic_s[t].append(scal_CAP[t, v]*instance.CostInvest[t, v])
            ic_s[t].append(instance.CostInvest[t, v])
            cap_s[t].append( value( instance.V_Capacity[t, v] ) )

        # print "Tech\tVintage\tL. RC\tCoef\tU .RC\tScale\tBE IC\tBE FC\tIC\tFC\tCap"
        print "{:>10s}\t{:>7s}\t{:>6s}\t{:>4s}\t{:>6s}\t{:>5s}\t{:>7s}\t{:>7s}\t{:>5s}\t{:>3s}\t{:>5s}".format('Tech','Vintage', 'L. RC', 'Coef', 'U. RC', 'Scale', 'BE IC', 'BE FC', 'IC', 'FC', 'Cap')
        for v in vintages:
            lrc = instance.lrc[instance.V_Capacity[t, v]]
            urc = instance.urc[instance.V_Capacity[t, v]]

            # print "{:>s}\t{:>g}\t{:>.0f}\t{:>.0f}\t{:>.0f}\t{:>.3f}\t{:>.1f}\t{:>.1f}\t{:>.0f}\t{:>.0f}\t{:>.3f}".format(
            print "{:>10s}\t{:>7g}\t{:>6.0f}\t{:>4.0f}\t{:>6.0f}\t{:>5.3f}\t{:>7.1f}\t{:>7.1f}\t{:>5.0f}\t{:>3.0f}\t{:>5.3f}".format(
            t, v, lrc, coef_CAP[t, v], urc, scal_CAP[t, v], 
            scal_CAP[t, v]*instance.CostInvest[t, v], 
            scal_CAP[t, v]*instance.CostFixed[v, t, v], # Use the FC of the first period
            instance.CostInvest[t,v],
            instance.CostFixed[v, t, v],
            value(instance.V_Capacity[t, v])
            )

    IP()
    print 'Dual and slack variables for emission caps:'
    for e in instance.commodity_emissions:
        for p in instance.time_optimize:
            if (p, e) in instance.EmissionLimitConstraint:
                print p, e, instance.dual[instance.EmissionLimitConstraint[p, e]], '\t', instance.slack[instance.EmissionLimitConstraint[p, e]]
    return years, bic_s, ic_s

    print 'Dual and slack variables for Commodity Demand Constraints'
    for c in instance.commodity_demand:
        for p in instance.time_optimize:
            for s in instance.time_season:
                for tod in instance.time_of_day:
                    print p, s, tod, instance.dual[instance.DemandConstraint[p,s,tod,c]], instance.slack[instance.DemandConstraint[p,s,tod,c]]

def sen_bin_search():
    # Sensitivity analysis by binary search to find break-even cost
    # years, bic_s, ic_s, cap_s = sensitivity(dat, techs)
    target_year = 2020
    target_tech = 'ECOALIGCCS'
    dat = ['NCupdated.dat']
    # dat = ['/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20151124/reference.dat']
    epsilon = 5

    t0 = time()
    time_mark = lambda: time() - t0 
    from temoa_model import temoa_create_model
    model = temoa_create_model()
    
    model.dual  = Suffix(direction=Suffix.IMPORT)
    model.rc    = Suffix(direction=Suffix.IMPORT)
    model.slack = Suffix(direction=Suffix.IMPORT)
    model.lrc   = Suffix(direction=Suffix.IMPORT)
    model.urc   = Suffix(direction=Suffix.IMPORT)

    optimizer = SolverFactory('cplex')
    optimizer.options['lpmethod'] = 2 # Use dual simplex

    data = DataPortal(model = model)
    for d in dat:
        data.load(filename=d)
    instance = model.create_instance(data)

    ic = data['CostInvest'][target_tech, target_year]
    fc = data['CostFixed'][target_year, target_tech, target_year]

    # bic_u = data['CostInvest'][target_tech, target_year]
    # bfc_u = data['CostFixed'][target_year, target_tech, target_year]
    # bic_l = 0
    # bfc_l = 0
    cap_target = 0
    scale_u = 1.0
    scale_l = 0.0

    history = dict()
    # history['bic_l']   = [bic_l]
    # history['bic_u']   = [bic_u]
    # history['bfc_l']   = [bfc_l]
    # history['bfc_u']   = [bfc_u]
    history['scale_u'] = [scale_u]
    history['scale_l'] = [scale_l]

    counter = 0
    scale_this = scale_u # Starting scale
    # while (bic_u - bic_l) >= 5 and counter <= 20:
    while (scale_u - scale_l) >= 0.01 and counter <= 20:
        # ic_this = data['CostInvest'][target_tech, target_year]
        # fc_this = data['CostFixed'][target_year, target_tech, target_year]
        if cap_target <= 0:
            scale_u = scale_this
            history['scale_u'].append(scale_u)
        else:
            scale_l = scale_this
            history['scale_l'].append(scale_l)
        counter += 1

        scale_this = (scale_u + scale_l)*0.5
        data['CostInvest'][target_tech, target_year] = scale_this*ic
        for y in range(2015, 2055, 5):
            if (y, target_tech, target_year) in data['CostFixed']:
                data['CostFixed'][y, target_tech, target_year] = fc*scale_this


        print 'Iteration # {} starts at {} s'.format( counter, time_mark() )
        instance = model.create_instance(data)
        instance.preprocess()
        results = optimizer.solve(instance, suffixes=['dual', 'urc', 'slack', 'lrc'])
        instance.solutions.load_from(results)
        cap_target = value( instance.V_Capacity[target_tech, target_year] )
        print 'Iteration # {} solved at {} s'.format( counter, time_mark() )
        print 'Iteration # {}, scale: {:1.2f}, capacity: {} GW'.format( 
            counter,
            scale_this,
            cap_target)
    return

def sen_range():
    # Given a range of scaling factor for coefficient of a specific V_Capacity, 
    # returns objective value, reduced cost, capacity etc. for each scaling 
    # factor
    target_year = 2020
    target_tech = 'ECOALIGCCS'
    dat = ['NCupdated.dat']
    scales = [ 0.001* i for i in range(150, 305, 5) ]
    algmap = {
        'primal simplex': 1,
        'dual simplex':   2,
        'barrier':        4
    } # cplex definition

    t0 = time()
    time_mark = lambda: time() - t0 
    from temoa_model import temoa_create_model
    model = temoa_create_model()
    
    model.dual  = Suffix(direction=Suffix.IMPORT)
    model.rc    = Suffix(direction=Suffix.IMPORT)
    model.slack = Suffix(direction=Suffix.IMPORT)
    model.lrc   = Suffix(direction=Suffix.IMPORT)
    model.urc   = Suffix(direction=Suffix.IMPORT)

    data = DataPortal(model = model)
    for d in dat:
        data.load(filename=d)
    ic0 = data['CostInvest'][target_tech, target_year]
    fc0 = data['CostFixed'][target_year, target_tech, target_year]

    optimizer = SolverFactory('cplex')
    obj = dict()
    cap = dict()
    lrc = dict()
    urc = dict()
    bic = dict()
    bfc = dict()
    ic  = dict() # Original IC
    fc  = dict() # Original FC

    for algorithm in ['barrier', 'dual simplex', 'primal simplex']:
        optimizer.options['lpmethod'] = algmap[algorithm]
        print 'Algorithm: {}'.format( algorithm )

        obj_alg = list()
        cap_alg = defaultdict(list)
        lrc_alg = defaultdict(list)
        urc_alg = defaultdict(list)
        bic_alg = defaultdict(list)
        bfc_alg = defaultdict(list)
        ic_alg  = defaultdict(list)
        fc_alg  = defaultdict(list)
        for s in scales:
            print 'Scale: {:>.3f} starts at t = {:>7.2f} s'.format(s, time_mark() )
            data['CostInvest'][target_tech, target_year] = s*ic0
            for y in data['time_future']:
                if (y, target_tech, target_year) in data['CostFixed']:
                    data['CostFixed'][y, target_tech, target_year] = s*fc0
            instance = model.create_instance(data)
            instance.preprocess()
            results = optimizer.solve(instance, suffixes=['dual', 'urc', 'slack', 'lrc'])
            instance.solutions.load_from(results)

            obj_alg.append( value(instance.TotalCost) )
            for y in instance.time_optimize:
                key = str(y)
                c_vector = return_c_vector(instance, [])
                coef = c_vector[ ( 'V_Capacity', (target_tech, target_year) )]
                capacity = value(instance.V_Capacity[target_tech, target_year])
                lower_rc = value(
                    instance.lrc[ instance.V_Capacity[target_tech, target_year] ]
                )
                upper_rc = value(
                    instance.urc[ instance.V_Capacity[target_tech, target_year] ]
                )
                cost_i   = value( instance.CostInvest[target_tech, target_year] )
                cost_f   = value( instance.CostFixed[target_year, target_tech, target_year] )
                s_be = ( coef - lower_rc ) / coef # Break-even scale

                cap_alg[key].append( capacity )
                lrc_alg[key].append(lower_rc)
                urc_alg[key].append(upper_rc)
                ic_alg[key].append(cost_i)
                fc_alg[key].append(cost_f)
                bic_alg[key].append(s_be*cost_i)
                bfc_alg[key].append(s_be*cost_f)

            obj[algorithm] = obj_alg
            cap[algorithm] = cap_alg
            lrc[algorithm] = lrc_alg
            urc[algorithm] = urc_alg
            bic[algorithm] = bic_alg
            bfc[algorithm] = bfc_alg
            ic[algorithm]  = ic_alg
            fc[algorithm]  = fc_alg
    IP()


def explore_Cost_marginal(dat):
    from temoa_model import temoa_create_model
    model = temoa_create_model()
    
    model.dual  = Suffix(direction=Suffix.IMPORT)
    model.rc    = Suffix(direction=Suffix.IMPORT)
    model.slack = Suffix(direction=Suffix.IMPORT)
    model.lrc   = Suffix(direction=Suffix.IMPORT)
    model.urc   = Suffix(direction=Suffix.IMPORT)

    data = DataPortal(model = model)
    for d in dat:
        data.load(filename=d)
    instance = model.create_instance(data)

    # Deactivate the DemandActivity constraint
    # instance.DemandActivityConstraint.deactivate()
    # instance.preprocess()

    optimizer = SolverFactory('cplex')
    results = optimizer.solve(
        instance, 
        keepfiles=True,
        suffixes=['dual', 'urc', 'slack', 'lrc']
        )
    instance.solutions.load_from(results)

    print 'Dual and slack variables for emission caps:'
    for e in instance.commodity_emissions:
        for p in instance.time_optimize:
            if (p, e) in instance.EmissionLimitConstraint:
                print p, e, instance.dual[instance.EmissionLimitConstraint[p, e]], '\t', instance.slack[instance.EmissionLimitConstraint[p, e]]

    print 'Dual and slack variables for Commodity Demand Constraints'
    for c in instance.commodity_demand:
        for p in instance.time_optimize:
            for s in instance.time_season:
                for tod in instance.time_of_day:
                    print p, s, tod, instance.dual[instance.DemandConstraint[p,s,tod,c]], instance.slack[instance.DemandConstraint[p,s,tod,c]]

def do_sensitivity_old():
    scenarios = ['LF', 'R', 'HF', 'CPPLF', 'CPP', 'CPPHF']
    techs = ['ESOLPVCEN', 'EWNDON', 'EWNDOFS', 'EWNDOFD']
    dat = dict()

    dat['R']     = ['/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20151124/reference.dat']
    dat['HF']    = ['/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20151124/reference.high_oil.dat']
    dat['LF']    = ['/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20151124/reference.high_res.dat']
    dat['CPP']   = ['/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20151124/cpp.dat']
    dat['CPPHF'] = ['/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20151124/cpp.high_oil.dat']
    dat['CPPLF'] = ['/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20151124/cpp.high_res.dat']

    years = list()
    bic_ESOLPVCEN = dict()
    bic_ESOLPVDIS = dict()
    bic_EWNDOFS   = dict()
    bic_EWNDON    = dict()
    ic_ESOLPVCEN  = list()
    ic_ESOLPVDIS  = list()
    ic_EWNDOFS    = list()
    ic_EWNDON     = list()

    bic_EWNDOFD   = dict()
    ic_EWNDOFD    = list()
    for s in scenarios:
        print 'Scenario: ', s
        years, bic_s, ic_s = sensitivity(dat[s], techs)
        bic_ESOLPVCEN[s], ic_ESOLPVCEN = bic_s['ESOLPVCEN'], ic_s['ESOLPVCEN']
        bic_EWNDON[s], ic_EWNDON       = bic_s['EWNDON'], ic_s['EWNDON']
        bic_EWNDOFS[s], ic_EWNDOFS     = bic_s['EWNDOFS'], ic_s['EWNDOFS']
        bic_EWNDOFD[s], ic_EWNDOFD     = bic_s['EWNDOFD'], ic_s['EWNDOFD']
    
    h = plot_breakeven(years, bic_ESOLPVCEN, ic_ESOLPVCEN)
    plt.title('ESOLPVCEN')
    plt.savefig('ESOLPVCEN.svg')

    h = plot_breakeven(years, bic_EWNDON, ic_EWNDON)
    plt.title('EWNDON')
    plt.savefig('EWNDON.svg')

    h = plot_breakeven(years, bic_EWNDOFS, ic_EWNDOFS)
    plt.title('EWNDOFS')
    plt.savefig('EWNDOFS.svg')

    h = plot_breakeven(years, bic_EWNDOFD, ic_EWNDOFD)
    plt.title('EWNDOFD')
    plt.savefig('EWNDOFD.svg')

    plt.show()

def do_sensitivity_new():
    scenarios = ['LF', 'R', 'HD', 'CPPLF', 'CPP', 'CPPHD']
    # scenarios = ['LF', 'R', 'HF', 'HD', 'CPPLF', 'CPP', 'CPPHF', 'CPPHD']
    techs = ['EURNALWR15', 'ECOALIGCC', 'ECOALIGCCS', 'ENGACCCCS', 'EBIOIGCC', 'ECOALIGCC_b']
    # techs = ['ESOLPVCEN', 'EWNDON', 'EWNDOFS', 'ESOLPVDIS']
    dat = OrderedDict()

    ############################################################################
    # Normal run
    # dat['LF'] = [
    #     '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/results/LF/NCreference.LF.dat'
    # ]
    # dat['R'] = [
    #     '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/results/R/NCreference.R.dat'
    # ]
    # dat['HF'] = [
    #     '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/results/HF/NCreference.HF.dat'
    # ]
    # dat['HD'] = [
    #     '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/results/HD/NCreference.HD.dat'
    # ]
    # dat['CPPLF'] = [
    #     '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/results/CPPLF/NCreference.CPPLF.dat'
    # ]
    # dat['CPP'] = [
    #     '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/results/CPP/NCreference.CPP.dat'
    # ]
    # dat['CPPHF'] = [
    #     '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/results/CPPHF/NCreference.CPPHF.dat'
    # ]
    # dat['CPPHD'] = [
    #     '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/results/CPPHD/NCreference.CPPHD.dat'
    # ]

    ############################################################################
    # No RPS run
    dat['LF'] = [
        '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/sensitivity_noRSP/NCreference.LF_noRPS.dat'
    ]
    dat['R'] = [
        '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/sensitivity_noRSP/NCreference.R_noRPS.dat'
    ]
    dat['HF'] = [
        '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/sensitivity_noRSP/NCreference.HF_noRPS.dat'
    ]
    dat['HD'] = [
        '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/sensitivity_noRSP/NCreference.HD_noRPS.dat'
    ]
    dat['CPPLF'] = [
        '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/sensitivity_noRSP/NCreference.CPPLF_noRPS.dat'
    ]
    dat['CPP'] = [
        '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/sensitivity_noRSP/NCreference.CPP_noRPS.dat'
    ]
    dat['CPPHF'] = [
        '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/sensitivity_noRSP/NCreference.CPPHF_noRPS.dat'
    ]
    dat['CPPHD'] = [
        '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/sensitivity_noRSP/NCreference.CPPHD_noRPS.dat'
    ]

    ############################################################################
    # No RPS and no NG cap run
    # dat['LF'] = [
    #     '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/sensitivity_noRPS_noNGcap/NCreference.LF_noRPS.dat'
    # ]
    # dat['R'] = [
    #     '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/sensitivity_noRPS_noNGcap/NCreference.R_noRPS.dat'
    # ]
    # dat['HF'] = [
    #     '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/sensitivity_noRPS_noNGcap/NCreference.HF_noRPS.dat'
    # ]
    # dat['HD'] = [
    #     '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/sensitivity_noRPS_noNGcap/NCreference.HD_noRPS.dat'
    # ]
    # dat['CPPLF'] = [
    #     '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/sensitivity_noRPS_noNGcap/NCreference.CPPLF_noRPS.dat'
    # ]
    # dat['CPP'] = [
    #     '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/sensitivity_noRPS_noNGcap/NCreference.CPP_noRPS.dat'
    # ]
    # dat['CPPHF'] = [
    #     '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/sensitivity_noRPS_noNGcap/NCreference.CPPHF_noRPS.dat'
    # ]
    # dat['CPPHD'] = [
    #     '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/sensitivity_noRPS_noNGcap/NCreference.CPPHD_noRPS.dat'
    # ]

    ############################################################################
    # No NG cap run
    # dat['LF'] = [
    #     '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/results/LF_noNGcap/NCreference.LF_noNGcap.dat'
    # ]
    # dat['R'] = [
    #     '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/results/R_noNGcap/NCreference.R_noNGcap.dat'
    # ]
    # dat['HD'] = [
    #     '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/results/HD_noNGcap/NCreference.HD_noNGcap.dat'
    # ]
    # dat['CPPLF'] = [
    #     '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/results/CPPLF_noNGcap/NCreference.CPPLF_noNGcap.dat'
    # ]
    # dat['CPP'] = [
    #     '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/results/CPP_noNGcap/NCreference.CPP_noNGcap.dat'
    # ]
    # dat['CPPHD'] = [
    #     '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/results/CPPHD_noNGcap/NCreference.CPPHD_noNGcap.dat'
    # ]

    years = list()
    # bic_ESOLPVCEN = dict()
    # bic_ESOLPVDIS = dict()
    # bic_EWNDOFS   = dict()
    # bic_EWNDON    = dict()
    # ic_ESOLPVCEN  = list()
    # ic_ESOLPVDIS  = list()
    # ic_EWNDOFS    = list()
    # ic_EWNDON     = list()

    for s in scenarios:
        print 'Scenario: ', s
        years, bic_s, ic_s = sensitivity(dat[s], techs)
        # bic_ESOLPVCEN[s], ic_ESOLPVCEN = bic_s['ESOLPVCEN'], ic_s['ESOLPVCEN']
        # bic_ESOLPVDIS[s], ic_ESOLPVDIS = bic_s['ESOLPVDIS'], ic_s['ESOLPVDIS']
        # bic_EWNDON[s], ic_EWNDON       = bic_s['EWNDON'], ic_s['EWNDON']
        # bic_EWNDOFS[s], ic_EWNDOFS     = bic_s['EWNDOFS'], ic_s['EWNDOFS']
    
    for t in techs:
        h = plot_breakeven(years, bic_s[t], ic_s[t])
        plt.title(t)
        plt.savefig(t + '.svg')

    # h = plot_breakeven(years, bic_ESOLPVCEN, ic_ESOLPVCEN)
    # plt.title('ESOLPVCEN')
    # plt.savefig('ESOLPVCEN.svg')

    # h = plot_breakeven(years, bic_ESOLPVDIS, ic_ESOLPVDIS)
    # plt.title('ESOLPVDIS')
    # plt.savefig('ESOLPVDIS.svg')

    # h = plot_breakeven(years, bic_EWNDON, ic_EWNDON)
    # plt.title('EWNDON')
    # plt.savefig('EWNDON.svg')

    # h = plot_breakeven(years, bic_EWNDOFS, ic_EWNDOFS)
    # plt.title('EWNDOFS')
    # plt.savefig('EWNDOFS.svg')

    # plt.show()

def LC_calculate(dat):

    # Electricity generating technologies
    tech_gen = [
        'IMPELCNGCEA',
        'IMPELCNGAEA',
        'IMPELCDSLEA',
        'IMPURNA',
        'IMPELCBIGCCEA',
        'IMPELCBIOSTM',
        'IMPELCGEO',
        'IMPSOL',
        'IMPWND',
        'IMPELCHYD',
        'IMPLFGICEEA',
        'IMPLFGGTREA',
        'IMPELCCOAB',

        'ENGACC05',
        'ENGACT05',
        'ENGAACC',
        'ENGAACT',
        'ENGACCCCS',
        'ENGACCR',
        'ENGACTR',
        'ECOALSTM',
        'ECOALIGCC',
        'ECOALIGCCS',
        'ECOASTMR',
        'EDSLCTR',
        'EURNALWR',
        'EURNALWR15',
        'EBIOIGCC',
        'EBIOSTMR',        
        'EGEOBCFS',        
        'ESOLPVCEN',       
        'ESOLSTCEN',       
        'ESOLPVR',         
        'EWNDON',          
        'EWNDOFS',         
        'EWNDOFD',         
        'EHYDCONR',        
        'EHYDREVR',        
        'ELFGICER',        
        'ELFGGTR',         
        'EHYDGS']
    
    tech_misc = [
        'E_BLND_BITSUBLIG_COALSTM_R',   # blending tech to collect bit subbit and lig coal for existing coal steam plant');
        'E_BLND_BIT_COALSTM_R',         # blending tech to collect bit coal for existing coal steam plant');
        'E_PTNOXSCR_COAB',              # nox passthrough tech for bituminous coal after LNB retrofit or passthrough and before existing coal steam plant');
        'E_PTNOXLNB_COAB',              # nox passthrough tech for bituminous coal after so2 or co2 passthrough and before SCR or SNCR or passthrough');
        'E_PTCO2_COAB',                 # co2 passthrough tech for bituminous coal after FGD and before LNB');
        'E_EA_COAB',                    # co2 emission accounting tech for coal bituminous');
        'E_PTSO2_COABH',                # passthrough tech with no so2 removal from bit high sulfur before existing coal steam plant');
        'E_PTSO2_COABM',                # passthrough tech with no so2 removal from bit medium sulfur before existing coal steam plant');
        'E_PTSO2_COABL',                # passthrough tech with no so2 removal from bit low sulfur before existing coal steam plant');
        'E_BLND_BITHML_COALIGCC_N',     # blending tech to collect high medium low sulfur bit coal for new coal IGCC plant');
        'E_BLND_BITSUBLIG_COALSTM_N',   # blending tech to collect bit subbit and lig coal for new coal steam plant');
        'E_BLND_BITHML_COALSTM_N',      # blending tech to collect high medium low sulfur bit coal for new coal steam plant');
        'ELC2DMD'
        ]
    
    # NOx removal technologies
    tech_NOx = [
        'E_LNBSNCR_COAB_R', # Existing LNB w/SNCR retrofit for nox removal from BIT before existing coal STM');
        'E_LNBSNCR_COAB_N', # new LNB combined with SNCR retrofit for nox removal from bituminous before existing coal steam plant');
        'E_LNBSCR_COAB_R',  # existing LNB combined with SCR retrofit for nox removal from bituminous before existing coal steam plant');
        'E_LNBSCR_COAB_N',  # new LNB combined with SCR retrofit for nox removal from bituminous before existing coal steam plant');
        'E_SNCR_COAB_R',    # existing SNCR retrofit for nox removal from bituminous before existing coal steam plant');
        'E_SNCR_COAB_N',    # new SNCR retrofit for nox removal from bituminous before existing coal steam plant');
        'E_SCR_COAB_N',     # new SCR retrofit for nox removal from bituminous before existing coal steam plant');
        'E_LNB_COAB_R',     # existing LNB retrofit tech for nox removal from bituminous before existing coal steam plant');
        'E_LNB_COAB_N'     # new LNB retrofit tech for nox removal from bituminous before existing coal steam plant
        ]
    
    # SO2 removal technologies
    tech_SO2 = [
        'E_FGD_COABH_N', # new FGD retrofit tech for so2 removal from bit high sulfur before existing coal steam plant');
        'E_FGD_COABM_R', # existing FGD retrofit tech for so2 removal from bit medium sulfur before existing coal steam plant');
        'E_FGD_COABM_N', # new FGD retrofit tech for so2 removal from bit medium sulfur before existing coal steam plant');
        'E_FGD_COABL_R', # existing FGD retrofit tech for so2 removal from bit low sulfur before existing coal steam plant');
        'E_FGD_COABL_N'  # new FGD retrofit tech for so2 removal from bit low sulfur before existing coal steam plant');
        ]
    
    # CO2 removal technologies
    tech_CO2 = [
        'E_CCR_COAB',                   # co2 capture retrofit tech for bituminous coal to existing power plant located after FGD or passthrough and before LNB');
        'E_CCR_COALIGCC_N',             # co2 capture retrofit tech before coal IGCC plant');
        'E_CCR_COALSTM_N',              # co2 capture retrofit tech before new coal steam plant');
        'E_BLND_BITSUBLIG_COALIGCC_N'   # blending tech to collect bit subbit and lig coal for new coal IGCC plant');
        ]

    data = DataPortal(model = model)
    data.load(filename=dat)
    instance = model.create_instance(data)
    optimizer = SolverFactory('cplex')
    results = optimizer.solve(instance, suffixes=['dual', 'rc', 'slack', 'lrc'])
    instance.solutions.load_from(results)

    m = instance
    TAE  = dict()
    # Cost format: [Gen, NOx, SO2, CO2, misc] where misc supposed to be 0, 
    # otherwise there is something wrong with the cost classification table.
    TAIC = dict() # Total annualized investment cost
    TAFC = dict() # Total annualized fixed O&M cost
    TAVC = dict() # Total annualized variable O&M cost
    TAMC = dict() 
    TAC  = dict() # Total annualized cost
    
    LCOE = dict() # Levelized cost of energy
    LCOI = dict() # Levelized cost of investment
    LCOF = dict() # Levelized cost of fixed O&M
    LCOV = dict() # Levelized cost of variable O&M

    for year in m.time_optimize:
        year_str = str(year)

        # Annual energy output
        TAE[year_str] = sum(
            1/3.6 # Convert PJ to TWh
            *value( m.V_ActivityByPeriodTechAndOutput[S_p, S_t, S_o] )
            for S_p, S_t, S_o in m.V_ActivityByPeriodTechAndOutput.iterkeys()
            if S_p == year and S_o == 'ELC'
            )
        
        # Annualized investment cost
        TAIC[year_str] = [0, 0, 0, 0, 0] 
        TAFC[year_str] = [0, 0, 0, 0, 0] 
        TAVC[year_str] = [0, 0, 0, 0, 0] 
        for S_t, S_v in m.CostInvest.sparse_iterkeys():
            if S_v == year:
                if S_t in tech_gen:
                    TAIC[year_str][0] += (
                        value( m.CostInvest[S_t, S_v] )
                        *value( m.LoanAnnualize[S_t, S_v] )
                        *value( m.V_Capacity[S_t, S_v] )
                    )
                elif S_t in tech_NOx:
                    TAIC[year_str][1] += (
                        value( m.CostInvest[S_t, S_v] )
                        *value( m.LoanAnnualize[S_t, S_v] )
                        *value( m.V_Capacity[S_t, S_v] )
                    )
                elif S_t in tech_SO2:
                    TAIC[year_str][2] += (
                        value( m.CostInvest[S_t, S_v] )
                        *value( m.LoanAnnualize[S_t, S_v] )
                        *value( m.V_Capacity[S_t, S_v] )
                    )
                elif S_t in tech_CO2:
                    TAIC[year_str][3] += (
                        value( m.CostInvest[S_t, S_v] )
                        *value( m.LoanAnnualize[S_t, S_v] )
                        *value( m.V_Capacity[S_t, S_v] )
                    )
                else:
                    TAIC[year_str][4] += (
                        value( m.CostInvest[S_t, S_v] )
                        *value( m.LoanAnnualize[S_t, S_v] )
                        *value( m.V_Capacity[S_t, S_v] )
                    )
            else:
                continue
        
        # Annualized fixed cost
        for S_p, S_t, S_v in m.CostFixed.sparse_iterkeys():
            if S_p == year:
                if S_t in tech_gen:
                    TAFC[year_str][0] += (
                        value( m.CostFixed[S_p, S_t, S_v] )
                        *value( m.V_Capacity[S_t, S_v] )
                    )
                elif S_t in tech_NOx:
                    TAFC[year_str][1] += (
                        value( m.CostFixed[S_p, S_t, S_v] )
                        *value( m.V_Capacity[S_t, S_v] )
                    )
                elif S_t in tech_SO2:
                    TAFC[year_str][2] += (
                        value( m.CostFixed[S_p, S_t, S_v] )
                        *value( m.V_Capacity[S_t, S_v] )
                    )
                elif S_t in tech_CO2:
                    TAFC[year_str][3] += (
                        value( m.CostFixed[S_p, S_t, S_v] )
                        *value( m.V_Capacity[S_t, S_v] )
                    )
                else:
                    TAFC[year_str][4] += (
                        value( m.CostFixed[S_p, S_t, S_v] )
                        *value( m.V_Capacity[S_t, S_v] )
                    )
            else:
                continue
        
        # Annualized variable cost
        for S_p, S_t, S_v in m.CostVariable.sparse_iterkeys():
            if S_p == year:
                if S_t in tech_gen:
                    TAVC[year_str][0] += (
                        value( m.CostVariable[S_p, S_t, S_v] )
                        *value( m.V_ActivityByPeriodAndProcess[S_p, S_t, S_v] )
                    )
                elif S_t in tech_NOx:
                    TAVC[year_str][1] += (
                        value( m.CostVariable[S_p, S_t, S_v] )
                        *value( m.V_ActivityByPeriodAndProcess[S_p, S_t, S_v] )
                    )
                elif S_t in tech_SO2:
                    TAVC[year_str][2] += (
                        value( m.CostVariable[S_p, S_t, S_v] )
                        *value( m.V_ActivityByPeriodAndProcess[S_p, S_t, S_v] )
                    )
                elif S_t in tech_CO2:
                    TAVC[year_str][3] += (
                        value( m.CostVariable[S_p, S_t, S_v] )
                        *value( m.V_ActivityByPeriodAndProcess[S_p, S_t, S_v] )
                    )
                else:
                    TAVC[year_str][4] += (
                        value( m.CostVariable[S_p, S_t, S_v] )
                        *value( m.V_ActivityByPeriodAndProcess[S_p, S_t, S_v] )
                    )
            else:
                continue

        TAC[year_str] = [
            TAIC[year_str][i] + 
            TAFC[year_str][i] + 
            TAVC[year_str][i] for i in range( 0, len(TAIC[year_str]) )
        ]
        LCOE[year_str] = [ i/TAE[year_str] for i in TAC[year_str] ]

        LCOI[year_str] = [ i/TAE[year_str] for i in TAIC[year_str] ]
        LCOF[year_str] = [ i/TAE[year_str] for i in TAFC[year_str] ]
        LCOV[year_str] = [ i/TAE[year_str] for i in TAVC[year_str] ]
        
        # print year_str, LCOE[year_str], sum(LCOI[year_str]), sum(LCOF[year_str]), sum(LCOV[year_str])

    return LCOI, LCOF, LCOV 

def do_LCcalculate():
    # print "Reference scenario"
    # dat = 'sql/reference.dat'
    # LC_calculate(dat)
    # print "High oil scenario"
    # dat = 'sql/reference.high_oil.dat'
    # LC_calculate(dat)
    # print "Low oil scenario"
    # dat = 'sql/reference.high_res.dat'
    # LC_calculate(dat)

    # print "CPP scenario"
    # dat = 'sql/cpp.dat'
    # LC_calculate(dat)
    # print "CPP + high oil scenario"
    # dat = 'sql/cpp.high_oil.dat'
    # LC_calculate(dat)
    # print "CPP + low oil scenario"
    # dat = 'sql/cpp.high_res.dat'
    # LC_calculate(dat)

    print "Test scenario"
    dat = 'sql20151124.test.dat'
    LC_calculate(dat)

def plot_LCOE():
    periods = range(2015, 2055, 5)
    LCOI = dict()
    LCOF = dict()
    LCOV = dict()
    scenarios = {'R':     'sql20151124/reference.dat',
                 'LF':    'sql20151124/reference.high_res.dat',
                 'HF':    'sql20151124/reference.high_oil.dat',
                 'CPP':   'sql20151124/cpp.dat',
                 'CPPLF': 'sql20151124/cpp.high_res.dat',
                 'CPPHF': 'sql20151124/cpp.high_oil.dat'
                }
    for s in scenarios.keys():
        print "Scenario: ", s
        LCOI_split, LCOF_split, LCOV_split = LC_calculate(scenarios[s])
        LCOI[s] = [sum(LCOI_split[str(p)]) for p in periods]
        LCOF[s] = [sum(LCOF_split[str(p)]) for p in periods]
        LCOV[s] = [sum(LCOV_split[str(p)]) for p in periods]

    bar_width = 0.7
    left_position = np.array(range(2015, 2055, 5))
    # c_ref = [0.1, 0.1, 0.1]
    # c_cpp = 'g'
    colors = {
        'LF':    [1, 1, 1],
        'R':     [1, 1, 1],
        'HF':    [1, 1, 1],
        'CPPLF': [0, 1, 0],
        'CPP':   [0, 1, 0],
        'CPPHF': [0, 1, 0]
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

    for s in ['LF', 'R', 'HF', 'CPPLF', 'CPP', 'CPPHF']:
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

    plt.xlabel('Periods')
    plt.ylabel('LCOE ($/MWh)')
    plt.xticks(left_position-3*bar_width, [str(p) for p in range(2015, 2055, 5)])
    plt.show()

def plot_breakeven(years, bic, ic):
    # bic is a dictionary, ic is a list of the raw investment costs
    # ic = [x, x, ..., x], the length of which equals to the length of years
    # bic[scenario] = [x, x, x... x] where the length equals to the number 
    # of optimized periods.
    sen_color_map = {
        'IC':    [0.9, 0.9, 0.9],
        'LF':    'black',
        'R':     'black',
        'HF':    'black',
        'HD':    'black',
        'CPPLF': 'green',
        'CPP':   'green',
        'CPPHF': 'green',
        'CPPHD': 'green'
    }

    sen_lstyle_map = {
        'IC':    None,
        'LF':    '--',
        'R':     '-',
        'HF':    '-.',
        'HD':    ':',
        'CPPLF': '--',
        'CPP':   '-',
        'CPPHF': '-.',
        'CPPHD': ':'
    }

    sen_marker_map = {
        'IC':    None,
        'LF':    's',
        'R':     's',
        'HF':    's',
        'HD':    's',
        'CPPLF': 's',
        'CPP':   's',
        'CPPHF': 's',
        'CPPHD': 's'
    }

    scenarios = bic.keys()
    h = plt.figure()
    ax = plt.subplot(111)
    ax.fill_between(years, 0, ic, 
        facecolor = sen_color_map['IC']
        )

    for s in scenarios:
        ax.plot(years, bic[s], 
            color = sen_color_map[s],
            # marker = sen_marker_map[s],
            linestyle = sen_lstyle_map[s]
            )
    ax.yaxis.grid(True)
    plt.ylabel('$/MWh')
    plt.xlim( ( years[0]-5, years[-1]+5 ) )
    return ax

if __name__ == "__main__":
    # sen_bin_search()
    sen_range()
    # do_sensitivity_new()
    # do_sensitivity_old()
    # explore_Cost_marginal(['/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/results/R/NCreference.R.dat'])