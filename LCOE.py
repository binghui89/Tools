import sqlite3, sys
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
from collections import OrderedDict
from IPython import embed as IP

# Electricity cost (2015 to 2050, every 5 years) from AEO 2017, in 2016 $/MWh.
# http://www.eia.gov/outlooks/aeo/data/browser/#/?id=8-AEO2017&region=0-0&cases=ref2017&start=2015&end=2050&f=Q&linechart=ref2017-d120816a.6-8-AEO2017&sid=&sourcekey=0
generation_aeo17   = [65, 62, 66, 71, 69, 68, 69, 72]
transmission_aeo17 = [11, 12, 13, 13, 14, 14, 14, 14]
distribution_aeo17 = [29, 31, 33, 31, 32, 32, 32, 32]

cost_var_coal = {
	'L': {
		(2015, ): 2.63,
		(2020, ): 2.65,
		(2025, ): 2.59,
		(2030, ): 2.59,
		(2035, ): 2.64,
		(2040, ): 2.70,
		(2045, ): 2.67,
		(2050, ): 2.67,
	},
	'H': {
		(2015, ): 2.62,
		(2020, ): 2.79,
		(2025, ): 2.70,
		(2030, ): 2.66,
		(2035, ): 2.74,
		(2040, ): 2.85,
		(2045, ): 2.89,
		(2050, ): 2.94,
	}
}

cost_var_ng = {
	'L': {
		(2015, ): 3.74,
		(2020, ): 4.21,
		(2025, ): 4.23,
		(2030, ): 4.44,
		(2035, ): 4.20,
		(2040, ): 4.29,
		(2045, ): 4.06,
		(2050, ): 4.09,
	},
	'H': {
		(2015, ):  3.74,
		(2020, ):  5.38,
		(2025, ):  7.71,
		(2030, ):  9.00,
		(2035, ):  9.41,
		(2040, ): 10.19,
		(2045, ): 10.07,
		(2050, ): 10.07,
	}
}

cost_var_oil = {
	'L': {
		(2015, ): 14.09,
		(2020, ): 16.65,
		(2025, ): 18.81,
		(2030, ): 20.50,
		(2035, ): 20.31,
		(2040, ): 21.68,
		(2045, ): 21.79,
		(2050, ): 22.56,
	},
	'H':{
		(2015, ): 14.27,
		(2020, ): 17.04,
		(2025, ): 19.41,
		(2030, ): 21.05,
		(2035, ): 22.34,
		(2040, ): 24.02,
		(2045, ): 24.47,
		(2050, ): 25.26,
	}
}


def plot_LCOE(LCOI, LCOF, LCOV):
	# This function makas a figure that displays the LCOEs during the optimized
	# periods from each of the 6 scenarios.
	LCOE_aeo17 = (
		np.array(generation_aeo17) + 
		np.array(transmission_aeo17) + 
		np.array(distribution_aeo17)
		)

	scenarios = [s for s in LCOI]
	bar_width = (5.0 - 1)/len(scenarios)
	left_position = np.array(range(2015, 2055, 5))
	# scenarios = ['LF', 'R', 'HF', 'CPPLF', 'CPP', 'CPPHF']
	colors = {
		'LF':    [1, 1, 1],
		'R':     [1, 1, 1],
		'HF':    [1, 1, 1],
		'HD':    [1, 1, 1],
		'CPPLF': [0, 1, 0],
		'CPP':   [0, 1, 0],
		'CPPHF': [0, 1, 0],
		'CPPHD': [0, 1, 0]
	}
	hatchs = {
		'LF':    '//',
		'R':     None,
		'HF':    '\\\\',
		'HD':    '++',
		'CPPLF': '//',
		'CPP':   None,
		'CPPHF': '\\\\',
		'CPPHD': '++'
	}
	lines = {
		'LF':    '--ks',
		'R':     '-ks',
		'HF':    '-.ks',
		'HD':    ':ks',
		'CPPLF': '--k^',
		'CPP':   '-k^',
		'CPPHF': '-.k^',
		'CPPHD': ':k^'
	}
	
	############################################################################
	# Bar plot
	plt.figure(0)
	handles = list()
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
	plt.plot(left_position-3*bar_width, LCOE_aeo17, '-r.')
	# ax.legend([h[0] for h in handles], 
	# 			techs, 
	# 			bbox_to_anchor = (1.01, 0.5), 
	# 			loc='center left')
	plt.xlabel('Periods')
	plt.ylabel('LCOE ($/MWh)')
	plt.xticks(left_position-3*bar_width, [str(p) for p in range(2015, 2055, 5)])
	ax.legend( handles, 
				['LF var',    'LF fixed',    'LF invest', 
				 'R var',     'R fixed',     'R invest',
				 'HF var',    'HF fixed',    'HF invest',
				 'HD var',    'HD fixed',    'HD invest',
				 'CPPLF var', 'CPPLF fixed', 'CPPLF invest',
				 'CPP var',   'CPP fixed',   'CPP invest',
				 'CPPHF var', 'CPPHF fixed', 'CPPHF invest',
				 'CPPHD var', 'CPPHD fixed', 'CPPHD invest',],
				bbox_to_anchor=(0.5, 1.2),
				ncol=8,
				loc=9)
	ax.yaxis.grid(True)

	############################################################################
	# Line plot
	plt.figure(1)
	handles = list()
	ax = plt.subplot(111)
	for s in scenarios:
		LCOE = (
			np.array(LCOI[s]) + np.array(LCOF[s]) + np.array(LCOV[s])
			)
		h = plt.plot(range(2015, 2055, 5), LCOE, lines[s], label=s)
		handles.append(h)
	h = plt.plot(range(2015, 2055, 5), LCOE_aeo17, '-r.', label='AEO2017')
	handles.append(h)

	plt.xlabel('Period')
	plt.ylabel('LCOE ($/MWh)')
	plt.legend(loc='lower right')
	ax.yaxis.grid(True)
	ax.set_xlim([2014, 2051])

	plt.show()

def LC_calculate_db(db, selected_scenario):

	def return_LoanAnnualize ( t, v ):
		# Return loan annualization factor
		if (t, v) in DiscountRate:
			dr = DiscountRate[t, v]
		else:
			dr = GDR

		if t in LifetimeLoanTech:
			lln = LifetimeLoanTech[t]
		else:
			lln = 10
		annualized_rate = ( dr / (1.0 - (1.0 + dr)**(-lln) ))
		return annualized_rate

	def update_scenario_cost( CostVariable, selected_scenario):
		if 'L' in selected_scenario:
			for key, value in cost_var_coal['L'].iteritems():
				p = key[0]
				CostVariable[p, 'IMPELCCOAB',  2015]  = value
				CostVariable[p, 'IMPELCNGAEA', 2015] = cost_var_ng['L'][key]
				CostVariable[p, 'IMPELCDSLEA', 2015] = cost_var_oil['L'][key]
		if 'H' in selected_scenario:
			for key, value in cost_var_coal['H'].iteritems():
				p = key[0]
				CostVariable[p, 'IMPELCCOAB',  2015]  = value
				CostVariable[p, 'IMPELCNGAEA', 2015] = cost_var_ng['H'][key]
				CostVariable[p, 'IMPELCDSLEA', 2015] = cost_var_oil['H'][key]

	con = sqlite3.connect(db)
	cur = con.cursor()

	qry = 'SELECT * FROM time_periods'
	cur.execute(qry)
	db_time_periods = cur.fetchall()

	qry = 'SELECT * FROM Output_V_Capacity WHERE scenario IS "' + selected_scenario + '"'
	cur.execute(qry)
	db_V_Capacity = cur.fetchall()

	qry = "SELECT * FROM Output_VFlow_Out WHERE scenario IS '" + selected_scenario + "'"
	cur.execute(qry)
	db_VFlow_Out = cur.fetchall()

	qry = "SELECT * FROM CostInvest"
	cur.execute(qry)
	db_CostInvest = cur.fetchall()

	qry = "SELECT * FROM CostFixed"
	cur.execute(qry)
	db_CostFixed = cur.fetchall()

	qry = "SELECT * FROM CostVariable"
	cur.execute(qry)
	db_CostVariable = cur.fetchall()

	qry = "SELECT * FROM LifetimeTech"
	cur.execute(qry)
	db_LifetimeTech = cur.fetchall()

	qry = 'SELECT * FROM LifetimeLoanTech'
	cur.execute(qry)
	db_LifetimeLoanTech = cur.fetchall()

	qry = 'SELECT * FROM GlobalDiscountRate'
	cur.execute(qry)
	db_GDR = cur.fetchall()

	qry = 'SELECT * FROM DiscountRate'
	cur.execute(qry)
	db_DiscountRate = cur.fetchall()

	qry = 'SELECT * FROM Output_Objective WHERE scenario IS "' + selected_scenario + '"'
	cur.execute(qry)
	db_Objective = cur.fetchall()

	con.close()

	V_Capacity = dict()
	for row in db_V_Capacity:
		scenario, sector, t, v, value = row
		if scenario == selected_scenario:
			V_Capacity[t, v] = value

	CostInvest = dict()
	for row in db_CostInvest:
		t, v, value, unit, notes = row
		CostInvest[t, v] = value

	CostFixed = dict()
	for row in db_CostFixed:
		p, t, v, value, unit, notes = row
		CostFixed[p, t, v] = value

	CostVariable = dict()
	for row in db_CostVariable:
		p, t, v, value, unit, notes = row
		CostVariable[p, t, v] = value
		# Because fuel costs in the L and H scenarios are different from R, we
		# have to update it manually.
	update_scenario_cost(CostVariable, selected_scenario)

	LifetimeTech = dict()
	for row in db_LifetimeTech:
		t, value, notes = row
		LifetimeTech[t] = value

	LifetimeLoanTech = dict()
	for row in db_LifetimeLoanTech:
		t, value, notes = row
		LifetimeLoanTech[t] = value

	GDR = db_GDR[0][0]
	DiscountRate = dict()
	for row in db_DiscountRate:
		t, v, value, notes = row
		DiscountRate[t, v] = value

	periods = set()
	for row in db_time_periods:
		p, label = row
		if label =='f': # Future periods only.
			periods.add(p)
	periods = list(periods)
	periods.sort()
	f_periods = deepcopy(periods) # Future periods including the last period
	periods.pop() # remove the last one since it is not in the optimized periods

	V_ActivityByPeriodAndProcess = dict()
	TAE  = [0]*len(periods)
	for row in db_VFlow_Out:
		scenario, sector, p, s, d, i, t, v, o, value = row
		# Scenario, sector, period, season, tod, input, tech, vintage, output, value
		if scenario == selected_scenario:
			if (p, t, v) in V_ActivityByPeriodAndProcess:
				V_ActivityByPeriodAndProcess[p, t, v] += value
			else:
				V_ActivityByPeriodAndProcess[p, t, v] = value
			if o == 'ELCDMD':
				pindex = periods.index(p) # period index
				TAE[pindex] += value/3.6 # Convert PJ into TWh

	# Note TAC_by_year is total annualized cost by year, not by period.
	# Active periods: 2015, 2020, 2025 ... 2050, 2055 (2055 is the stopping year)
	# Active years: 2015, 2016, 2017 ... 2050, 2051, 2052, 2053, 2054
	# Note that each period is represented by its first year, e.g., The period
	# represented by 2015 include five years: 2015, 2016, 2017, 2018, 2019
	years   = range(f_periods[0], f_periods[-1])
	f_years = deepcopy(years)
	f_years.append(f_periods[-1])
	TAC_by_year  = [0]*len(years)

	# Start LCOE calculating
	TAIC  = [0]*len(periods)
	TAFC  = [0]*len(periods)
	TAVC  = [0]*len(periods)

	for S_t, S_v in CostInvest:
		if (S_t, S_v) in V_Capacity:
			if S_t in LifetimeLoanTech:
				lln = int( LifetimeLoanTech[S_t] )
			else:
				lln = 10
			if S_t in LifetimeTech:
				lp = int( LifetimeTech[S_t] )
			else:
				lp = 30
			LoanAnnualize_factor = return_LoanAnnualize(S_t, S_v)
			LoanAnnualize = (
				CostInvest[S_t, S_v]
				*LoanAnnualize_factor
				*V_Capacity[S_t, S_v]
			)
			LoanNPV = (
				LoanAnnualize * 
				(
					1 - (1 + GDR)**(-lln)
				) /
				GDR
			) # NPV of all loan payments, including those beyond end of model horizon
			LoanAmortize = (
				LoanNPV * 
				GDR /
				(
					1 - (1 + GDR)**(-lp)
				)
			) # Amortized loan NPV over process lifetime (lp)

			vindex = periods.index(S_v)

			for i in range(vindex, len(periods)):
				# The loan payment last from the 0th year to the (n-1)th year 
				# where n is the lifetime of loan, which defaults to 10 years.
				if periods[i] - S_v < lp:
					TAIC[i] += LoanAmortize
			
			# Calculate all years LCOE, i.e., 2015, 2016 ... 2054
			this_payment = LoanAmortize
			for i in range(0, lp):
				this_year = S_v + i
				if this_year in years:
					TAC_by_year[years.index(this_year)] += this_payment

	for S_p, S_t, S_v in CostFixed:
		if (S_t, S_v) in V_Capacity:
			pindex = periods.index(S_p)
			TAFC[pindex] += (
				CostFixed[S_p, S_t, S_v]
				*V_Capacity[S_t, S_v]
			)

			# Calculate all years LCOE, i.e., 2015, 2016 ... 2054
			S_p_next = f_periods[pindex + 1]
			for i in range(f_years.index(S_p), f_years.index(S_p_next)):
				if years[i] - S_v < LifetimeTech[S_t]:
					TAC_by_year[i] +=(
						CostFixed[S_p, S_t, S_v]
						*V_Capacity[S_t, S_v]
					)

	for S_p, S_t, S_v in CostVariable:
		if (S_p, S_t, S_v) in V_ActivityByPeriodAndProcess:
			pindex = periods.index(S_p)
			TAVC[pindex] += (
				CostVariable[S_p, S_t, S_v]
				*V_ActivityByPeriodAndProcess[S_p, S_t, S_v]
			)

			# Calculate all years LCOE, i.e., 2015, 2016 ... 2054
			S_p_next = f_periods[pindex + 1]
			for i in range(f_years.index(S_p), f_years.index(S_p_next)):
				if years[i] - S_v < LifetimeTech[S_t]:
					TAC_by_year[i] +=(
						CostVariable[S_p, S_t, S_v]
						*V_ActivityByPeriodAndProcess[S_p, S_t, S_v]
					)

	TAC = [
		TAIC[i] + TAFC[i] + TAVC[i] 
		for i in range( 0, len(periods) ) 
	]

	LCOE = [ TAC[i]/TAE[i] for i in range( 0, len(periods) ) ]

	LCOI = [ TAIC[i]/TAE[i] for i in range( 0, len(periods) ) ]
	LCOF = [ TAFC[i]/TAE[i] for i in range( 0, len(periods) ) ]
	LCOV = [ TAVC[i]/TAE[i] for i in range( 0, len(periods) ) ]

	# The following calculation gives the difference between the objective 
	# functions calculated from two different methods, one is what is used in 
	# the model, where discounted costs are summed over all processes, while the
	# second is discounting all annualized costs to present and summing over all
	# periods.
	# If the LCOE calculation is correct, then the two methods should give 
	# identical solutions.
	for row in db_Objective:
		scenario, name, value = row
		if scenario == selected_scenario:
			NPV_1 = value
			break
	NPV_2 = sum( [ TAC_by_year[i]/(1 + GDR)**i for i in range(0, len(TAC_by_year)) ] )
	deltaNPV = NPV_2 - NPV_1
	print 'Total NPV error is {:3.3f} %'.format(100*deltaNPV/NPV_1)
	return LCOI, LCOF, LCOV

def LCOE_target():
	# Test LCOE calculation for a specific technology through all periods
	Ttarget = 'ENGAACC'
	Ctarget = 'ELC'
	# Ttarget = 'IMPELCNGAEA'
	# Ctarget = 'ELCNGAEA'
	TAICtarget = [0]*len(periods)
	TAFCtarget = [0]*len(periods)
	TAVCtarget = [0]*len(periods)
	TAEtarget  = [0]*len(periods)
	for S_t, S_v in CostInvest:
		if (S_t, S_v) in V_Capacity and S_t == Ttarget:
			LoanAnnualize = return_LoanAnnualize(S_t, S_v)
			pindex = periods.index(S_v)
			
			if S_t in LifetimeLoanTech:
				lln = LifetimeLoanTech[S_t]
			else:
				lln = 10

			for i in range(pindex, len(periods)):
				# The loan payment last from the 0th year to the (n-1)th year 
				# where n is the lifetime of loan, which defaults to 10 years.
				if periods[i] - S_v < lln:
					TAICtarget[i] += (
						CostInvest[S_t, S_v]
						*LoanAnnualize
						*V_Capacity[S_t, S_v]
					)

	for S_p, S_t, S_v in CostFixed:
		if (S_t, S_v) in V_Capacity and S_t == Ttarget:
			pindex = periods.index(S_p)
			TAFCtarget[pindex] += (
				CostFixed[S_p, S_t, S_v]
				*V_Capacity[S_t, S_v]
			)

	for S_p, S_t, S_v in CostVariable:
		if (S_p, S_t, S_v) in V_ActivityByPeriodAndProcess and S_t == Ttarget:
			pindex = periods.index(S_p)
			TAVCtarget[pindex] += (
				CostVariable[S_p, S_t, S_v]
				*V_ActivityByPeriodAndProcess[S_p, S_t, S_v]
			)

	for row in db_VFlow_Out:
		scenario, sector, p, s, d, i, t, v, o, value = row
		# Scenario, sector, period, season, tod, input, tech, vintage, output, value
		if o == Ctarget and t == Ttarget:
			pindex = periods.index(p) # period index
			TAEtarget[pindex] += value

	TACtarget = [
		TAICtarget[i] + TAFCtarget[i] + TAVCtarget[i] 
		for i in range( 0, len(periods) ) 
	]
	LCOEtarget = [ TACtarget[i]/TAEtarget[i] for i in range( 0, len(periods) ) ]

def LC_db(db):
	# This script calculates the levelized cost of a given commodity produced
	# from a specific process .

	def return_LoanAnnualize ( t, v ):		
		if (t, v) in DiscountRate:
			dr = DiscountRate[t, v]
		else:
			dr = GDR

		if t in LifetimeLoanTech:
			lln = LifetimeLoanTech[t]
		else:
			lln = 10
		annualized_rate = ( dr / (1.0 - (1.0 + dr)**(-lln) ))
		return annualized_rate

	def return_SalvageRate( t, v ):
		P_0  = periods[0]
		if t in LifetimeTech:
			n = LifetimeTech[t]
		else:
			n = 30
		L    = f_periods[-1] - f_periods[0]

		if P_0 + L - v >= n:
			return 0
		else:
			return 1 - (P_0 + L - v)/n # Linear depreciation

	def return_CostInvest( t, v ):
		if (t, v) in CostInvest:
			return CostInvest[t, v]
		else:
			return 0

	def return_CostFixed(p, t, v):
		if (p, t, v) in CostFixed:
			return CostFixed[p, t, v]
		else:
			return 0

	def return_CostVariable(p, t, v):
		if (p, t, v) in CostVariable:
			return CostVariable[p, t, v]
		else:
			return 0

	con = sqlite3.connect(db)
	cur = con.cursor()

	qry = 'SELECT * FROM time_periods'
	cur.execute(qry)
	db_time_periods = cur.fetchall()

	qry = 'SELECT * FROM time_season'
	cur.execute(qry)
	db_time_season = cur.fetchall()

	qry = 'SELECT * FROM time_of_day'
	cur.execute(qry)
	db_time_of_day = cur.fetchall()

	qry = 'SELECT * FROM Efficiency'
	cur.execute(qry)
	db_Efficiency = cur.fetchall()

	qry = 'SELECT * FROM Output_V_Capacity'
	cur.execute(qry)
	db_V_Capacity = cur.fetchall()

	qry = "SELECT * FROM Output_VFlow_In"
	cur.execute(qry)
	db_VFlow_In = cur.fetchall()

	qry = "SELECT * FROM Output_VFlow_Out"
	cur.execute(qry)
	db_VFlow_Out = cur.fetchall()

	qry = "SELECT * FROM CostInvest"
	cur.execute(qry)
	db_CostInvest = cur.fetchall()

	qry = "SELECT * FROM CostFixed"
	cur.execute(qry)
	db_CostFixed = cur.fetchall()

	qry = "SELECT * FROM CostVariable"
	cur.execute(qry)
	db_CostVariable = cur.fetchall()

	qry = "SELECT * FROM LifetimeTech"
	cur.execute(qry)
	db_LifetimeTech = cur.fetchall()

	qry = 'SELECT * FROM LifetimeLoanTech'
	cur.execute(qry)
	db_LifetimeLoanTech = cur.fetchall()

	qry = 'SELECT * FROM GlobalDiscountRate'
	cur.execute(qry)
	db_GDR = cur.fetchall()

	qry = 'SELECT * FROM DiscountRate'
	cur.execute(qry)
	db_DiscountRate = cur.fetchall()

	qry = 'SELECT * FROM Output_Objective'
	cur.execute(qry)
	db_Objective = cur.fetchall()

	con.close()

	time_season = set( db_time_season )
	time_of_day = set( db_time_of_day )

	V_Capacity = dict()
	for row in db_V_Capacity:
		scenario, sector, t, v, value = row
		V_Capacity[t, v] = value

	V_Flow_In = dict()
	for row in db_VFlow_In:
		scenario, sector, p, s, d, i, t, v, o, value = row
		V_Flow_In[p, s, d, i, t, v, o] = value

	V_Flow_Out = dict()
	for row in db_VFlow_Out:
		scenario, sector, p, s, d, i, t, v, o, value = row
		V_Flow_Out[p, s, d, i, t, v, o] = value

	CostInvest = dict()
	for row in db_CostInvest:
		t, v, value, unit, notes = row
		CostInvest[t, v] = value

	CostFixed = dict()
	for row in db_CostFixed:
		p, t, v, value, unit, notes = row
		CostFixed[p, t, v] = value

	CostVariable = dict()
	for row in db_CostVariable:
		p, t, v, value, unit, notes = row
		CostVariable[p, t, v] = value

	LifetimeTech = dict()
	for row in db_LifetimeTech:
		t, value, notes = row
		LifetimeTech[t] = value

	LifetimeLoanTech = dict()
	for row in db_LifetimeLoanTech:
		t, value, notes = row
		LifetimeLoanTech[t] = value

	GDR = db_GDR[0][0]
	DiscountRate = dict()
	for row in db_DiscountRate:
		t, v, value, notes = row
		DiscountRate[t, v] = value

	periods = set()
	for row in db_time_periods:
		p, label = row
		if label =='f': # Future periods only.
			periods.add(p)
	periods = list(periods)
	periods.sort()
	f_periods = deepcopy(periods) # Future periods including the last period
	periods.pop() # remove the last one since it is not in the optimized periods

	V_ActivityByPeriodAndProcess = dict()
	for row in db_VFlow_Out:
		scenario, sector, p, s, d, i, t, v, o, value = row
		# Scenario, sector, period, season, tod, input, tech, vintage, output, value
		if (p, t, v) in V_ActivityByPeriodAndProcess:
			V_ActivityByPeriodAndProcess[p, t, v] += value
		else:
			V_ActivityByPeriodAndProcess[p, t, v] = value

	#---------------------------------------------------------------------------
	ptv_valid = set()
	VOP = dict() # V_FlowOut by periods
	VIP = dict() # V_FlowIn by periods
	g_ProcessInputsByOutput = dict()
	g_commodityUStreamProcess = dict()
	flag_loan = dict()
	for p, s, d, i, t, v, o in V_Flow_In:
		ptv_valid.add( (p, t, v) )
		if (p, i, t, v, o) in VIP:
			VIP[p, i, t, v, o] += V_Flow_In[p, s, d, i, t, v, o]
			VOP[p, i, t, v, o] += V_Flow_Out[p, s, d, i, t, v, o]
		else:
			VIP[p, i, t, v, o] = V_Flow_In[p, s, d, i, t, v, o]
			VOP[p, i, t, v, o] = V_Flow_Out[p, s, d, i, t, v, o]

		if (p, t, v, o) in g_ProcessInputsByOutput:
			g_ProcessInputsByOutput[p, t, v, o].add(i)
		else:
			g_ProcessInputsByOutput[p, t, v, o] = {i}

		if (p, o) in g_commodityUStreamProcess:
			g_commodityUStreamProcess[p, o].add( (t, v) )
		else:
			g_commodityUStreamProcess[p, o] = { (t, v) }

		if (p, t, v) not in flag_loan:
			if t not in LifetimeLoanTech:
				flag_loan[p, t, v] = 0
			elif LifetimeLoanTech[t] + v >= p:
				# Loan is not paid off
				flag_loan[p, t, v] = 1
			else:
				# Loan is paid off
				flag_loan[p, t, v] = 0
	
	if ptv_valid - set(V_ActivityByPeriodAndProcess.keys()):
		# These two sets should be identical, otherwise something is wrong
		sys.stderr.write('\n Sets not identical: ptv_valid and ActivityByPeriodAndProcess\n')

	LC = dict() # LC indexed by p, t, v, o
	LC_bar = dict() # Weighted average LC indexed by p, o

	def return_LC(p, t, v, o):
		if (p, t, v, o) in LC:
			return LC[p, t, v, o]
		I = g_ProcessInputsByOutput[p, t, v, o]
		sigmaVOP = sum(VOP[p, i, t, v, o] for i in I)
		denominator = sigmaVOP
		numerator = (
			sum( return_LCbar(p, i) * VIP[p, i, t, v, o] for i in I) 
			+ (
				flag_loan[p, t, v]*return_LoanAnnualize(t, v)*return_CostInvest(t, v) +
				return_CostFixed(p, t, v)
				) * V_Capacity[t, v] * sigmaVOP / V_ActivityByPeriodAndProcess[p, t, v]
			+ return_CostVariable(p, t, v)*sigmaVOP
			)
		LC[p, t, v, o] = numerator/denominator
		return numerator/denominator

	def return_LCbar(p, c):
		if (p, c) in LC_bar:
			return LC_bar[p, c]
		if c == 'ethos':
			return 0
		I = g_commodityUStreamProcess[p, c]
		numerator = sum(
			return_LC(p, t, v, c)* 
			sum(
				VOP[p, i, t, v, c]
				for i in g_ProcessInputsByOutput[p, t, v, c]
				)
			for (t, v) in I
			)
		denominator = sum(
			sum(
				VOP[p, i, t, v, c]
				for i in g_ProcessInputsByOutput[p, t, v, c]
				)
			for (t, v) in I
			)
		LC_bar[p, c] = numerator/denominator
		return numerator/denominator



if __name__ == "__main__":
	if len(sys.argv) == 1: 
		dbs = OrderedDict( [
			('LF',    '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/results/LF/NCreference.LF.db'),
			('R',     '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/results/R/NCreference.R.db'),
			('HF',    '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/results/HF/NCreference.HF.db'),
			('HD',    '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/results/HD/NCreference.HD.db'),
			('CPPLF', '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/results/CPPLF/NCreference.CPPLF.db'),
			('CPP',   '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/results/CPP/NCreference.CPP.db'),
			('CPPHF', '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/results/CPPHF/NCreference.CPPHF.db'),
			('CPPHD', '/afs/unity.ncsu.edu/users/b/bli6/TEMOA_NC/sql20170417/results/CPPHD/NCreference.CPPHD.db'),
		] )

		# dbs = {
		# 	'LF':    'C:\\Users\\bli\\OneDrive\\Tmp_TEMOA_paper\\Results20170417\\LF\\NCreference.LF.db',
		# 	'R':     'C:\\Users\\bli\\OneDrive\\Tmp_TEMOA_paper\\Results20170417\\R\\NCreference.R.db',
		# 	'HF':    'C:\\Users\\bli\\OneDrive\\Tmp_TEMOA_paper\\Results20170417\\HF\\NCreference.HF.db',
		# 	'HD':    'C:\\Users\\bli\\OneDrive\\Tmp_TEMOA_paper\\Results20170417\\HD\\NCreference.HD.db',
		# 	'CPPLF': 'C:\\Users\\bli\\OneDrive\\Tmp_TEMOA_paper\\Results20170417\\CPPLF\\NCreference.CPPLF.db',
		# 	'CPP':   'C:\\Users\\bli\\OneDrive\\Tmp_TEMOA_paper\\Results20170417\\CPP\\NCreference.CPP.db',
		# 	'CPPHF': 'C:\\Users\\bli\\OneDrive\\Tmp_TEMOA_paper\\Results20170417\\CPPHF\\NCreference.CPPHF.db',
		# 	'CPPHD': 'C:\\Users\\bli\\OneDrive\\Tmp_TEMOA_paper\\Results20170417\\CPPHD\\NCreference.CPPHD.db',
		# }

		LCOI = OrderedDict()
		LCOF = OrderedDict()
		LCOV = OrderedDict()
		for s in dbs:
			print s
			(this_LCOI, this_LCOF, this_LCOV) = LC_calculate_db(dbs[s])
			LCOI[s] = this_LCOI
			LCOF[s] = this_LCOF
			LCOV[s] = this_LCOV
		plot_LCOE(LCOI, LCOF, LCOV)
	else:
		db = sys.argv[1]
		s  = sys.argv[2]
		(LCOI, LCOF, LCOV) = LC_calculate_db(db, s)
		LCOE = [LCOI[i] + LCOF[i] + LCOV[i] for i in range(0, len(LCOI))]
		for y, lcoe in zip(range(2015, 2055, 5), LCOE):
			print '{}   {:4.2f}   {}'.format(y, lcoe, '$/MWh')

		# LC_db(db)

