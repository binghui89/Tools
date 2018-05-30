# https://stackoverflow.com/questions/37737538/merge-matplotlib-subplots-with-shared-x-axis

from openpyxl import Workbook, load_workbook
from matplotlib import ticker, gridspec, pyplot as plt
import sys, platform
from IPython import embed as IP

if platform.system() == 'Linux':
    sys.path.append("/afs/unity.ncsu.edu/users/b/bli6/temoa/temoa_model")
    sys.path.append('/opt/ibm/ILOG/CPLEX_Studio1263/cplex/python/2.6/x86-64_linux')
elif platform.system() == 'Windows':
    sys.path.append('C:\\Users\\bli\\GitHub\\Temoa\\temoa_model')
    sys.path.append('C:\\Program Files\\IBM\\ILOG\\CPLEX_Studio128\\cplex\\python\\2.7\\x64_win64')
    sys.path.append('C:\\Users\\bxl180002\\git\\Tools')
elif platform.system() == 'Darwin':
    sys.path.append('/Users/bli/git/temoa/temoa_model')
    sys.path.append('/Users/bli/Applications/IBM/ILOG/CPLEX_Studio1263/cplex/python/2.7/x86-64_osx')
    sys.path.append('/Users/bli/git/tools')
else:
    print 'Unrecognized system! Exiting...'
    sys.exit(0)

from plot_result import plot_NCdemand_all, plot_emis_all, TemoaNCResult
from mapping import *

def plot_6bars(ax, periods, colortypes, scenarios, values):
    handles = list()
    w     = 0.1*5
    gap   = 0.04*5 # w + gap = 5*(1/7)
    ymax  = 0 # To determine ylimits

    for k in range(0, len(scenarios)):
        b = [0]*len(periods) # Bottom height
        s = scenarios[k]
        offset_bar0 = -(3*w + 2.5*gap) + k*(w + gap)
        x = [i + offset_bar0 for i in periods]
        for t in colortypes:
            y = values[t][s]
            if hatch_map[ category_map[t] ]:
                h = ax.bar(
                    x,
                    y,
                    width = w,
                    bottom = b,
                    color = color_map[ category_map[t] ], 
                    hatch = hatch_map[ category_map[t] ],
                )
            else:
                h = ax.bar(
                    x, 
                    y, 
                    width = w,
                    bottom = b,
                    color = color_map[ category_map[t] ], 
                    edgecolor = edge_map[ category_map[t] ],
                )
            b = [b[i] + values[t][s][i] for i in range(0, len(b))]
            handles.append(h)
        if ymax < max(b):
            ymax = max(b)
    ymax = int(ymax)
    if ymax > 100:
        ymax = ymax - ymax%50 + 65
    elif ymax < 100:
        ymax = ymax - ymax%10 + 15
    ax.set_ylim(0, ymax)

    return handles

def plot_breakeven(ax, years, scenarios, bic, ic, i_subplot):
    # bic is a dictionary, ic is a list of the raw investment costs
    # ic = [x, x, ..., x], the length of which equals to the length of years
    # bic[scenario] = [x, x, x... x] where the length equals to the number 
    # of optimized periods.
    sen_color_map = {
        'IC':    [0.9, 0.9, 0.9],
        'L':     'black',
        'R':     'black',
        'H':     'black',
        'CPP-L': 'green',
        'CPP-R': 'green',
        'CPP-H': 'green'
    }

    sen_lstyle_map = {
        'IC':    None,
        'L':     '--',
        'R':     '-',
        'H':     'dotted',
        'CPP-L': '--',
        'CPP-R': '-',
        'CPP-H': 'dotted'
    }

    sen_marker_map = {
        'IC':    None,
        'L':     's',
        'R':     's',
        'H':     's',
        'CPP-L': 's',
        'CPP-R': 's',
        'CPP-H': 's'
    }

    handles = list()

    ymax = max(ic)
    for s in scenarios:
        h, = ax.plot(years, bic[s], 
            color = sen_color_map[s],
            marker = sen_marker_map[s],
            linestyle = sen_lstyle_map[s],
            markersize = 2,
            linewidth = 1,
        )
        handles.append(h)

    h = ax.fill_between(years, 0, ic, 
        facecolor = sen_color_map['IC']
        )
    handles.append(h)

    ax.set_xlim( ( years[0]-2, years[-1]+2 ) )
    ymax = int(ymax)
    ymax = ymax - ymax%100 + 100
    ax.set_ylim(0, ymax)
    ax.set_xticks(years)
    ax.annotate(
        i_subplot,
        xy=(0.1, 0.9), xycoords='axes fraction',
        xytext=(.1, .9), textcoords='axes fraction',
            )
    return handles

def plot_marginalCO2():
    fig = plt.figure(figsize=(5, 3), dpi=80, facecolor='w', edgecolor='k')
    ax  = plt.subplot(111)
    w = 1
    gap = 2
    years = range(2025, 2055, 5)
    values = dict()
    values['CPP-L']         = [0.00, 4.68, 0.00, 51.12, 53.86, 28.89]
    values['CPP-R']         = [17.88, 33.40, 23.92, 50.79, 41.97, 27.05]
    values['CPP-H']         = [86.17, 81.90, 57.81, 58.51, 47.70, 27.61]
    values['SCC in 2015 $'] = [16, 18, 21, 24, 26, 30]

    scenarios = ['CPP-L', 'CPP-R', 'CPP-H']
    fcolor    = ['k', [0.8, 0.8, 0.8],  'w']
    ecolor    = ['k', [0.8, 0.8, 0.8],  'k']
    handles   = list()
    for i in range(0, len(scenarios)):
        s = scenarios[i]
        offset = (i - 1)*w
        h = ax.bar(
            [j + offset for j in years],
            values[s],
            width=w,
            color=fcolor[i],
            edgecolor=ecolor[i],
        )
        handles.append(h)
    h, = ax.plot(
        years,
        values['SCC in 2015 $'],
        linewidth=1,
        color='k',
        marker='s',
        markersize=2,
    )
    handles.append(h)
    plt.legend(
        handles,
        scenarios+['SCC in 2015 $'],
        loc='upper center', 
        ncol=4, 
        bbox_to_anchor=(0.5, 1.15),
        edgecolor=None
    )
    ax.set_ylabel(r'\$/tonne $\mathregular{CO}_2$')
    return fig

wb1 = load_workbook('Graph.EST.xlsx', data_only=True)
# Size of each individual subplot, in inch
hfig = 4
wfig = 10

def cap_and_act(db):
    fig = plt.figure(figsize=(wfig, hfig*2), dpi=80, facecolor='w', edgecolor='k')
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1]) 
    ax = list()
    attributes = ['activities', 'capacities']
    for a in attributes:
        ax.append( plt.subplot(gs[attributes.index(a)]) )
        scenarios  = ['L', 'R', 'H', 'CPPL', 'CPP', 'CPPH']
        values     = dict()
        periods    = None
        for s in scenarios:
            instance = TemoaNCResult(db, s)
            periods  = instance.periods
            for t in instance.techs:
                t_full = [k for k, v in category_map.items() if v == t][0]
                if t_full not in values:
                    # Inverse lookup, see https://stackoverflow.com/q/2568673
                    values[t_full] = dict()
                values[t_full][s] = getattr(instance, a)[t]
                if a is 'activities':
                    values[t_full][s] = [ i/3.6 for i in values[t_full][s] ]
        technology = values.keys()
        for t in technology:
            for s in scenarios:
                if s not in values[t]:
                    values[t][s] = [0 for i in periods]
        if 'Geothermal' in technology:
            technology.remove('Geothermal')
        if 'EE' in technology:
            technology.remove('EE')
        for t in [
            'Nuclear',      'Hydro',     'Oil',
            'Pumped hydro', 'Coal',      'Natural gas',
            'Wind',         'Solar PV',  'Bioenergy',
        ]:
            if t in technology:
                technology.remove(t)
                technology.append(t)
        handles = plot_6bars(ax[attributes.index(a)], periods, technology, scenarios, values)

    ax[1].set_xlabel('Periods')
    ax[1].set_ylabel('GW')
    ax[0].set_ylabel('TWh')
    plt.setp(ax[0].get_xticklabels(), visible=False) # shared x axis
    yticks = ax[1].yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False) # remove last tick label for the second subplot
    plt.subplots_adjust(hspace=.0) # remove vertical gap between subplots
    # ax[1].legend(handles, technology, loc='upper left')
    fig.legend(
        handles, technology, 
        loc='upper center', 
        ncol=5, 
        bbox_to_anchor=(0.5, 0.065),
        edgecolor=None
    )
    # https://stackoverflow.com/q/18619880/4626354
    plt.subplots_adjust(left=0.06, right=0.94, top=0.99, wspace=0, hspace=0)
    return fig

def emissions_new(db):
    emissions = ['so2_ELC', 'nox_ELC', 'co2_ELC']
    scenarios = ['L', 'R', 'H', 'CPPL', 'CPP', 'CPPH']

    fig = plt.figure(figsize=(wfig, hfig*3), dpi=80, facecolor='w', edgecolor='k')
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1]) 
    ax = list()
    for ie in range(0, len(emissions)):
        e = emissions[ie]
        ax.append( plt.subplot(gs[ie]) )

        values     = dict()
        periods    = None
        for s in scenarios:
            instance = TemoaNCResult(db, s)
            periods  = instance.periods
            for t in instance.emissions1[e]:
                t_full = [k for k, v in category_map.items() if v == t][0]
                if t_full not in values:
                    values[t_full] = dict()
                values[t_full][s] = instance.emissions1[e][t]
            if 'Emission reduction' not in values:
                values['Emission reduction'] = dict()
            values['Emission reduction'][s] = instance.emis_redct[e]

        technology = values.keys()
        if 'other' in technology:
            technology.remove('other')
        if 'Total raw' in technology:
            technology.remove('Total raw')
        for t in ['Oil', 'Bioenergy', 'Natural gas', 'Coal', 'Emission reduction']:
            if t in technology:
                technology.remove(t)
                technology.append(t)
        handles = plot_6bars(ax[ie], periods, technology, scenarios, values)

    ax[-1].set_xlabel('Periods')
    ax[0].set_ylabel('Emission (kilotonnes)')
    ax[1].set_ylabel('Emission (kilotonnes)')
    ax[2].set_ylabel('Emission (megatonnes)')
    plt.setp(ax[0].get_xticklabels(), visible=False) # shared x axis
    plt.setp(ax[1].get_xticklabels(), visible=False) # shared x axis
    yticks = ax[2].yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False) # remove last tick label for the last subplot
    yticks = ax[1].yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False) # remove last tick label for the second last subplot
    plt.subplots_adjust(hspace=.0) # remove vertical gap between subplots
    # ax[1].legend(handles, technology, loc='upper left')
    fig.legend(
        handles, technology, 
        loc='upper center', 
        ncol=5, 
        bbox_to_anchor=(0.5, 0.065),
        edgecolor=None
    )
    # https://stackoverflow.com/q/18619880/4626354
    plt.subplots_adjust(left=0.06, right=0.94, top=0.99, wspace=0, hspace=0)
    return fig

def break_even():
    def return_figure(tech):
        fig  = plt.figure(figsize=(wfig*len(tabs), hfig*len(tech)), dpi=80, facecolor='w', edgecolor='k')
        gs = gridspec.GridSpec(len(tech), len(tabs), wspace=0.05, hspace=0.0) 
        ax = list()
        handles = None
        counter = 0
        for it in range(0, len(tech)):
            t = tech[it]
            for itab in range(0, len(tabs)):
                tab = tabs[itab]
                ax.append( plt.subplot(gs[it*2+itab]) )
                scenarios = ['L', 'R', 'H', 'CPP-L', 'CPP-R', 'CPP-H']
                periods   = range(2015, 2055, 5)
                ws1 = wb1[tab]
                ic  = list()
                bic = dict()
                for j in range(0, len(periods)):
                    v = ws1.cell(row=nr0[t]+j, column=nc0[t]+6).value
                    ic.append(v)

                for i in range(0, len(scenarios)):
                    nc = nc0[t] + i
                    s  = scenarios[i]
                    bic[s] = list()
                    for j in range(0, len(periods)):
                        nr = nr0[t] + j
                        v = ws1.cell(row=nr, column=nc).value
                        bic[s].append(v)
                handles = plot_breakeven(ax[it*2+itab], periods, scenarios, bic, ic, '('+chr(ord('a') + counter)+')')
                counter += 1
        
        for i in range(0, len(ax)):
            if i%2 == 1:
                plt.setp(ax[i].get_yticklabels(), visible=False)
            if i%2 == 0:
                ax[i].set_ylabel('Capital cost ($/MW)')
            if i<len(ax)-2:
                plt.setp(ax[i].get_xticklabels(), visible=False)
            else:
                for tick in ax[i].get_xticklabels():
                    tick.set_rotation(90)

        fig.legend(
            handles, scenarios+['Capex'], 
            loc='upper center', 
            ncol=7, 
            bbox_to_anchor=(0.5, 0.045),
            edgecolor=None
        )
        # https://stackoverflow.com/q/18619880/4626354
        plt.subplots_adjust(left=0.10, right=0.9, top=0.99, wspace=0, hspace=0)
        return fig

    hfig = 2.5
    wfig = 4
    tabs = ['SenAnalysis RPS (CPLEX API)', 'SenAnalysis no RPS (CPLEX API)']
    nr0 = {
        'Solar PV':      1451,
        'Onshore wind':  1461,
        'Biomass IGCC':  1471,
        'Nuclear':       1481,
        'Coal IGCC':     1491, 
        'NG IGCC':       1501,
        'Offshore wind': 1511,
    }
    nc0 = {
        'Solar PV':      3,
        'Onshore wind':  3,
        'Biomass IGCC':  3,
        'Nuclear':       3,
        'Coal IGCC':     3, 
        'NG IGCC':       3,
        'Offshore wind': 3,
    }

    figs = list()
    tech = ['Solar PV', 'Onshore wind', 'Biomass IGCC']
    fig = return_figure(tech)
    figs.append(fig)
    tech = ['Nuclear','Coal IGCC', 'NG IGCC','Offshore wind']
    fig = return_figure(tech)
    figs.append(fig)
    return figs

def CF_solar_wind_load():
    tech = ['Onshore wind', 'Offshore wind', 'Solar PV, rooftop', 'Solar PV, utility']
    color_tech = {
        'Onshore wind':      'g',
        'Offshore wind':     'g',
        'Solar PV, rooftop': 'r',
        'Solar PV, utility': 'r',
        'Hourly load'      : 'k',
    }
    lstyle_tech = {
        'Onshore wind':      '--',
        'Offshore wind':     '-',
        'Solar PV, rooftop': '--',
        'Solar PV, utility': '-',
        'Hourly load'      : '-',

    }
    seg  = range(1, 97)
    wb = load_workbook(
        'LDC.xlsx',
        data_only=True,
    )
    nr0 = {
        'Onshore wind':       7,
        'Offshore wind':      7,
        'Solar PV, rooftop':  7,
        'Solar PV, utility':  7,
    }
    nc0 = {
        'Onshore wind':       7,
        'Offshore wind':      8,
        'Solar PV, rooftop':  10,
        'Solar PV, utility':  11,
    }
    ws = wb['CF_summary']
    values = dict()
    for j in range(0, len(tech)):
        t = tech[j]
        values[t] = list()
        for i in range(0, len(seg)):
            v = ws.cell(row=nr0[t]+i, column=nc0[t]).value
            values[t].append(v)

    ws = wb['By Seasons']
    nr0 = 83
    nc0 = 7
    values['Hourly load'] = list()
    for j in range(0, len(seg)):
        v = ws.cell(row=nr0, column=nc0+j).value
        values['Hourly load'].append(v)

    fig = plt.figure(figsize=(7, 4), dpi=80, facecolor='w', edgecolor='w')
    ax1 = plt.subplot(111)
    handles = list()
    for t in tech:
        h, = ax1.plot(
            seg,
            values[t],
            linewidth=1,
            color=color_tech[t],
            linestyle=lstyle_tech[t],
        )
        handles.append(h)
    xticks      = [1, 25, 49, 73]
    yticks      = [i*0.1 for i in range(0, 7)]
    vals        = ax1.get_yticks()
    ax1.set_yticklabels(['{:d}%'.format(int(x*100)) for x in vals])
    ax1.xaxis.set_major_locator(ticker.FixedLocator(xticks))
    ax1.set_xlabel('Time Slice')
    ax1.set_ylabel('Availability factor')

    ax2 = ax1.twinx()
    h, = ax2.plot(
        seg,
        [i/1E3 for i in values['Hourly load']],
        linewidth=1,
        color=color_tech['Hourly load'],
        linestyle=lstyle_tech['Hourly load'],
    )
    handles.append(h)
    ax2.set_ylabel('Load (GW)')
    plt.legend(
        handles,
        tech + ['Hourly load'],
        loc = 'upper center',
        bbox_to_anchor=[0.5, 1.17],
        ncol=3,
    )
    return fig

if __name__ == "__main__":
    # # Chapter 3, Fig 3.3
    fig = plot_marginalCO2()
    # fig.savefig('fig3_3.pdf', format='pdf')

    # # Appendix Fig B.1, need to save manually
    # plot_NCdemand_all()

    # Appendix Fig B.2, run FERC714.py

    # # Appendix Fig B.4, capacity factor and load as a function of time slices
    # fig = CF_solar_wind_load()
    # fig.savefig('figB_4.pdf', format='pdf')

    # # Appendix Fig B.5, SO2 and NOX emissions,  need to save manually
    # # Appendix Fig B.9 and Chapter 4 Fig 4.5, need to change code, need to save manually
    # plot_emis_all()

    # # Appendix Fig B.11, without NG supply limits
    # tabs = ['Act Analysis noNGcap (6 bars)', 'Cap Analysis noNGcap (6 bars)']
    # fig = cap_and_act(tabs)
    # fig.savefig('figB_11.pdf', format='pdf')

    # # Appendix Fig B.12, without NG supply limits
    # fig = emissions('Emissions noNGcap (6 bars)')
    # fig.savefig('figB_12.pdf', format='pdf')

    figs = break_even()
    # figs[0].savefig('fig3_4.pdf', format='pdf')
    # figs[1].savefig('figB_10.pdf', format='pdf')

    db = 'C:\\Users\\bxl180002\\OneDrive\\Tmp_TEMOA_paper\\Results201803\\NCreference.db'
    # Chapter 3, Fig 3.1, with NG supply limits
    fig = cap_and_act(db)
    # fig.savefig('fig3_1.pdf', format='pdf')

    # Chapter 3, Fig 3.2, with NG supply limits
    fig = emissions_new(db)
    # fig.savefig('fig3_2.pdf', format='pdf')
    plt.show()
