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
category_map = {
    'Bioenergy':          'BIO',
    'Coal':               'COA',
    'Oil':                'OIL',
    'EE':                 'EE',
    'Geothermal':         'GEO',
    'Hydro':              'HYD',
    'Pumped hydro':       'PUM',
    'Natural gas':        'NGA',
    'Solar PV':           'SOL',
    'Nuclear':            'NUC',
    'Wind':               'WND',
    'Emission reduction': 'Emission reduction',
    'other':              'other',
}
color_map['Emission reduction'] = 'white'
edge_map['Emission reduction'] = 'black'
hatch_map['Emission reduction'] = None
