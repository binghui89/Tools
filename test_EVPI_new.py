# This version is compatible with Pyomo 5.2

import os
import sys
from pyomo.environ import *
from pyomo.pysp.scenariotree.manager import \
    ScenarioTreeManagerClientSerial
from pyomo.pysp.ef import create_ef_instance
from pyomo.opt import SolverFactory
from time import time

from IPython import embed as IP


# To see detailed information about options
#for name in options.keys():
#    print(options.about(name))

# To see a more compact display of options
#options.display()

# options.model_location = \
#     os.path.join(farmer_example_dir, 'models')
# options.scenario_tree_location = \
#     os.path.join(farmer_example_dir, 'scenariodata')
    
def runef_farmer():
    # using the 'with' block will automatically call
    # manager.close() and gracefully shutdown

    options = ScenarioTreeManagerClientSerial.register_options()

    options.model_location = "/afs/unity.ncsu.edu/users/b/bli6/TEMOA_stochastic/Farmer"
    options.scenario_tree_location = "/afs/unity.ncsu.edu/users/b/bli6/TEMOA_stochastic/Farmer/scenariodata"

    with ScenarioTreeManagerClientSerial(options) as manager:
        manager.initialize()
    
        ef_instance = create_ef_instance(manager.scenario_tree,
                                         verbose_output=options.verbose)
    
        ef_instance.dual = Suffix(direction=Suffix.IMPORT)
    
        with SolverFactory('cplex') as opt:
    
            ef_result = opt.solve(ef_instance)
            IP()

def runef_temoa_twotechs():
    # using the 'with' block will automatically call
    # manager.close() and gracefully shutdown
    t0 = time()
    timeit = lambda: time() - t0

    options = ScenarioTreeManagerClientSerial.register_options()

    options.model_location = "/afs/unity.ncsu.edu/users/b/bli6/temoa/temoa_model"
    options.scenario_tree_location = "/afs/unity.ncsu.edu/users/b/bli6/TEMOA_stochastic/test_twotechs_1"

    sys.path.append(options.model_location)
    from pformat_results import pformat_results
    from temoa_config import TemoaConfig

    temoa_options = TemoaConfig()
    temoa_options.config = None
    temoa_options.keepPyomoLP = False
    temoa_options.saveTEXTFILE = False
    temoa_options.path_to_db_io = None
    temoa_options.saveEXCEL = False

    with ScenarioTreeManagerClientSerial(options) as manager:
        manager.initialize()
    
        ef_instance = create_ef_instance(manager.scenario_tree,
                                         verbose_output=options.verbose)
    
        ef_instance.dual = Suffix(direction=Suffix.IMPORT)
    
        with SolverFactory('cplex') as opt:
    
            ef_result = opt.solve(ef_instance)

        ef_result.solution.Status = 'feasible' # Assume it is feasible
        for s in manager.scenario_tree.scenarios:
            ins = s._instance
            temoa_options.scenario = s.name
            temoa_options.dot_dat = [ os.path.join(options.scenario_tree_location, s.name + '.dat') ]
            temoa_options.output = os.path.join(options.scenario_tree_location, 'two_techs.db')
            print s.name
            print temoa_options.dot_dat
            formatted_results = pformat_results( ins, ef_result, temoa_options )
            print timeit()
            print formatted_results.getvalue()

def runef_temoa():
    # using the 'with' block will automatically call
    # manager.close() and gracefully shutdown
    t0 = time()
    timeit = lambda: time() - t0

    options = ScenarioTreeManagerClientSerial.register_options()

    options.model_location = "/afs/unity.ncsu.edu/users/b/bli6/temoa/temoa_model"
    scenario_trees = [
    "/afs/unity.ncsu.edu/users/b/bli6/TEMOA_stochastic/NC/noIGCC-CP",
    "/afs/unity.ncsu.edu/users/b/bli6/TEMOA_stochastic/NC/noIGCC-noCP",
    "/afs/unity.ncsu.edu/users/b/bli6/TEMOA_stochastic/NC/IGCC-CP",
    "/afs/unity.ncsu.edu/users/b/bli6/TEMOA_stochastic/NC/IGCC-noCP",
    ]

    sys.path.append(options.model_location)
    from pformat_results import pformat_results
    from temoa_config import TemoaConfig

    temoa_options = TemoaConfig()
    temoa_options.config = None
    temoa_options.keepPyomoLP = False
    temoa_options.saveTEXTFILE = False
    temoa_options.path_to_db_io = None
    temoa_options.saveEXCEL = False

    for options.scenario_tree_location in scenario_trees:
        with ScenarioTreeManagerClientSerial(options) as manager:
            manager.initialize()

            ef_instance = create_ef_instance(manager.scenario_tree,
                                             verbose_output=options.verbose)

            ef_instance.dual = Suffix(direction=Suffix.IMPORT)

            with SolverFactory('cplex') as opt:

                ef_result = opt.solve(ef_instance)

            ef_result.solution.Status = 'feasible' # Assume it is feasible
            for s in manager.scenario_tree.scenarios:
                ins = s._instance
                temoa_options.scenario = s.name
                temoa_options.dot_dat = [ os.path.join(options.scenario_tree_location, s.name + '.dat') ]
                temoa_options.output = os.path.join(options.scenario_tree_location, 'NCreference.db')
                # temoa_options.output = os.path.join(options.scenario_tree_location, 'two_techs.db')
                print s.name
                print temoa_options.dot_dat
                formatted_results = pformat_results( ins, ef_result, temoa_options )
                print "Time elapsed: {} s".format( timeit() )
                # print formatted_results.getvalue()

if __name__ == "__main__":
    runef_temoa()
    # runef_farmer()
