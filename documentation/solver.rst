Package solver
=================

master
------------------------------

.. automodule:: master
	:members:
	:show-inheritance:

worker
------------------------------

.. automodule:: worker
	:members:
	:undoc-members:
	:show-inheritance:

forest
------------------------------

.. automodule:: forest
	:members:
	:undoc-members:
	:show-inheritance:

Basic use example of the forest class:
::

    import forest as ft
    from master_node import deepcopy_dict
    from master import MasterTree

    # Scenario parameters
    date = '20180223'  # January 30, 2018
    latBounds = [40, 50]
    lonBounds = [360 - 15, 360]

    ## Download and saves the 5 scenarios
    ## run this in a terminal if you actually want to download anything
    # ft.download_scenarios(date, latBound=latBounds, lonBound=lonBounds, scenario_ids=range(1, 5))

    # Load the weathers scnearios
    Weathers = ft.load_scenarios(date, latBound=latBounds, lonBound=lonBounds, scenario_ids=range(1, 5))

    # Simulators parameters
    NUMBER_OF_SIM = 4  # <=20
    SIM_TIME_STEP = 6  # in hours
    STATE_INIT = [0, 42.5, 360 - 11.5]  # first state
    N_DAYS_SIM = 4  # time horizon in days
    missionheading = 0
    n_trajectories = 50

    # Create the simulators
    sims = ft.create_simulators(Weathers, numberofsim=NUMBER_OF_SIM, simtimestep=SIM_TIME_STEP,
                                stateinit=STATE_INIT, ndaysim=N_DAYS_SIM)

    # Initialize the simulators to get common destination and individual time min
    destination, timemin = ft.initialize_simulators(sims, n_trajectories, STATE_INIT, missionheading, plot=True)
    print("destination : " + str(destination) + "  &  timemin : " + str(timemin) + "\n")

    # Search parameters
    name = "tree_exemple"
    frequency = 10
    budget = 100

    # Initialize the forest
    forest = ft.Forest(listsimulators=sims, destination=destination, timemin=timemin, budget=budget)

    # Launch the search
    master_nodes = forest.launch_search(STATE_INIT, frequency)

    # Save the result as a tree
    new_dict = deepcopy_dict(master_nodes)
    forest.master = MasterTree(sims, destination, nodes=new_dict)
    forest.master.save_tree(name)

    # Displays some juicy results
    forest.master.plot_tree_uct()
    forest.master.get_best_policy()
    forest.master.plot_best_policy()
    forest.master.plot_hist_best_policy(interactive=True)

master_node
---------------------------------

.. automodule:: master_node
	:members:
	:undoc-members:
	:show-inheritance:
