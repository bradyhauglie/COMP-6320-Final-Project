Two-Queue Simulation

Quick Start:
    python queue_simulation.py

This will run all the experiments and generate the 9 figures in the outputs/ folder. 
Takes about 90 seconds.


Files:
- queue_simulation.py: the simulation code
- PROJECT_REPORT.txt: written report with results
- fig1-9: all the performance graphs

The simulation compares two strategies for assigning packets to queues:
1. Random - 50/50 chance for each queue
2. Min-queue - always pick the shorter queue

Each run simulates 10,000 packets and we average over 10 runs to smooth out randomness.


Changing Parameters:

If you want to modify things:
- Queue capacity: line 20, change capacity=10
- Packets per run: line 130, change 10000 to something else
- Number of runs: line 141, change num_runs=10
- Parameter ranges: lines 250-300 in the vary_* functions


How It Works:

The simulator uses event-driven simulation - it jumps between packet arrivals and 
departures instead of running continuously. Events are stored in a priority queue 
sorted by time.

For the random strategy, we can compare against M/M/1/K queueing theory. The 
simulated results match theory within 2-3%, which confirms the code is correct.


Running Custom Tests:

To test a specific configuration:

>>> sim = QueueingSimulator(arrival_rate=2.0, service_rate=1.0, strategy='min', seed=42)
>>> sim.run()
>>> metrics = sim.get_metrics()
>>> print(metrics)


Main Results:

Min-queue is way better than random at moderate loads (ρ = 0.8 to 1.3):
- 50-70% less blocking
- 15-20% lower sojourn time

The critical region is around ρ = 0.8-1.0 where performance starts degrading fast.