TWO-QUEUE SYSTEM SIMULATION - README
================================================================================

PROJECT OVERVIEW
----------------
Event-driven simulation of a two-queue system comparing Random vs Min-Queue 
packet assignment strategies. Generates 9 performance figures and comprehensive 
analysis.

QUICK START
-----------
Run the complete simulation:
    python queue_simulation.py

This will:
1. Run 720 simulations (9 parameter combinations × 8 data points × 10 seeds)
2. Generate 9 PNG figures in the outputs directory
3. Print progress and summary statistics
4. Take approximately 90 seconds to complete

FILES INCLUDED
--------------
queue_simulation.py     - Main simulation code (~400 lines)
PROJECT_REPORT.txt      - Comprehensive written report
README.txt              - This file
fig1_blocking_vs_lambda.png    - Blocking probability vs arrival rate
fig2_queue_vs_lambda.png       - Queue length vs arrival rate
fig3_sojourn_vs_lambda.png     - Sojourn time vs arrival rate
fig4_blocking_vs_mu.png        - Blocking probability vs service rate
fig5_queue_vs_mu.png           - Queue length vs service rate
fig6_sojourn_vs_mu.png         - Sojourn time vs service rate
fig7_blocking_vs_rho.png       - Blocking probability vs traffic load
fig8_queue_vs_rho.png          - Queue length vs traffic load
fig9_sojourn_vs_rho.png        - Sojourn time vs traffic load

IMPLEMENTATION DETAILS
----------------------

Architecture:
- Event-driven discrete simulation (not continuous time)
- Priority queue for events (Python heapq)
- Two Queue objects with finite capacity (K=10)
- Statistics collection at runtime

Key Classes:
- Event: Represents arrival/departure with timestamp
- Queue: Manages FCFS queue state and capacity
- QueueingSimulator: Main simulation engine

Strategies Implemented:
1. Random: Uniform random assignment (p=0.5 each queue)
2. Min-Queue: Always assign to shorter queue

Performance Metrics:
1. Blocking Probability = dropped / offered
2. Average Queue Length = mean length sampled at arrivals
3. Average Sojourn Time = mean time in system (admitted packets only)

Theoretical Validation:
- Random strategy modeled as two independent M/M/1/K queues
- Formulas for blocking probability, queue length, sojourn time
- Simulation matches theory within 2-3% error

CUSTOMIZATION
-------------

To modify simulation parameters, edit these sections in queue_simulation.py:

1. Change number of runs (default: 10):
   Line ~220: def run_multiple_simulations(..., num_runs=10)

2. Change packets per run (default: 10,000):
   Line ~130: while ... and self.packets_offered < 10000

3. Change queue capacity (default: 10):
   Line ~24: def __init__(self, capacity=10)

4. Change parameter ranges:
   Lines ~250-300 in vary_arrival_rate(), vary_service_rate(), vary_traffic_load()

5. Adjust figure quality:
   Line ~365: fig.savefig(..., dpi=150)  # Increase DPI for higher quality

RUNNING CUSTOM EXPERIMENTS
---------------------------

Example: Single simulation with specific parameters
>>> sim = QueueingSimulator(arrival_rate=2.0, service_rate=1.0, 
                            strategy='min', seed=42)
>>> sim.run()
>>> metrics = sim.get_metrics()
>>> print(metrics)

Example: Compare strategies at one operating point
>>> random_metrics = run_multiple_simulations(2.0, 1.0, 'random', num_runs=20)
>>> min_metrics = run_multiple_simulations(2.0, 1.0, 'min', num_runs=20)
>>> print(f"Blocking - Random: {random_metrics['blocking_prob']:.3f}")
>>> print(f"Blocking - Min: {min_metrics['blocking_prob']:.3f}")

Example: Quick theoretical check
>>> theory = theoretical_random_strategy(arrival_rate=2.0, service_rate=1.0)
>>> print(f"Theoretical blocking: {theory['blocking_prob']:.3f}")

UNDERSTANDING THE RESULTS
--------------------------

What the figures show:

Figures 1, 4, 7 (Blocking Probability):
- Y-axis: Fraction of packets dropped (0 to 1)
- Lower is better
- Min-queue substantially reduces blocking at medium loads

Figures 2, 5, 8 (Average Queue Length):
- Y-axis: Mean number of packets in queue
- Includes packet in service
- Min-queue has slightly HIGHER queue length (accepts more packets)

Figures 3, 6, 9 (Average Sojourn Time):
- Y-axis: Mean time packet spends in system
- Only counts admitted packets
- Lower is better
- Min-queue reduces waiting time

Key insights from experiments:

Varying λ (Figures 1-3):
- Shows how increasing traffic affects performance
- Critical region: λ > 1.5 with μ=1.0

Varying μ (Figures 4-6):
- Shows benefit of faster servers
- Performance improves as μ increases

Varying ρ (Figures 7-9):
- Most intuitive parameter (normalized load)
- Critical threshold around ρ = 0.8-1.0
- Min-queue provides largest advantage at ρ = 1.0-1.3

VALIDATION AND DEBUGGING
-------------------------

How to verify the simulation is working correctly:

1. Check conservation:
   - packets_offered = packets_admitted + packets_dropped ✓

2. Check theoretical match:
   - Random strategy results should match formulas closely
   - Look for <3% difference in blocking probability

3. Check bounds:
   - Queue length should never exceed capacity (10)
   - Blocking probability should be between 0 and 1
   - Sojourn time should be positive

4. Check trends:
   - Metrics should increase with λ, decrease with μ
   - Min-queue should outperform random (lower blocking)

5. Check randomness:
   - Different seeds should give slightly different results
   - Averaging 10 runs should reduce variance

THEORY REFERENCE
----------------

M/M/1/K Queue Formulas (Random Strategy):

Given:
- λ_q = λ/2 (arrival rate per queue)
- μ = service rate
- ρ_q = λ_q/μ (utilization per queue)
- K = 10 (capacity)

Blocking probability:
P_b = (1 - ρ_q) * ρ_q^K / (1 - ρ_q^(K+1))

Expected queue length:
E[N] = ρ_q * (1 - (K+1)*ρ_q^K + K*ρ_q^(K+1)) / ((1 - ρ_q) * (1 - ρ_q^(K+1)))

Expected sojourn time:
E[T] = E[N] / (λ_q * (1 - P_b))

Special case when ρ_q = 1:
P_b = K/(K+1) = 10/11 ≈ 0.909
E[N] = K/2 = 5

================================================================================
