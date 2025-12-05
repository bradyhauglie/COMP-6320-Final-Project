import heapq
import random
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from enum import Enum

class EventType(Enum):
    ARRIVAL = 1
    DEPARTURE = 2

@dataclass(order=True)
class Event:
    time: float
    event_type: EventType = field(compare=False)
    queue_id: int = field(compare=False, default=-1)

class Queue:
    def __init__(self, capacity=10):
        self.capacity = capacity
        self.packets = []
        self.in_service = None
        
    def length(self):
        return len(self.packets) + (1 if self.in_service is not None else 0)
    
    def is_full(self):
        return self.length() >= self.capacity
    
    def add_packet(self, arrival_time):
        # Return True if successful, False if dropped
        if self.is_full():
            return False
        self.packets.append(arrival_time)
        return True
    
    def start_service(self, current_time):
        # Start serving next packet if available
        if self.in_service is None and len(self.packets) > 0:
            self.in_service = self.packets.pop(0)
            return True
        return False
    
    def finish_service(self):
        # Complete service and return arrival time
        packet = self.in_service
        self.in_service = None
        return packet

class QueueingSimulator:
    def __init__(self, arrival_rate, service_rate, strategy='random', seed=None):
        self.lambda_ = arrival_rate
        self.mu = service_rate
        self.strategy = strategy
        self.rng = random.Random(seed)
        
        self.queues = [Queue(capacity=10), Queue(capacity=10)]
        
        self.event_queue = []
        self.current_time = 0.0
        
        # Statistics
        self.packets_offered = 0
        self.packets_dropped = 0
        self.packets_admitted = 0
        self.total_sojourn_time = 0.0
        self.queue_length_samples = []
        
    def schedule_event(self, event):
        heapq.heappush(self.event_queue, event)
    
    def generate_arrival_time(self):
        inter_arrival = self.rng.expovariate(self.lambda_)
        return self.current_time + inter_arrival
    
    def generate_service_time(self):
        return self.rng.expovariate(self.mu)
    
    def select_queue(self):
        if self.strategy == 'random':
            return self.rng.randint(0, 1)
        elif self.strategy == 'min':
            if self.queues[0].length() <= self.queues[1].length():
                return 0
            else:
                return 1
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def handle_arrival(self, event):
        self.packets_offered += 1
        
        # Sample queue lengths at arrival time
        self.queue_length_samples.append(self.queues[0].length())
        self.queue_length_samples.append(self.queues[1].length())
        
        # Select queue
        queue_id = self.select_queue()
        queue = self.queues[queue_id]
        
        # Try to add packet
        if queue.add_packet(self.current_time):
            self.packets_admitted += 1
            
            # If queue was empty, start service immediately
            if queue.start_service(self.current_time):
                service_time = self.generate_service_time()
                departure_event = Event(
                    time=self.current_time + service_time,
                    event_type=EventType.DEPARTURE,
                    queue_id=queue_id
                )
                self.schedule_event(departure_event)
        else:
            self.packets_dropped += 1
        
        # Schedule next arrival
        if self.packets_offered < 10000:
            next_arrival = Event(
                time=self.generate_arrival_time(),
                event_type=EventType.ARRIVAL
            )
            self.schedule_event(next_arrival)
    
    def handle_departure(self, event):
        queue_id = event.queue_id
        queue = self.queues[queue_id]
        
        # Complete service
        arrival_time = queue.finish_service()
        sojourn_time = self.current_time - arrival_time
        self.total_sojourn_time += sojourn_time
        
        # Start serving next packet if available
        if queue.start_service(self.current_time):
            service_time = self.generate_service_time()
            departure_event = Event(
                time=self.current_time + service_time,
                event_type=EventType.DEPARTURE,
                queue_id=queue_id
            )
            self.schedule_event(departure_event)
    
    def run(self):
        # Schedule first arrival
        first_arrival = Event(
            time=self.generate_arrival_time(),
            event_type=EventType.ARRIVAL
        )
        self.schedule_event(first_arrival)
        
        # Process events
        while self.event_queue and self.packets_offered < 10000:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time
            
            if event.event_type == EventType.ARRIVAL:
                self.handle_arrival(event)
            elif event.event_type == EventType.DEPARTURE:
                self.handle_departure(event)
        
        # Wait for remaining packets to finish service
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            if event.event_type == EventType.DEPARTURE:
                self.current_time = event.time
                self.handle_departure(event)
    
    def get_metrics(self):
        blocking_prob = self.packets_dropped / self.packets_offered if self.packets_offered > 0 else 0
        avg_queue_length = np.mean(self.queue_length_samples) if self.queue_length_samples else 0
        avg_sojourn_time = self.total_sojourn_time / self.packets_admitted if self.packets_admitted > 0 else 0
        
        return {
            'blocking_prob': blocking_prob,
            'avg_queue_length': avg_queue_length,
            'avg_sojourn_time': avg_sojourn_time
        }

def run_multiple_simulations(arrival_rate, service_rate, strategy, num_runs=10):
    results = {
        'blocking_prob': [],
        'avg_queue_length': [],
        'avg_sojourn_time': []
    }
    
    for seed in range(num_runs):
        sim = QueueingSimulator(arrival_rate, service_rate, strategy, seed=seed)
        sim.run()
        metrics = sim.get_metrics()
        
        for key in results:
            results[key].append(metrics[key])
    
    # Return averages
    return {key: np.mean(values) for key, values in results.items()}

def theoretical_random_strategy(arrival_rate, service_rate):
    # Each queue sees half the arrival rate
    lambda_per_queue = arrival_rate / 2
    mu = service_rate
    rho = lambda_per_queue / mu
    K = 10  # Capacity
    
    if abs(rho - 1.0) < 1e-6:
        # Special case when rho = 1
        blocking_prob = K / (K + 1)
        avg_queue_length = K / 2
    else:
        # Blocking probability: P(K)
        blocking_prob = ((1 - rho) * rho**K) / (1 - rho**(K + 1))
        
        # Average queue length (including packet in service)
        numerator = rho * (1 - (K + 1) * rho**K + K * rho**(K + 1))
        denominator = (1 - rho) * (1 - rho**(K + 1))
        avg_queue_length = numerator / denominator
    
    # Average sojourn time (Little's Law for admitted packets)
    effective_arrival_rate = lambda_per_queue * (1 - blocking_prob)
    avg_sojourn_time = avg_queue_length / effective_arrival_rate if effective_arrival_rate > 0 else 0
    
    return {
        'blocking_prob': blocking_prob,
        'avg_queue_length': avg_queue_length,
        'avg_sojourn_time': avg_sojourn_time
    }

def vary_arrival_rate(service_rate=1.0, num_points=10):
    # Vary arrival rate from low to high
    arrival_rates = np.linspace(0.2, 3.8, num_points)
    
    random_results = {'blocking_prob': [], 'avg_queue_length': [], 'avg_sojourn_time': []}
    min_results = {'blocking_prob': [], 'avg_queue_length': [], 'avg_sojourn_time': []}
    theory_results = {'blocking_prob': [], 'avg_queue_length': [], 'avg_sojourn_time': []}
    
    for lambda_ in arrival_rates:
        print(f"  lambda = {lambda_:.2f}")
        
        # Random strategy
        metrics = run_multiple_simulations(lambda_, service_rate, 'random')
        for key in random_results:
            random_results[key].append(metrics[key])
        
        # Min-queue strategy
        metrics = run_multiple_simulations(lambda_, service_rate, 'min')
        for key in min_results:
            min_results[key].append(metrics[key])
        
        # Theoretical
        metrics = theoretical_random_strategy(lambda_, service_rate)
        for key in theory_results:
            theory_results[key].append(metrics[key])
    
    return arrival_rates, random_results, min_results, theory_results

def vary_service_rate(arrival_rate=2.0, num_points=10):
    # Vary service rate
    service_rates = np.linspace(0.6, 3.0, num_points)
    
    random_results = {'blocking_prob': [], 'avg_queue_length': [], 'avg_sojourn_time': []}
    min_results = {'blocking_prob': [], 'avg_queue_length': [], 'avg_sojourn_time': []}
    theory_results = {'blocking_prob': [], 'avg_queue_length': [], 'avg_sojourn_time': []}
    
    for mu in service_rates:
        print(f"  mu = {mu:.2f}")
        
        # Random strategy
        metrics = run_multiple_simulations(arrival_rate, mu, 'random')
        for key in random_results:
            random_results[key].append(metrics[key])
        
        # Min-queue strategy
        metrics = run_multiple_simulations(arrival_rate, mu, 'min')
        for key in min_results:
            min_results[key].append(metrics[key])
        
        # Theoretical
        metrics = theoretical_random_strategy(arrival_rate, mu)
        for key in theory_results:
            theory_results[key].append(metrics[key])
    
    return service_rates, random_results, min_results, theory_results

def vary_traffic_load(num_points=10):
    # Keep mu=1, vary lambda to get different rho values
    mu = 1.0
    rho_values = np.linspace(0.1, 1.9, num_points)
    arrival_rates = rho_values * 2 * mu  # lambda = 2murho
    
    random_results = {'blocking_prob': [], 'avg_queue_length': [], 'avg_sojourn_time': []}
    min_results = {'blocking_prob': [], 'avg_queue_length': [], 'avg_sojourn_time': []}
    theory_results = {'blocking_prob': [], 'avg_queue_length': [], 'avg_sojourn_time': []}
    
    for lambda_ in arrival_rates:
        rho = lambda_ / (2 * mu)
        print(f"  rho = {rho:.2f}")
        
        # Random strategy
        metrics = run_multiple_simulations(lambda_, mu, 'random')
        for key in random_results:
            random_results[key].append(metrics[key])
        
        # Min-queue strategy
        metrics = run_multiple_simulations(lambda_, mu, 'min')
        for key in min_results:
            min_results[key].append(metrics[key])
        
        # Theoretical
        metrics = theoretical_random_strategy(lambda_, mu)
        for key in theory_results:
            theory_results[key].append(metrics[key])
    
    return rho_values, random_results, min_results, theory_results

def plot_results(x_values, x_label, random_results, min_results, theory_results, metric_name, metric_label):
    plt.figure(figsize=(10, 6))
    
    plt.plot(x_values, random_results[metric_name], 'o-', label='Random (Simulated)', linewidth=2, markersize=6)
    plt.plot(x_values, min_results[metric_name], 's-', label='Min-Queue (Simulated)', linewidth=2, markersize=6)
    plt.plot(x_values, theory_results[metric_name], '^--', label='Random (Theoretical)', linewidth=2, markersize=6)
    
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(metric_label, fontsize=12)
    plt.title(f'{metric_label} vs {x_label}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def main():
    print("Running simulations...")
    
    # Create output directory
    import os
    os.makedirs('outputs', exist_ok=True)
    
    # Experiment 1: Vary arrival rate
    print("\n1. Varying arrival rate (lambda)...")
    arrival_rates, random_res1, min_res1, theory_res1 = vary_arrival_rate(service_rate=1.0, num_points=8)
    
    fig1 = plot_results(arrival_rates, 'Arrival Rate (lambda)', random_res1, min_res1, theory_res1,
                        'blocking_prob', 'Blocking Probability')
    fig1.savefig('outputs/fig1_blocking_vs_lambda.png', dpi=150)
    plt.close()
    
    fig2 = plot_results(arrival_rates, 'Arrival Rate (lambda)', random_res1, min_res1, theory_res1,
                        'avg_queue_length', 'Average Queue Length')
    fig2.savefig('outputs/fig2_queue_vs_lambda.png', dpi=150)
    plt.close()
    
    fig3 = plot_results(arrival_rates, 'Arrival Rate (lambda)', random_res1, min_res1, theory_res1,
                        'avg_sojourn_time', 'Average Sojourn Time')
    fig3.savefig('outputs/fig3_sojourn_vs_lambda.png', dpi=150)
    plt.close()
    
    # Experiment 2: Vary service rate
    print("\n2. Varying service rate (mu)...")
    service_rates, random_res2, min_res2, theory_res2 = vary_service_rate(arrival_rate=2.0, num_points=8)
    
    fig4 = plot_results(service_rates, 'Service Rate (mu)', random_res2, min_res2, theory_res2,
                        'blocking_prob', 'Blocking Probability')
    fig4.savefig('outputs/fig4_blocking_vs_mu.png', dpi=150)
    plt.close()
    
    fig5 = plot_results(service_rates, 'Service Rate (mu)', random_res2, min_res2, theory_res2,
                        'avg_queue_length', 'Average Queue Length')
    fig5.savefig('outputs/fig5_queue_vs_mu.png', dpi=150)
    plt.close()
    
    fig6 = plot_results(service_rates, 'Service Rate (mu)', random_res2, min_res2, theory_res2,
                        'avg_sojourn_time', 'Average Sojourn Time')
    fig6.savefig('outputs/fig6_sojourn_vs_mu.png', dpi=150)
    plt.close()
    
    # Experiment 3: Vary traffic load
    print("\n3. Varying traffic load (rho)...")
    rho_values, random_res3, min_res3, theory_res3 = vary_traffic_load(num_points=8)
    
    fig7 = plot_results(rho_values, 'Traffic Load (rho)', random_res3, min_res3, theory_res3,
                        'blocking_prob', 'Blocking Probability')
    fig7.savefig('outputs/fig7_blocking_vs_rho.png', dpi=150)
    plt.close()
    
    fig8 = plot_results(rho_values, 'Traffic Load (rho)', random_res3, min_res3, theory_res3,
                        'avg_queue_length', 'Average Queue Length')
    fig8.savefig('outputs/fig8_queue_vs_rho.png', dpi=150)
    plt.close()
    
    fig9 = plot_results(rho_values, 'Traffic Load (rho)', random_res3, min_res3, theory_res3,
                        'avg_sojourn_time', 'Average Sojourn Time')
    fig9.savefig('outputs/fig9_sojourn_vs_rho.png', dpi=150)
    plt.close()
    
    print("\nDone! All figures saved to ./outputs/")

if __name__ == "__main__":
    main()