# call_center_simulation_complete.py
import simpy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class CallCenter:
    def __init__(self, env, num_agents, call_arrival_rate, service_rate):
        self.env = env
        self.agents = simpy.Resource(env, num_agents)
        self.call_arrival_rate = call_arrival_rate
        self.service_rate = service_rate
        self.metrics = {
            'wait_times': [],
            'service_times': [],
            'total_times': [],
            'queue_lengths': [],
            'abandoned_calls': 0,
            'total_calls': 0
        }
        
    def call_arrival_process(self):
        """Generate incoming calls according to Poisson process"""
        call_id = 0
        while True:
            # Exponential inter-arrival times
            yield self.env.timeout(random.expovariate(self.call_arrival_rate))
            call_id += 1
            self.metrics['total_calls'] += 1
            self.env.process(self.handle_call(call_id))
    
    def handle_call(self, call_id):
        """Process a single call"""
        arrival_time = self.env.now
        
        # Customer patience - will abandon if wait too long
        max_wait_time = random.expovariate(1/10)  # Average patience of 10 minutes
        
        with self.agents.request() as request:
            # Wait for agent or abandon
            results = yield request | self.env.timeout(max_wait_time)
            
            if request in results:
                # Call answered
                wait_time = self.env.now - arrival_time
                self.metrics['wait_times'].append(wait_time)
                
                # Record queue length at arrival
                queue_length = len(self.agents.queue)
                self.metrics['queue_lengths'].append(queue_length)
                
                # Service time (exponential distribution)
                service_time = random.expovariate(self.service_rate)
                self.metrics['service_times'].append(service_time)
                yield self.env.timeout(service_time)
                
                total_time = self.env.now - arrival_time
                self.metrics['total_times'].append(total_time)
            else:
                # Call abandoned
                self.metrics['abandoned_calls'] += 1

def run_simulation(num_agents=5, call_arrival_rate=0.8, service_rate=0.2, sim_time=480):
    """Run a single simulation scenario"""
    env = simpy.Environment()
    call_center = CallCenter(env, num_agents, call_arrival_rate, service_rate)
    env.process(call_center.call_arrival_process())
    env.run(until=sim_time)
    
    return calculate_metrics(call_center.metrics, num_agents, sim_time)

def calculate_metrics(metrics, num_agents, sim_time):
    """Calculate performance metrics from simulation data"""
    if not metrics['wait_times']:
        return {
            'avg_wait_time': 0,
            'max_wait_time': 0,
            'agent_utilization': 0,
            'abandonment_rate': 1.0,
            'service_level': 0,
            'avg_queue_length': 0
        }
    
    total_service_time = sum(metrics['service_times'])
    potential_agent_time = num_agents * sim_time
    
    return {
        'avg_wait_time': np.mean(metrics['wait_times']),
        'max_wait_time': np.max(metrics['wait_times']),
        'agent_utilization': total_service_time / potential_agent_time,
        'abandonment_rate': metrics['abandoned_calls'] / metrics['total_calls'],
        'service_level': len([wt for wt in metrics['wait_times'] if wt <= 1.0]) / len(metrics['wait_times']),
        'avg_queue_length': np.mean(metrics['queue_lengths'])
    }

def scenario_agent_staffing():
    """Test different staffing levels"""
    agents_range = range(3, 9)
    results = []
    
    for num_agents in agents_range:
        print(f"Testing with {num_agents} agents...")
        metrics = run_simulation(num_agents=num_agents)
        metrics['num_agents'] = num_agents
        results.append(metrics)
    
    return pd.DataFrame(results)

def scenario_call_volume():
    """Test different call arrival rates"""
    arrival_rates = [0.5, 0.8, 1.1, 1.4, 1.7]  # calls per minute
    results = []
    
    for rate in arrival_rates:
        print(f"Testing arrival rate: {rate} calls/min...")
        metrics = run_simulation(num_agents=5, call_arrival_rate=rate)
        metrics['arrival_rate'] = rate
        metrics['offered_load'] = rate / 0.2  # arrival_rate / service_rate
        results.append(metrics)
    
    return pd.DataFrame(results)

def scenario_service_improvement():
    """Test impact of service time reduction"""
    service_rates = [0.15, 0.18, 0.2, 0.22, 0.25]  # faster service = higher rate
    results = []
    
    for rate in service_rates:
        print(f"Testing service rate: {rate}...")
        metrics = run_simulation(num_agents=5, service_rate=rate)
        metrics['avg_service_time'] = 1/rate
        metrics['service_rate'] = rate
        results.append(metrics)
    
    return pd.DataFrame(results)

# Run all scenarios and create visualizations
print("Starting Call Center Simulation...")
print("=" * 50)

# Scenario 1: Agent staffing
print("\nRunning Scenario 1: Agent Staffing Analysis")
agent_results = scenario_agent_staffing()

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(agent_results['num_agents'], agent_results['avg_wait_time'], 'bo-')
plt.xlabel('Number of Agents')
plt.ylabel('Average Wait Time (min)')
plt.title('Wait Time vs. Staffing Level')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(agent_results['num_agents'], agent_results['abandonment_rate'], 'ro-')
plt.xlabel('Number of Agents')
plt.ylabel('Abandonment Rate')
plt.title('Abandonment Rate vs. Staffing Level')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(agent_results['num_agents'], agent_results['agent_utilization'], 'go-')
plt.xlabel('Number of Agents')
plt.ylabel('Agent Utilization')
plt.title('Utilization vs. Staffing Level')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(agent_results['num_agents'], agent_results['service_level'], 'mo-')
plt.xlabel('Number of Agents')
plt.ylabel('Service Level (%)')
plt.title('Service Level vs. Staffing Level')
plt.grid(True)

plt.tight_layout()
plt.savefig('agent_staffing_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Scenario 2: Call volume
print("\nRunning Scenario 2: Call Volume Analysis")
volume_results = scenario_call_volume()

plt.figure(figsize=(10, 6))
plt.plot(volume_results['offered_load'], volume_results['avg_wait_time'], 's-', label='Wait Time')
plt.plot(volume_results['offered_load'], volume_results['abandonment_rate']*10, '^-', label='Abandonment Rate (x10)')
plt.xlabel('Offered Load (Erlangs)')
plt.ylabel('Performance Metrics')
plt.title('System Performance vs. Offered Load')
plt.legend()
plt.grid(True)
plt.savefig('load_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Scenario 3: Service time optimization
print("\nRunning Scenario 3: Service Time Analysis")
service_results = scenario_service_improvement()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(service_results['avg_service_time'], service_results['avg_wait_time'], 'o-')
plt.xlabel('Average Service Time (min)')
plt.ylabel('Average Wait Time (min)')
plt.title('Wait Time vs. Service Time')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(service_results['avg_service_time'], service_results['agent_utilization'], 'o-')
plt.xlabel('Average Service Time (min)')
plt.ylabel('Agent Utilization')
plt.title('Utilization vs. Service Time')
plt.grid(True)

plt.tight_layout()
plt.savefig('service_time_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary results
print("\n" + "=" * 50)
print("SIMULATION RESULTS SUMMARY")
print("=" * 50)

print("\nScenario 1 - Agent Staffing:")
print(agent_results[['num_agents', 'avg_wait_time', 'abandonment_rate', 'service_level', 'agent_utilization']].round(3))

print("\nScenario 2 - Call Volume:")
print(volume_results[['arrival_rate', 'offered_load', 'avg_wait_time', 'abandonment_rate']].round(3))

print("\nScenario 3 - Service Time:")
print(service_results[['avg_service_time', 'avg_wait_time', 'agent_utilization']].round(3))

print(f"\nSimulation completed successfully!")
print("Charts saved as:")
print("- agent_staffing_analysis.png")
print("- load_analysis.png") 
print("- service_time_analysis.png")