"""
Project Prometheus Demo Scenario: 
"Global Supply Chain Optimization"
"""
import time
from datetime import datetime, timedelta
import random
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

class SupplyChainEvent:
    def __init__(self, event_type: str, location: str, severity: float):
        self.timestamp = datetime.now()
        self.event_type = event_type  # "delay", "demand_spike", "disruption"
        self.location = location
        self.severity = severity
        self.resolved = False

class SupplyChainSimulator:
    def __init__(self):
        self.events = []
        self.agents_active = {}
        
    def generate_events(self):
        """Simulate real-world supply chain disruptions"""
        scenarios = [
            ("port_strike", "Shanghai", 0.8),
            ("demand_spike", "Chicago", 0.6), 
            ("trucker_shortage", "Rotterdam", 0.7),
            ("customs_delay", "Singapore", 0.5)
        ]
        return [SupplyChainEvent(*random.choice(scenarios)) 
               for _ in range(random.randint(1, 3))]

    def run_demo(self):
        print("=== PROJECT PROMETHEUS DEMO ===")
        print("Scenario: Global Supply Chain Crisis Response\n")
        
        # Initial event detection
        print(f"{datetime.now()} - Monitoring agents detect events:")
        events = self.generate_events()
        for event in events:
            print(f"❗ {event.event_type.upper()} at {event.location} (Severity: {event.severity})")
        
        # Dynamic agent response
        print("\n🌐 DAF Controller deploying specialist agents:")
        time.sleep(1)
        
        # Simulate agent spawning
        specialist_agents = [
            ("logistics_optimizer", "Route recalculation"),
            ("inventory_balancer", "Stock redistribution"),
            ("supplier_finder", "Alternative sourcing")
        ]
        
        for agent_type, capability in specialist_agents:
            agent_id = f"agent_{random.randint(1000,9999)}"
            self.agents_active[agent_id] = (agent_type, capability)
            print(f"🔄 Spawned {agent_type} ({agent_id}) for {capability}")
            time.sleep(0.5)
        
        # Simulate collaborative problem solving
        print("\n🤖 Agents collaborating to resolve crisis:")
        time.sleep(1)
        
        solutions = [
            "Rerouted shipments through Dubai",
            "Activated backup suppliers in Vietnam",
            "Optimized warehouse inventory allocation",
            "Negotiated expedited customs clearance"
        ]
        
        for event in events:
            if random.random() > 0.3:  # 70% success rate
                solution = random.choice(solutions)
                print(f"✅ Resolved {event.event_type} at {event.location}: {solution}")
                event.resolved = True
                time.sleep(1)
            else:
                print(f"⚠️ Partial resolution for {event.event_type} at {event.location}")
        
        # Performance metrics
        print("\n📊 Crisis Response Metrics:")
        print(f"- Events detected: {len(events)}")
        print(f"- Resolution rate: {sum(e.resolved for e in events)/len(events):.0%}")
        print(f"- Agents deployed: {len(self.agents_active)}")
        print(f"- Response time: {timedelta(seconds=len(events)*2)}")
        
        print("\nDEMO COMPLETE - Project Prometheus successfully mitigated supply chain disruption")

if __name__ == "__main__":
    simulator = SupplyChainSimulator()
    simulator.run_demo()