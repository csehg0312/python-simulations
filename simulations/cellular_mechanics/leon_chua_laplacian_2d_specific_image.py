import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp

class LeonChuaNetwork:
    def __init__(self, a_set, b_set, initial_state=None):
        self.a_set = np.array(a_set)
        self.b_set = np.array(b_set)
        self.initial_state = initial_state if initial_state is not None else np.random.rand(2)
    
    
    
    def nonlinear_transfer_function(self, x):
        # Ellenőrizzük, hogy x numpy tömb-e
        x = np.asarray(x)
        nonlinear_x = np.zeros_like(x)
        
        for i in range(len(x)):
            nonlinear_x[i] = (
                self.a_set[i] * x[i] + 
                np.sin(self.b_set[i] * x[i])
            )
        
        return nonlinear_x
    
    def network_dynamics(self, t, state):
        state = np.asarray(state)  # Biztosítjuk, hogy a state numpy tömb legyen
        dx_dt = np.zeros_like(state)
        
        # A nemlineáris transzformációt a teljes állapotvektorra alkalmazzuk
        nonlinear_output = self.nonlinear_transfer_function(state)
        
        for i in range(len(state)):
            dx_dt[i] = nonlinear_output[i] + np.random.normal(0, 0.01)
        
        return dx_dt
    
    def simulate(self, time_span, dt=0.01):
        t = np.arange(0, time_span, dt)
        
        # Numerikus integrálás solve_ivp használatával
        solution = solve_ivp(
            self.network_dynamics, 
            [0, time_span], 
            self.initial_state, 
            t_eval=t,
            method='RK45',  # Választható metódus
            atol=1e-6, 
            rtol=1e-6
        )
        
        return {
            'time': solution.t,
            'states': solution.y.T  # Transzponálás, hogy a megfelelő formátumot kapjuk
        }
        
        return {
            'time': solution.t,
            'states': solution.y.T  # Transzponálás, hogy a megfelelő formátumot kapjuk
        }
    
    def bifurcation_analysis(self, param_range, num_points=100):
        bifurcation_data = []
        for param_value in np.linspace(param_range[0], param_range[1], num_points):
            print(f"Bifurcation {param_value}")
            modified_a_set = self.a_set * param_value
            modified_b_set = self.b_set * param_value
            temp_network = LeonChuaNetwork(modified_a_set, modified_b_set, self.initial_state)
            simulation = temp_network.simulate(time_span=1)
            bifurcation_data.append({
                'param': param_value,
                'states': simulation['states'][-20:]  # Utolsó 20 állapot
            })
        return bifurcation_data
    
    def visualize_dynamics(self, simulation_result):
        plt.figure(figsize=(15, 5))
        
        # Időfejlődés
        plt.subplot(131)
        plt.plot(simulation_result['time'], simulation_result['states'][:, 0], label='X1')
        plt.plot(simulation_result['time'], simulation_result['states'][:, 1], label='X2')
        plt.title('Állapotterek időfejlődése')
        plt.xlabel('Idő')
        plt.ylabel('Állapotérték')
        plt.legend()
        
        # Fázisdiagram
        plt.subplot(132)
        plt.plot(simulation_result['states'][:, 0], simulation_result['states'][:, 1])
        plt.title('Fázisdiagram')
        plt.xlabel('X1')
        plt.ylabel('X2')
        
        # Spektrum
        plt.subplot(133)
        plt.plot(simulation_result['time'], simulation_result['states'][:, 0])
        plt.title('Spektrum')
        plt.xlabel('Idő')
        plt.ylabel('X1 érték')
        
        plt.tight_layout()
        plt.show()

def main():
    a_set = np.array([0.5, 1.2])
    b_set = np.array([2.3, 0.7])
    
    leon_chua_network = LeonChuaNetwork(
        a_set, 
        b_set, 
        initial_state=np.array([0.1, 0.2])
    )
    
    simulation = leon_chua_network.simulate(time_span=1)
    
    leon_chua_network.visualize_dynamics(simulation)
    
    bifurcation_result = leon_chua_network.bifurcation_analysis(
        param_range=[0.1, 2.0], 
        num_points=50
    )
    
    plt.figure(figsize=(10, 6))
    for data in bifurcation_result:
        plt.scatter(
            [data['param']] * len(data['states']), 
            data['states'][:, 0], 
            color='blue', 
            alpha=0.1
        )
    plt.title('Bifurkációs diagram')
    plt.xlabel('Paraméter skála')
    plt.ylabel('Állapotértékek')
    plt.show()

if __name__ == "__main__":
    main()