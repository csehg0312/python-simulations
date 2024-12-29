import numpy as np
import matplotlib.pyplot as plt

class ChuaYangCNN:
    def __init__(self, size, a=10, b=15, coupling_strength=0.1):
        """
        Inicializálja a Chua-Yang Celluláris Neurális Hálózatot
        
        Paraméterek:
        - size: A hálózat térbeli mérete (NxN)
        - a, b: Nemlineáris paraméterek
        - coupling_strength: Térbeli csatolási erősség
        """
        self.size = size
        self.a = a
        self.b = b
        self.coupling_strength = coupling_strength
        
        # Inicializálja a hálózat állapotát
        self.state = np.random.rand(size, size)
    
    def nonlinear_activation(self, u):
        """
        Chua-féle nemlineáris aktivációs függvény
        
        u: Bemeneti érték
        return: Nemlineárisan transzformált érték
        """
        return (self.a * u + 
                0.5 * (self.b - self.a) * 
                (np.abs(u + 1) - np.abs(u - 1)))
    
    def spatial_laplacian(self, u):
        """
        Térbeli Laplace-operátor numerikus közelítése
        
        u: 2D állapottér
        return: Laplace-operátor által transzformált tér
        """
        # Peremfeltételek: nullás perem
        laplacian = np.zeros_like(u)
        
        # Belső pontok Laplace-operátora
        for i in range(1, u.shape[0] - 1):
            for j in range(1, u.shape[1] - 1):
                laplacian[i, j] = (
                    u[i+1, j] + u[i-1, j] + 
                    u[i, j+1] + u[i, j-1] - 
                    4 * u[i, j]
                )
        
        return laplacian
    
    def spatial_convolution(self, u):
        """
        Térbeli konvolúció közelítő számítása
        
        u: 2D állapottér
        return: Konvolúció eredménye
        """
        # Egyszerűsített szomszédsági súlyozás
        kernel = np.array([
            [0.25, 0.5, 0.25],
            [0.5,  0,    0.5],
            [0.25, 0.5, 0.25]
        ])
        
        convolved = np.zeros_like(u)
        for i in range(1, u.shape[0] - 1):
            for j in range(1, u.shape[1] - 1):
                local_region = u[i-1:i+2, j-1:j+2]
                convolved[i, j] = np.sum(local_region * kernel)
        
        return convolved
    
    def evolve(self, dt=0.01, steps=100):
        """
        A hálózat dinamikai fejlődésének szimulációja
        
        dt: Időlépés
        steps: Szimulációs lépések száma
        """
        history = [self.state.copy()]
        
        for _ in range(steps):
            # Nemlineáris dinamikai egyenlet numerikus integrálása
            delta_u = (
                self.spatial_laplacian(self.state) + 
                self.nonlinear_activation(self.state) + 
                self.coupling_strength * self.spatial_convolution(self.state)
            )
            
            # Euler-módszer szerinti időfejlődés
            self.state += dt * delta_u
            
            history.append(self.state.copy())
        
        return np.array(history)
    
    def visualize(self, history):
        """
        A hálózat térdinamikájának vizualizációja
        
        history: A hálózat állapotainak sorozata
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        timesteps = [0, len(history)//4, len(history)//2, 
                     3*len(history)//4, len(history)-1]
        
        for i, ts in enumerate(timesteps):
            im = axes[i].imshow(history[ts], cmap='viridis')
            axes[i].set_title(f'Állapot t = {ts}')
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        plt.show()

# Példa használat
cnn = ChuaYangCNN(size=50, a=10, b=15, coupling_strength=0.1)
history = cnn.evolve(steps=200)
cnn.visualize(history)