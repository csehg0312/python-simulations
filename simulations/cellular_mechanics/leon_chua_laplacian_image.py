import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.ndimage as ndimage
import os

class ChuaYangImageProcessor:
    def __init__(self, a=10, b=15, coupling_strength=0.1):
        self.a = a
        self.b = b
        self.coupling_strength = coupling_strength
    
    def load_image(self, image_path):
        img = Image.open(image_path).convert('L')
        img_array = np.array(img, dtype=np.float64)
        return img_array / 255.0
    
    def nonlinear_activation(self, u):
        return (self.a * u + 
                0.5 * (self.b - self.a) * 
                (np.abs(u + 1) - np.abs(u - 1)))
    
    def spatial_dynamics_filter(self, image):
        laplacian = ndimage.laplace(image)
        nonlinear_transformed = self.nonlinear_activation(image)
        spatial_coupling = ndimage.gaussian_filter(image, sigma=1)
        
        filtered_image = (
            image + 
            0.1 * laplacian + 
            0.2 * nonlinear_transformed + 
            self.coupling_strength * spatial_coupling
        )
        
        return np.clip(filtered_image, 0, 1)
    
    def edge_enhancement(self, image):
        edges = ndimage.sobel(image)
        enhanced = image + 0.5 * edges
        return np.clip(enhanced, 0, 1)
    
    def noise_reduction(self, image, iterations=3):
        processed_image = image.copy()
        for _ in range(iterations):
            processed_image = self.spatial_dynamics_filter(processed_image)
        return processed_image
    
    def process_image(self, image_path, operations=['dynamics', 'edges', 'denoise']):
        original_image = self.load_image(image_path)
        results = {'original': original_image}
        
        if 'dynamics' in operations:
            results['spatial_dynamics'] = self.spatial_dynamics_filter(original_image)
        
        if 'edges' in operations:
            results['edge_enhanced'] = self.edge_enhancement(original_image)
        
        if 'denoise' in operations:
            results['denoised'] = self.noise_reduction(original_image)
        
        return results
    
    def visualize_results(self, results):
        plt.figure(figsize=(15, 5))
        
        titles = list(results.keys())
        for i, (title, image) in enumerate(results.items()):
            plt.subplot(1, len(results), i + 1)
            plt.imshow(image, cmap='gray')
            plt.title(title)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    processor = ChuaYangImageProcessor(
        a=10,
        b=15,
        coupling_strength=0.001
    )
    
    IMAGE_DIR = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        'images',
    ))
    
    image_path = os.path.join(IMAGE_DIR, 'input8.bmp')

    try:
        results = processor.process_image(
            image_path,
            operations=['dynamics', 'edges', 'denoise']
        )
        
        processor.visualize_results(results)
    
    except FileNotFoundError:
        print("Kérem, ellenőrizze a képfájl elérési útját!")

if __name__ == "__main__":
    main()