"""
data_generator.py

Generate synthetic waveguide spectral measurements for testing

Creates realistic test data with controllable uniformity characteristics
Simulates spatial variations typical in waveguide manufacturing

Author: Olufemi Balogun
Date: October 2025
"""

import numpy as np
from typing import Tuple, Optional
import pandas as pd


class WaveguideDataGenerator:
    """
    Generate synthetic waveguide transmission spectra with spatial variations

    Simulate realistic manufacturing defects:
    - Center-to-edge uniformity gradients
    - Localized defects (e.g., scratches, pits, hotspots, dead spots)
    - Wavelength-dependent variations
    - Measurement noise
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize generator
        :param seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        self.wavelengths = np.linspace(400, 800,200)  # 2nm resolution

    def generate_grid(self, grid_size: Tuple[int, int] = (5, 5), spatial_extent: Tuple[float, float] = (10.0, 10.0),
                      quality: str = 'good') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate grid of waveguide measurements
        :param grid_size: (nx, ny) number of measurement points
        :param spatial_extent: (width_mm, height_mm) physical size
        :param quality: 'excellent', 'good', 'defect', 'poor'
        :return: Tuple of (positions, spectra, wavelengths)
        """
        nx, ny = grid_size
        width, height = spatial_extent

        # Create grid positions
        x = np.linspace(0, width, nx)
        y = np.linspace(0, height, ny)
        X, Y = np.meshgrid(x, y)
        positions = np.column_stack([X.ravel(), Y.ravel()])

        n_positions = len(positions)
        n_wavelengths = len(self.wavelengths)

        # Generate spectra for each position
        spectra = np.zeros((n_positions, n_wavelengths))

        centre_x, centre_y = width / 2, height / 2

        for i in range(n_positions):
            x_pos, y_pos = positions[i]

            # Calculate distance from center
            r = np.sqrt((x_pos - centre_x)**2 + (y_pos - centre_y)**2)
            max_r = np.sqrt(centre_x**2 + centre_y**2)
            r_norm = r / max_r # 0 at center, 1 at edge

            # Generate base spectrum
            spectrum = self.generate_spectrum(
                position=(x_pos, y_pos),
                distance_from_center=r_norm,
                quality=quality
            )

            spectra[i] = spectrum

        return positions, spectra, self.wavelengths

    def generate_spectrum(self, position: Tuple[float, float], distance_from_center: float, quality: str) -> np.ndarray:
        """
        Generate a single spectrum for a given position and quality
        :param position: (x, y) position
        :param distance_from_center: Normalized distance from (0,1)
        :param quality: Quality level
        :return: Spectrum transmission array
        """
        wl = self.wavelengths


        # Base transmission profile (typical waveguide)
        # Peak in green-yellow region, fall off at edges
        base_transmission = 0.75 + 0.15 * np.exp(-((wl-550)**2) / (2 * 80**2))

        if quality == 'excellent':
            # Minimal variation
            centre_shift = 0
            uniformity_factor = 1.0
            noise_level = 0.002

        elif quality == 'good':
            # Slight centre-to-edge variation
            centre_shift = distance_from_center * 5 # nm
            uniformity_factor = 1.0 - distance_from_center * 0.05
            noise_level = 0.005

        elif quality == 'defect':
            # Significant defect at specific location
            x_pos, y_pos = position

            # Create localized defect
            defect_x, defect_y = 7.0, 3.0 # Defect location
            defect_distance = np.sqrt((x_pos - defect_x)**2 + (y_pos - defect_y)**2)

            if defect_distance < 2.0:
                # Near defect, reduce transmission
                defect_factor = 0.6 + 0.3 * (defect_distance / 2.0)
            else:
                defect_factor = 1.0

            centre_shift = distance_from_center * 8
            uniformity_factor = defect_factor * (1.0 - distance_from_center * 0.1)
            noise_level = 0.008

        elif quality == 'poor':
            # Poor uniformity, strong edge effects
            centre_shift = distance_from_center * 15
            uniformity_factor = 1.0 - distance_from_center * 0.25
            noise_level = 0.015

            # Add wavelength-dependent artifact
            artifact = 0.05 * np.sin(wl / 30) * distance_from_center
            base_transmission += artifact

        #Apply spatial effects
        # 1. Wavelength shift (simulates thickness variation)
        spectrum = np.interp(wl, wl + centre_shift, base_transmission, left=base_transmission[0], right=base_transmission[-1])

        # 2. Amplitude scaling (simulates coupling efficiency)
        spectrum *= uniformity_factor

        # 3. Add realistic interference fringes
        fringe_period = 5.0 # nm
        fringe_amplitude = 0.02
        fringes = fringe_amplitude * np.sin(2 * np.pi * wl / fringe_period)
        spectrum += fringes

        # 4. Add measurement noise
        noise = np.random.normal(0, noise_level, len(wl))
        spectrum += noise

        #5. Physical constraints
        spectrum = np.clip(spectrum, 0.0, 1.0)

        return spectrum

    def generate_dark_spectrum(self) -> np.ndarray:
        """
        Generate a dark spectrum (offset measurement)
        :return: Dark spectrum array
        """
        # Small offset with noise
        dark = np.random.normal(0, 0.002, len(self.wavelengths))
        dark = np.clip(dark, 0.0, 0.05)
        return dark

    def generate_reference_spectrum(self) -> np.ndarray:
        """
        Generate a reference spectrum (ideal white)
        :return: Reference spectrum
        """
        # Nearly flat, high transmission
        reference = 0.95 * np.ones(len(self.wavelengths))
        reference += np.random.normal(0, 0.005, len(self.wavelengths))
        reference = np.clip(reference, 0.9, 1.0)
        return reference

    def add_temporal_drift(self, spectra: np.ndarray, drift_amplitude: float = 0.02) -> np.ndarray:
        """
        Add temporal drift to simulate instrument warm-up or aging
        :param spectra: Array of spectra (n_measurements, n_wavelengths)
        :param drift_amplitude: Maximum drift amount
        :return: Array of spectra with added drift
        """
        n_measurements = spectra.shape[0]


        # Exponential drift (common in instruments)
        time_constant = n_measurements / 3
        drift = drift_amplitude * (1 - np.exp(-np.arange(n_measurements) / time_constant))

        # Apply drift
        spectra_drifted = spectra + drift[:, np.newaxis]
        spectra_drifted = np.clip(spectra_drifted, 0.0, 1.0)

        return spectra_drifted

    def save_to_csv(self, positions: np.ndarray, spectra: np.ndarray, wavelengths: np.ndarray, filename: str, metadata: Optional[dict] = None):
        """
        Save generated data to CSV file
        :param positions: Array of positions (N, 2)
        :param spectra: Array of spectra (N, M)
        :param wavelengths: Array of wavelengths
        :param filename: Output file name
        :param metadata: Optional metadata dictionary
        """
        n_positions = positions.shape[0]

        # Create DataFrame
        data = {'position_x_mm': positions[:, 0],
                'position_y_mm': positions[:, 1]}

        # Add spectral columns
        for i, wl in enumerate(wavelengths):
            data[f'wl_{wl:.2f}nm'] = spectra[:, i]

        df = pd.DataFrame(data)

        # Write with metadata as comments
        with open(filename, 'w') as f:
            if metadata:
                f.write("# Waveguide Uniformity Measurement Data\n")
                for key, value in metadata.items():
                    f.write(f"# {key}: {value}\n")
                f.write("\n")

            # Write data
            df.to_csv(f, index=False)

        print(f"Data saved to {filename}")

    def load_from_csv(self, filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data from CSV format
        :param filename: Input filename
        :return: Tuple of (position, wavelengths, spectra)
        """
        df = pd.read_csv(filename, comment='#')

        # Extract positions
        positions = df[['position_x_mm', 'position_y_mm']].values

        # Extract wavelengths and spectra
        wl_columns = [col for col in df.columns if col.startswith('wl_')]
        wavelengths = np.array([float(col.split('_')[1].replace('nm', '')) for col in wl_columns])
        spectra = df[wl_columns].values

        return positions, wavelengths, spectra

# Test
if __name__ == "__main__":
    print("=== Waveguide Data Generator Test===\n")

    generator = WaveguideDataGenerator(seed=42)

    # Test 1: Generate different quality grids
    print("1. Generating sample datasets...")

    quality_grids = ['excellent', 'good', 'defect', 'poor']
    saved_good_spectra = None  # Save for integrity check later
    for quality in quality_grids:
        print(f"\nGenerating {quality} quality waveguide...")
        positions, spectra, wavelengths = generator.generate_grid(grid_size=(5, 5), spatial_extent=(10.0, 10.0), quality=quality)

        print(f"    Grid: {positions.shape[0]} positions")
        print(f"    Spectrum: {len(wavelengths)} wavelengths ({wavelengths[0]:.0f}-{wavelengths[-1]:.0f}nm)")
        print(f"    Mean Transmission: {np.mean(spectra):.3f}")
        print(f"    Std Transmission: {np.std(spectra):.3f}")

        # Save to file
        metadata = {
            "quality": quality,
            "grid_size": '5x5',
            "spatial_extent": '10x10 mm',
            "wavelength_range": f'{wavelengths[0]:.0f}-{wavelengths[-1]:.0f} nm',
            "generated": 'synthetic'
        }

        filename = f"waveguide_{quality}_5x5.csv"
        generator.save_to_csv(positions, spectra, generator.wavelengths, filename, metadata)
        
        # Save good quality data for integrity check
        if quality == 'good':
            saved_good_spectra = spectra.copy()

    # Test 2: Generate higher resolution
    print("\n2. Generating higher resolution dataset (10x10 grid)...")
    positions, spectra, wavelengths = generator.generate_grid(grid_size=(10, 10), spatial_extent=(10.0, 10.0), quality='good')
    print(f"    Generated {len(positions)} measurements")

    # Test 3: Generate dark and reference
    print("\n3. Generating calibration spectra...")
    dark = generator.generate_dark_spectrum()
    reference = generator.generate_reference_spectrum()
    print(f"    Dark mean: {np.mean(dark):.4f}")
    print(f"Reference mean: {np.mean(reference):.4f}")

    # Test 4: Test loading
    print("\n4. Testing loading CSV...")
    loaded_pos, loaded_wl, loaded_spectra = generator.load_from_csv("waveguide_good_5x5.csv")
    print(f"    Loaded {len(loaded_pos)} positions")
    print(f"    Data integrity check: {'PASS' if np.allclose(loaded_spectra, saved_good_spectra) else 'FAIL'}")

    print("\n=== End of test ===")
    print("\nGenerated files")
    print(" - waveguide_excellent_5x5.csv")
    print(" - waveguide_good_5x5.csv")
    print(" - waveguide_defect_5x5.csv")
    print(" - waveguide_poor_5x5.csv")
