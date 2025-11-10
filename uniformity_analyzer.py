"""
uniformity_analyzer.py
Spectral uniformity analysis for optical QC

Analyzes colour uniformity across spatial measurements (e.g., waveguide surface)
Implements production-ready QC metrics with full traceability

Methodology adapted from display color characterization work:
"Colour Characterisation of a LCD and Mobile Display Using Polynomial and 
Masking Models" (Balogun, Mohamed, Das, 2017)

Author: Olufemi Balogun
Date: October 2025
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from datetime import datetime
from scipy import interpolate
from color_converter import ColorConverter
from dataclasses import dataclass, field


@dataclass
class MeasurementMetadata:
    """
    Traceable metadata for QC
    """
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    operator: str = "Unknown"
    instrument_model: str = "Unknown"
    instrument_serial: str = "Unknown"
    calibration_date: str = "Unknown"
    calibration_certificate: str = "Unknown"
    device_serial: str = "Unknown"
    temperature_c: float = None
    humidity_percent: float = None
    ambient_light_lux: float = None
    software_version: str = "1.0"
    notes: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for export"""
        return {
            'timestamp': self.timestamp,
            'operator': self.operator,
            'instrument': f"{self.instrument_model} (SN: {self.instrument_serial})",
            'calibration': f"{self.calibration_date} (Cert: {self.calibration_certificate})",
            'device_serial': f"{self.device_serial}",
            'environment': {
                'temperature_C': self.temperature_c,
                'humidity': self.humidity_percent,
                'ambient_lux': self.ambient_light_lux
            },
            'software': self.software_version,
            'notes': self.notes
        }

@dataclass
class SpecLimits:
    """
    QC specification limits
    pass/fail criteria
    """
    max_deltaE_mean: float = 3.0    # Average deltaE across surface
    max_deltaE_any: float = 5.0     # Maximum deltaE at any point
    max_deltaE_std: float = 2.0     # Standard deviation of deltaE
    max_L_star: float = None         # Maximum lightness
    min_L_star: float = None         # Minimum lightness
    chromaticity_tolerance: float = 0.01 # Max xy deviation

    def to_dict(self) -> Dict:
        """Convert to dictionary for export"""
        return {
            'max_deltaE_mean': self.max_deltaE_mean,
            'max_deltaE_any': self.max_deltaE_any,
            'max_deltaE_std': self.max_deltaE_std,
            'max_L_star': self.max_L_star,
            'min_L_star': self.min_L_star,
            'chromaticity_tolerance': self.chromaticity_tolerance
        }

class UniformityAnalyzer:
    """
    Main class for spectral uniformity analysis from grid measurements
    Features:
    - Full traceability metadata
    - Measurement uncertainty quantification
    - Pass/fail analysis with justification
    - Spatial uniformity metrics
    - Repeatability testing
    """

    def __init__(self, metadata: Optional[MeasurementMetadata] = None, spec_limits: Optional[SpecLimits] = None, metric_type: str = "cie2000"):
        """
        Initialize the analyzer
        :param metadata:
        :param spec_limits:
        :param metric_type: Color difference metric: "deltaEab" or "cie2000"
        """
        self.converter = ColorConverter()
        self.metadata = metadata if metadata else MeasurementMetadata()
        self.spec_limits = spec_limits if spec_limits else SpecLimits()
        self.metric_type = metric_type

        # Storage for analysis results
        self.grid_data = None
        self.analysis_results = None
        self.repeatability_data = None

    def load_grid_measurement(self,
                              positions: np.ndarray,
                              wavelengths: np.ndarray,
                              spectra: np.ndarray,
                              dark_spectrum: Optional[np.ndarray] = None) -> Dict:
        """
        Load and preprocess grid measurement
        :param positions: Array of shape (N, 2) with [x, y] coordinates
        :param wavelengths: Wavelength array
        :param spectra: Array of shape (N, M) where N=positions, M=wavelengths
        :param dark_spectrum: Optional dark spectrum for correction
        :return: Dictionary with processed data
        """

        n_positions = positions.shape[0]

        # Apply dark correction if provided
        if dark_spectrum is not None:
            spectra_corrected = np.array([
                self.converter.black_correction(spectra[i], dark_spectrum)
                for i in range(n_positions)
            ])
        else:
            spectra_corrected = spectra

        # Convert all spectra to XYZ
        XYZ_values = np.array([
            self.converter.spectrum_to_xyz(wavelengths, spectra_corrected[i], mode='radiance')
            for i in range(n_positions)
        ])

        # Convert to Lab ( use first measurement as white point reference)
        white_point = XYZ_values[0] # Or could use centre position
        Lab_values = np.array([
            self.converter.XYZ_to_Lab(XYZ_values[i], white_point)
            for i in range(n_positions)
        ])

        # Convert to chromaticity
        xy_values = np.array([
            self.converter.XYZ_to_xy(XYZ_values[i])
            for i in range(n_positions)
        ])

        self.grid_data = {
            'positions': positions,
            'wavelengths': wavelengths,
            'spectra': spectra_corrected,
            'XYZ_values': XYZ_values,
            'Lab_values': Lab_values,
            'xy_values': xy_values,
            'reference_position': 0, # Index of reference measurement
            'white_point': white_point
        }

        return self.grid_data

    def calculate_uniformity_metrics(self, reference_mode: str = 'first') -> Dict:
        """
        Calculate spatial uniformity metrics
        :param reference_mode: 'first', 'center', 'white' or 'mean'
                    - 'first': Use first measurement as reference
                    - 'center': Use center position as reference
                    - 'white': Use theoretical white (L*=100, a*=0, b*=0)
                    - 'mean': Use mean Lab as reference
        :return: Dictionary with uniformity metrics
        """
        if self.grid_data is None:
            raise ValueError("No grid data loaded. Please load grid data first.")

        Lab_values = self.grid_data['Lab_values']
        positions = self.grid_data['positions']
        xy_values = self.grid_data['xy_values']

        # Determine reference
        if reference_mode == 'first':
            Lab_ref = Lab_values[0]
        elif reference_mode == 'center':
            # Find position closest to center
            center = np.mean(positions, axis=0)
            distances = np.linalg.norm(positions - center, axis=1)
            centre_idx  = np.argmin(distances)
            Lab_ref = Lab_values[centre_idx]
        elif reference_mode == 'white':
            Lab_ref = np.array([100.0, 0.0, 0.0])
        else:
            Lab_ref = np.mean(Lab_values, axis=0)

        # Calculate color difference based on metric type
        if self.metric_type == "deltaEab":
            deltaE_values = np.array([
                self.converter.calculate_deltaE_ab(Lab_values[i], Lab_ref)
                for i in range(len(Lab_values))
            ])
        else:  # "cie2000" or default
            deltaE_values = np.array([
                self.converter.calculate_deltaE_00(Lab_values[i], Lab_ref)
                for i in range(len(Lab_values))
            ])

        # For backward compatibility, also store as deltaE00_values
        deltaE00_values = deltaE_values

        # Calculate chromatic deviations
        xy_ref = self.converter.XYZ_to_xy(self.grid_data['XYZ_values'][0])
        xy_deviations = np.array([
            np.linalg.norm(xy_values[i] - xy_ref)
            for i in range(len(xy_values))
        ])

        # Statistical metrics
        metrics = {
            # CIEDE2000 statistics (primary metric)
            'deltaE_mean': np.mean(deltaE00_values),
            'deltaE_std': np.std(deltaE00_values),
            'deltaE_min': np.min(deltaE00_values),
            'deltaE_max': np.max(deltaE00_values),
            'deltaE_median': np.median(deltaE00_values),
            'deltaE_values': deltaE00_values,


            # deltaE00 statistics (duplicate for backward compatibility)
            'deltaE00_mean': np.mean(deltaE00_values),
            'deltaE00_std': np.std(deltaE00_values),
            'deltaE00_max': np.max(deltaE00_values),
            'deltaE00_values': deltaE00_values,

            # Lab statistics
            'Lab_mean': np.mean(Lab_values[:, 0]),
            'Lab_std': np.std(Lab_values[:, 0]),
            'Lab_min': np.min(Lab_values[:, 0]),
            'Lab_max': np.max(Lab_values[:, 0]),

            'a_mean': np.mean(Lab_values[:, 1]),
            'a_std': np.std(Lab_values[:, 1]),

            'b_mean': np.mean(Lab_values[:, 2]),
            'b_std': np.std(Lab_values[:, 2]),

            # Chromaticity statistics
            'xy_deviation_mean': np.mean(xy_deviations),
            'xy_deviation_max': np.max(xy_deviations),
            'xy_deviations': xy_deviations,

            # Reference used
            'reference_mode': reference_mode,
            'reference_Lab': Lab_ref,

            # Positions of extremes
            'max_deltaE_position': positions[np.argmax(deltaE_values)],
            'min_deltaE_position': positions[np.argmin(deltaE_values)]
        }

        self.analysis_results = metrics
        return metrics

    def check_specifications(self) -> Dict:
        """
        Check measurements against QC specification
        :return: Dictionary with pass/fail results and details
        """
        if self.analysis_results is None:
            raise ValueError("No analysis results available. Please run calculate_uniformity_metrics first.")

        metrics = self.analysis_results
        limits = self.spec_limits

        failures = []
        warnings = []

        # Check mean deltaE
        if metrics['deltaE_mean'] > limits.max_deltaE_mean:
            failures.append(f"Mean deltaE ({metrics['deltaE_mean']:.2f}) exceeds limit ({limits.max_deltaE_mean:.2f})")

        # Check max deltaE
        if metrics['deltaE_max'] > limits.max_deltaE_any:
            failures.append(f"Max deltaE ({metrics['deltaE_max']:.2f}) at position {metrics['max_deltaE_position']} "
                            f"exceeds limit ({limits.max_deltaE_any:.2f})")

        # Check deltaE std deviation (uniformity)
        if metrics['deltaE_std'] > limits.max_deltaE_std:
            failures.append(f"DeltaE std deviation ({metrics['deltaE_std']:.2f}) exceeds limit ({limits.max_deltaE_std:.2f})")

        # Check lightness range
        if limits.min_L_star is not None and metrics['Lab_min'] < limits.min_L_star:
            failures.append(f"Min lightness ({metrics['Lab_min']:.2f}) below limit ({limits.min_L_star:.2f})")

        if limits.max_L_star is not None and metrics['Lab_max'] > limits.max_L_star:
            failures.append(f"Max lightness ({metrics['Lab_max']:.2f}) above limit ({limits.max_L_star:.2f})")

        # Check chromaticity tolerance
        if metrics['xy_deviation_max'] > limits.chromaticity_tolerance:
            failures.append(f"Chromaticity deviation ({metrics['xy_deviation_max']:.4f}) "
                            f"exceeds tolerance ({limits.chromaticity_tolerance:.4f})")

        # Warnings for borderline cases
        if 0.8 * limits.max_deltaE_mean < metrics['deltaE_mean'] <= limits.max_deltaE_mean:
            warnings.append(f"Mean deltaE ({metrics['deltaE_mean']:.2f}) is close to limit ({limits.max_deltaE_mean:.2f})")

        results = {
            'pass': len(failures) == 0,
            'failures': failures,
            'warnings': warnings,
            'metrics': metrics,
            'spec_limits': limits.to_dict(),
            'metadata': self.metadata.to_dict()
        }

        return results

    def assess_repeatability(self, repeat_measurements: List[np.ndarray],
                             wavelengths: np.ndarray, position_index: int = 0) -> Dict:
        """
        Assess measurement repeatability (uncertainty quantification)
        :param repeat_measurements: List of N repeated spectral measurements at same position
        :param wavelengths: Wavelength array
        :param position_index: Which position was repeated (for labeling)
        :return: Repeatability statistics
        """
        n_repeats = len(repeat_measurements)

        # Convert all to Lab
        XYZ_repeats = [
            self.converter.spectrum_to_xyz(wavelengths, spectrum, mode='radiance')
            for spectrum in repeat_measurements
        ]

        # Use first as reference white point
        white_point = XYZ_repeats[0]

        Lab_repeats = [
            self.converter.XYZ_to_Lab(XYZ, white_point=white_point)
            for XYZ in XYZ_repeats
        ]

        # Calculate statistics
        Lab_array = np.array(Lab_repeats)
        Lab_mean = np.mean(Lab_array, axis=0)
        Lab_std = np.std(Lab_array, axis=0)

        # Calculate deltaE between each measurement and mean
        if self.metric_type == "deltaEab":
            deltaE_from_mean = [
                self.converter.calculate_deltaE_ab(Lab_repeats[i], Lab_mean)
                for i in range(n_repeats)
            ]
        else:  # "cie2000" or default
            deltaE_from_mean = [
                self.converter.calculate_deltaE_00(Lab_repeats[i], Lab_mean)
                for i in range(n_repeats)
            ]

        repeatability_stats = {
            'n_repeats': n_repeats,
            'position_index': position_index,
            'Lab_mean': Lab_mean,
            'Lab_std': Lab_std,
            'L_std': Lab_std[0],
            'a_std': Lab_std[1],
            'b_std': Lab_std[2],
            'deltaE_std': np.std(deltaE_from_mean),
            'deltaE_max': np.max(deltaE_from_mean),
            'deltaE_values': deltaE_from_mean,
            'Lab_measurements': Lab_repeats
        }

        self.repeatability_data = repeatability_stats
        return repeatability_stats

    def interpolate_uniformity_map(self, grid_resolution: Tuple[int, int] = (50, 50),
                                   parameter: str = 'deltaE') -> Dict:
        """
        Interpolate measurements to create smooth 2D uniformity map
        :param grid_resolution: (nx, ny) resolution of interpolated grid
        :param parameter: deltaE, L, a, b, xy_deviation
        :return: Dictionary with interpolated grid data
        """
        if self.grid_data is None or self.analysis_results is None:
            raise ValueError("No data available")

        positions = self.grid_data['positions']

        # Select parameter to interpolate
        if parameter == 'deltaE':
            values = self.analysis_results['deltaE_values']
        elif parameter == 'L':
            values = self.grid_data['Lab_values'][:, 0]
        elif parameter == 'a':
            values = self.grid_data['Lab_values'][:, 1]
        elif parameter == 'b':
            values = self.grid_data['Lab_values'][:, 2]
        elif parameter == 'xy_deviation':
            values = self.analysis_results['xy_deviations']
        else:
            raise ValueError(f"Invalid parameter: {parameter}")

        # Create regular grid
        x = positions[:, 0]
        y = positions[:, 1]


        xi = np.linspace(x.min(), x.max(), grid_resolution[0])
        yi = np.linspace(y.min(), y.max(), grid_resolution[1])
        Xi, Yi = np.meshgrid(xi, yi)

        # Interpolate using griddata (handles irregular grids)
        from scipy.interpolate import griddata
        Zi = griddata((x, y), values, (Xi, Yi), method='cubic')

        return {
            'X': Xi,
            'Y': Yi,
            'Z': Zi,
            'resolution': grid_resolution,
            'parameter': parameter,
            'original_positions': positions,
            'original_values': values
        }


# Test code
if __name__ == '__main__':
    print("=== Uniformity Analyzer Test ===\n")

    # Create synthetic grid data (5x5 grid)
    print("1. Creating synthetic 5x5 grid measurements...")

    # Grid positions
    x = np.linspace(0, 10, 5)
    y = np.linspace(0, 10, 5)
    X, Y = np.meshgrid(x, y)
    positions = np.column_stack([X.ravel(), Y.ravel()])

    # Synthetic spectra (greenish with spatial variation)
    wavelengths = np.linspace(400, 700, 100)
    n_positions = len(positions)

    spectra = []
    for i in range(n_positions):
        # Centre peak shifts with position
        centre = 550 + (positions[i, 0] - 5) * 2    # Shift with x-position
        width = 30 + (positions[i, 1] -5) * 0.5     # Width varies with y
        spectrum = np.exp(-((wavelengths - centre) ** 2) / (2 * width ** 2))
        # Add some noise
        spectrum += np.random.normal(0, 0.01, len(wavelengths))
        spectrum = np.clip(spectrum, 0, 1)
        spectra.append(spectrum)

    spectra = np.array(spectra)


    # Create analyzer with metadata
    print("\n2. Setting up analyzer with production metadata...")
    metadata = MeasurementMetadata(
        operator="O.Balogun",
        instrument_model="Minolta CS-2000",
        instrument_serial="CS2000-12345",
        calibration_date="2025-10-01",
        device_serial="WAVEGUIDE-TEST-001",
        temperature_c=23.5,
        humidity_percent=45.0,
        ambient_light_lux=0.5
    )


    spec_limits = SpecLimits(
        max_deltaE_mean=3.0,
        max_deltaE_any=5.0,
        max_deltaE_std=2.0
    )

    analyzer = UniformityAnalyzer(metadata=metadata, spec_limits=spec_limits)

    # Load data
    print("\n3. Loading grid measurements...")
    analyzer.load_grid_measurement(positions, wavelengths, spectra)
    print(f"    Loaded {n_positions} measurements.")

    # Calculate Uniformity
    print("\n4. Calculating uniformity metrics...")
    metrics = analyzer.calculate_uniformity_metrics(reference_mode='first')
    print(f"    Mean deltaE: {metrics['deltaE_mean']:.2f}")
    print(f"    Max deltaE: {metrics['deltaE_max']:.2f}")
    print(f"    Std deltaE: {metrics['deltaE_std']:.2f}")

    # Check specifications
    print("\n5. Checking QC specifications...")
    qc_results = analyzer.check_specifications()
    print(f"    Result: {'Pass' if qc_results['pass'] else 'Fail'}")
    if qc_results['failures']:
        print("    Failures:")
        for failure in qc_results['failures']:
            print(f"        - {failure}")


    # Test repeatability
    print("\n6. Testing repeatability (10 repeat measures)...")
    repeat_spectra = [spectra[0] + np.random.normal(0, 0.005, len(wavelengths)) for _ in range(10)]
    repeatability = analyzer.assess_repeatability(repeat_spectra, wavelengths)
    print(f"    Repeatability deltaE std: {repeatability['deltaE_std']:.3f}")
    print(f"    L* std: {repeatability['L_std']:.3f}")

    print("\n=== Test complete ===")