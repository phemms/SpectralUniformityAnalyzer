"""
end_to_end_analysis.py
This script performs end-to-end colour uniformity analysis on waveguide data.

Demonstrates the full workflow:
1. Generate/load data
2. Apply corrections
3. Analyze uniformity
4. Check Specifications
5. Generate visualizations
6. Export results

This quality control workflow applies colorimetry expertise from display 
characterization research to photonic device manufacturing.

Reference: "Colour Characterisation of a LCD and Mobile Display Using 
Polynomial and Masking Models" (Mohamed, Balogun, Das, 2017) - See lab project 2.pdf

Author: Olufemi Balogun
Date: October 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import our modules
from data_generator import WaveguideDataGenerator
from color_converter import ColorConverter
from uniformity_analyzer import UniformityAnalyzer, MeasurementMetadata, SpecLimits
from visualizer import UniformityVisualizer

def run_end_to_end_analysis(quality: str = 'good', grid_size: tuple = (5, 5), show_plots: bool = True):
    """
    Run end to end uniformity analysis workflow
    :param quality: 'excellent', 'good', 'defect', 'poor'
    :param grid_size: (nx, ny) measurement grid
    :param show_plots: to display or not display plots
    """
    print("="*70)
    print("SPECTRAL UNIFORMITY ANALYSIS - COMPLETE END-TO-END WORKFLOW")
    print("="*70)
    print(f"\nAnalysing {quality} quality waveguide ({grid_size[0]}×{grid_size[1]} grid)\n")

    # ============================================================================
    # Step 1: Generate/load data
    # ============================================================================
    print("STEP 1: Data Generation")
    print("-"*70)

    generator = WaveguideDataGenerator(seed=42)

    # Generate measurements
    positions, spectra, wavelengths = generator.generate_grid(grid_size=grid_size, spatial_extent=(10.0, 10.0), quality=quality)

    # Generate dark spectrum for correction
    dark = generator.generate_dark_spectrum()

    print(f"    Generated {len(positions)} measurements")
    print(f"    Wavelength range: {wavelengths[0]:.0f}-{wavelengths[-1]:.0f}nm")
    print(f"    Spectral points: {len(wavelengths)}")
    print(f"    Mean transmission: {np.mean(spectra):.3f}")

    # Save to file
    metadata_csv = {
        "quality": quality,
        "grid_size": f'{grid_size[0]}x{grid_size[1]}',
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    csv_filename = f"waveguide_{quality}_{grid_size[0]}x{grid_size[1]}.csv"
    generator.save_to_csv(positions, spectra, wavelengths, csv_filename, metadata_csv)
    print(f"    Data saved to {csv_filename}")

    # ===========================================================================
    # Step 2: SETUP ANALYZER WITH METADATA
    # ===========================================================================
    print("\nSTEP 2: Setup Analyzer")
    print("-" * 70)


    # Create metadata object
    metadata = MeasurementMetadata(
        operator="Olufemi Balogun",
        instrument_model="Minolta CS-2000",
        instrument_serial="Cs2000-12345",
        calibration_date="2025-10-15",
        calibration_certificate="CAL-2025-10-001",
        device_serial=f"WAVEGUIDE-{quality.upper()}-001",
        temperature_c=23.5,
        humidity_percent=45.0,
        ambient_light_lux=0.3,
        notes=f"Simulated {quality} quality waveguide for testing"
    )

    # Define specification limits
    spec_limits = SpecLimits(
        max_deltaE_mean=3.0,        # Mean deltaE across surface
        max_deltaE_any=5.0,         # Max deltaE at any point
        max_deltaE_std=2.0,         # Uniformity requirement
        chromaticity_tolerance=0.01  # Max xy deviation
    )

    # Initialize analyzer
    analyzer = UniformityAnalyzer(metadata=metadata, spec_limits=spec_limits)

    print("    Analyzer configured with device metadata")
    print(f"    Operator: {metadata.operator}")
    print(f"    Instrument: {metadata.instrument_model} (SN: {metadata.instrument_serial})")
    print(f"    Device: {metadata.device_serial}")
    print(f"    Environment: {metadata.temperature_c}°C, {metadata.humidity_percent}% RH")

    # ===========================================================================
    # Step 3: LOAD AND PROCESS DATA
    # ===========================================================================
    print("\nSTEP 3: Data Processing")
    print("-" * 70)

    # Load measurements with black correction
    grid_data = analyzer.load_grid_measurement(
        positions=positions,
        spectra=spectra,
        wavelengths=wavelengths,
        dark_spectrum=dark
    )

    print("    Measurements loaded and processed")
    print(f"    Black correction applied")
    print(f"    Converted to XYZ tristimulus values")
    print(f"    Converted to CIELAB color space")
    print(f"    Chromaticity coordinates computed")

    # ===========================================================================
    # STEP 4: ASSESS REPEATABILITY (Optional but recommended)
    # ===========================================================================
    print("\nSTEP 4: Assessment of Repeatability")
    print("-" * 70)

    # Simulate 10 repeat measurements at first position
    repeat_measurements = [
        spectra[0] + np.random.normal(0, 0.003, len(wavelengths))
        for _ in range(10)
    ]

    repeatability = analyzer.assess_repeatability(
        repeat_measurements=repeat_measurements,
        wavelengths=wavelengths,
        position_index=0
    )

    print(f"Repeatability assessed (n={repeatability['n_repeats']})")
    print(f"    DeltaE std: {repeatability['deltaE_std']:.3f}")
    print(f"    L* std: {repeatability['L_std']:.3f}")
    print(f"    Interpretation: ", end="")
    if repeatability['deltaE_std'] < 0.1:
        print("Excellent (< 0.1)")
    elif repeatability['deltaE_std'] < 0.5:
        print("Good (< 0.5)")
    else:
        print("Poor (>= 0.5) - Check instrument")

    # ===========================================================================
    # STEP 5: CALCULATE UNIFORMITY METRICS
    # ===========================================================================
    print("\nSTEP 5: Uniformity Analysis")
    print("-" * 70)

    metrics = analyzer.calculate_uniformity_metrics(reference_mode='first')

    print("✓ Uniformity metrics calculated")
    print(f"\n  ΔE Statistics:")
    print(f"    Mean:   {metrics['deltaE_mean']:.2f}")
    print(f"    Median: {metrics['deltaE_median']:.2f}")
    print(f"    Std:    {metrics['deltaE_std']:.2f}")
    print(f"    Min:    {metrics['deltaE_min']:.2f}")
    print(f"    Max:    {metrics['deltaE_max']:.2f}")

    print(f"\n  Lab Statistics:")
    print(f"    L* (lightness):  {metrics['Lab_mean']:.1f} ± {metrics['Lab_std']:.1f}")
    print(f"    a* (red-green):  {metrics['a_mean']:.2f} ± {metrics['a_std']:.2f}")
    print(f"    b* (yellow-blue): {metrics['b_mean']:.2f} ± {metrics['b_std']:.2f}")

    print(f"\n  Chromaticity:")
    print(f"    Mean deviation: {metrics['xy_deviation_mean']:.4f}")
    print(f"    Max deviation:  {metrics['xy_deviation_max']:.4f}")

    print(f"\n  Worst Position:")
    worst_pos = metrics['max_deltaE_position']
    print(f"    Location: ({worst_pos[0]:.1f}, {worst_pos[1]:.1f}) mm")
    print(f"    ΔE: {metrics['deltaE_max']:.2f}")

    # ===========================================================================
    # Step 6: QC SPECIFICATION CHECK
    # ===========================================================================
    print("\nSTEP 6: QC Specification Check")
    print("-" * 70)

    qc_results = analyzer.check_specifications()

    if qc_results['pass']:
        print("✓ RESULT: PASS")
        print("  Device meets all specifications ")
    else:
        print("✗ RESULT: FAIL")
        print("\n   Failures:")
        for failure in qc_results['failures']:
            print(f"    - {failure}")

    if qc_results['warnings']:
        print("\n   Warnings:")
        for warning in qc_results['warnings']:
            print(f"    - {warning}")

    print(f"\n  Specifications Limits:")
    print(f"    Mean deltaE: < {spec_limits.max_deltaE_mean}")
    print(f"    Max deltaE: < {spec_limits.max_deltaE_any}")
    print(f"    deltaE std: < {spec_limits.max_deltaE_std}")

    # ===========================================================================
    # Step 7: GENERATE VISUALIZATION
    # ===========================================================================
    print("\nSTEP 7: Visualization Generation")
    print("-" * 70)

    viz = UniformityVisualizer()

    # 7a. Uniformity heatmap
    print("  Generating uniformity heatmap...")
    interpolated_data = analyzer.interpolate_uniformity_map(grid_resolution=(50, 50), parameter='deltaE')

    fig1 = viz.plot_uniformity_heatmap(interpolated_data, spec_limit=spec_limits.max_deltaE_any, title=f'Colour Uniformity Map - {quality.capitalize()} Quality', save_path=f'uniformity_heatmap_{quality}.png')

    # 7b. DeltaE distribution
    print("  Generating deltaE distribution...")
    spec_dict = {'acceptable': spec_limits.max_deltaE_mean, 'marginal': spec_limits.max_deltaE_any}

    fig2 = viz.plot_deltaE_distribution(metrics['deltaE_values'], spec_limits=spec_dict, title=f'DeltaE Distribution - {quality.capitalize()} Quality', save_path=f'deltaE_distribution_{quality}.png')

    # 7c. Chromaticity diagram
    print("  Creating chromaticity diagram...")
    fig3 = viz.plot_chromaticity_diagram(grid_data['xy_values'], positions=positions, title=f"Chromaticity Uniformity - {quality.capitalize()} Quality", save_path = f'chromaticity_diagram_{quality}.png')

    # 7d. Spectral plots
    print("  Creating spectral plots...")
    fig4 = viz.plot_spectral_profile(wavelengths, spectra, positions=positions, title=f"Spectral Measurements - {quality.capitalize()} Quality", save_path=f'spectra_{quality}.png')

    # 7e. Complete QC report figure
    print("  Creating QC report...")
    fig = viz.create_qc_report_figure(metrics, qc_results, grid_data, save_path=f'qc_report_{quality}.png')

    print("✓ All visualization generated")

    # ===========================================================================
    # Step 8: SUMMARY
    # ===========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(f"\nDevice: {metadata.device_serial}")
    print(f"Quality: {quality.capitalize()}")
    print(f"Result: {'PASS' if qc_results['pass'] else 'FAIL'}")
    print(f"Mean deltaE: {metrics['deltaE_mean']:.2f} (limit: {spec_limits.max_deltaE_mean})")
    print(f"Max deltaE: {metrics['deltaE_max']:.2f} (limit: {spec_limits.max_deltaE_any})")

    print(f"\nGenerated files:")
    print(f"  {csv_filename}")
    print(f"  uniformity_heatmap_{quality}.png")
    print(f"  deltaE_distribution_{quality}.png")
    print(f"  chromaticity_diagram_{quality}.png")
    print(f"  spectra_{quality}.png")
    print(f"  qc_report_{quality}.png")

    if show_plots:
        plt.show()

    return qc_results, metrics, grid_data

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SPECTRAL UNIFORMITY ANALYSIS - DEMONSTRATION")
    print("=" * 70)
    print("\nThis script demonstrates a complete QC workflow")
    print("for optical waveguide colour uniformity analysis.\n")

    # Run analysis for different quality levels

    qualities = ['excellent', 'good', 'defect']

    for quality in qualities:
        print(f"\n" + "=" * 70)
        print(f"ANALYSIS FOR {quality.upper()} QUALITY")
        print(f"{'=' * 70}\n")

        results, metrics, data = run_end_to_end_analysis(quality=quality, grid_size=(5, 5), show_plots=False)  # set to True to show plots

        input(f"\nPress Enter to continue to next analysis...")

    print("\n" + "=" * 70)
    print("ALL ANALYSIS COMPLETE")
    print("=" * 70)