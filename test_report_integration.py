"""
test_report_integration.py
Quick test to verify report generation integration

This demonstrates the report generation capability without running
the full end-to-end analysis pipeline.

Author: Olufemi Balogun
Date: November 2025
"""

import numpy as np
from report_generator import UniformityReportGenerator

# Mock data that would come from uniformity_analyzer
mock_metrics = {
    'deltaE_mean': 2.15,
    'deltaE_median': 1.98,
    'deltaE_std': 1.12,
    'deltaE_min': 0.23,
    'deltaE_max': 4.87,
    'deltaE_values': np.random.uniform(0, 5, 25),
    'Lab_mean': 87.2,
    'Lab_std': 1.8,
    'a_mean': -1.85,
    'a_std': 0.95,
    'b_mean': 6.32,
    'b_std': 1.45,
    'xy_deviation_mean': 0.0038,
    'xy_deviation_max': 0.0085,
    'max_deltaE_position': (7.5, 5.0),
}

mock_qc_results = {
    'pass': True,
    'failures': [],
    'warnings': ['ΔE approaching specification limit at position (7.5, 5.0)'],
}

mock_metadata = {
    'device_serial': 'WAVEGUIDE-EXCELLENT-001',
    'operator': 'Olufemi Balogun',
    'instrument_model': 'Minolta CS-2000',
    'instrument_serial': 'CS2000-12345',
    'calibration_date': '2025-10-15',
    'temperature_c': 23.5,
    'humidity_percent': 45.0,
    'ambient_light_lux': 0.3,
    'notes': 'Demonstration of report generation capability for uniformity analysis',
}

mock_spec_limits = {
    'max_deltaE_mean': 3.0,
    'max_deltaE_any': 5.0,
    'max_deltaE_std': 2.0,
    'chromaticity_tolerance': 0.01,
}

print("=" * 70)
print("REPORT GENERATION INTEGRATION TEST")
print("=" * 70)
print("\nTesting report generation with mock uniformity analysis data...\n")

# Initialize report generator
report_gen = UniformityReportGenerator(
    company_name="Photonics QC Lab",
    author="Olufemi Balogun"
)

# Test CSV report
print("1. Generating CSV analysis report...")
success_csv = report_gen.generate_csv_report(
    'test_integration_report.csv',
    mock_metrics,
    mock_qc_results,
    mock_metadata,
    mock_spec_limits
)

# Test PDF report (without visualizations for this test)
print("\n2. Generating PDF analysis report...")
success_pdf = report_gen.generate_pdf_report(
    'test_integration_report.pdf',
    mock_metrics,
    mock_qc_results,
    mock_metadata,
    mock_spec_limits
)

print("\n" + "=" * 70)
print("INTEGRATION TEST COMPLETE")
print("=" * 70)

if success_csv and success_pdf:
    print("\n✓ All report formats generated successfully!")
    print("\nGenerated files:")
    print("  - test_integration_report.csv")
    print("  - test_integration_report.pdf")
    print("\nThese reports can be integrated into the end_to_end_analysis.py workflow.")
else:
    print("\n✗ Some reports failed to generate. Check error messages above.")

print("\n" + "=" * 70)

