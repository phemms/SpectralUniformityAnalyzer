# Spectral Uniformity Analyzer

**Color uniformity analysis for optical waveguides and photonic devices**

A Python toolkit for analyzing spectral uniformity using CIE colorimetry standards (CIE 1931, CIELAB, CIEDE2000). Implements QC workflows with spatial analysis, pass/fail criteria, and measurement traceability.

---

## 🎯 Project Overview

This tool analyzes color uniformity across optical devices by converting spectral measurements to perceptually-uniform color spaces and calculating spatial variations. Built for QC of waveguides, displays, and other photonic components.

### Key Features

- **CIE 1931 Color Science** - Color matching functions, XYZ tristimulus values.
- **Colorimetry** - CIELAB color space, CIEDE2000 color difference calculation.
- **Spatial Analysis** - 2D uniformity mapping with interpolation.
- **QC System** - Pass/fail criteria with failure reporting.
- **Traceability** - Operator, instrument info, calibration data, environmental conditions.
- **Visualization** - Chromaticity diagrams, heatmaps, histograms, spectral profiles.

---

## 🖼️ Example Outputs

The tool generates 5 types of visualizations:

1. **Uniformity Heatmap** - 2D spatial distribution of color differences.
2. **QC Report** - Single-page summary with spatial map, spectra, statistics, and pass/fail.
3. **Chromaticity Diagram** - CIE 1931 diagram showing color distribution.
4. **ΔE Distribution** - Histogram and box plot.
5. **Spectral Profiles** - Wavelength measurements with mean and standard deviation bands.

Run `python end_to_end_analysis.py` to generate examples.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/phemms/SpectralUniformityAnalyzer.git
cd SpectralUniformityAnalyzer

# Create virtual environment (Python 3.9+, tested on 3.13)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Complete Demo

```bash
python end_to_end_analysis.py
```

This runs analysis on 3 quality levels (excellent, good, defect):
- 3 CSV data files with synthetic waveguide measurements.
- 15 visualization plots (5 types × 3 quality levels).
- QC reports with pass/fail decisions.
- Statistical analysis.

**Runtime:** ~30 seconds.

---

## 📊 Usage Examples

### Analyze Synthetic Waveguide Data

```python
from data_generator import WaveguideDataGenerator
from uniformity_analyzer import UniformityAnalyzer, MeasurementMetadata, SpecLimits
from visualizer import UniformityVisualizer

# Generate test data
generator = WaveguideDataGenerator(seed=42)
positions, spectra, wavelengths = generator.generate_grid(
    grid_size=(5, 5), 
    spatial_extent=(10.0, 10.0), 
    quality='good'
)

# Setup analyzer with metadata
metadata = MeasurementMetadata(
    operator="John Doe",
    instrument_model="Minolta CS-2000",
    instrument_serial="CS2000-12345"
)

spec_limits = SpecLimits(
    max_deltaE_mean=3.0,
    max_deltaE_any=5.0,
    max_deltaE_std=2.0
)

analyzer = UniformityAnalyzer(metadata=metadata, spec_limits=spec_limits)

# Process and analyze
grid_data = analyzer.load_grid_measurement(positions, wavelengths, spectra)
metrics = analyzer.calculate_uniformity_metrics(reference_mode='first')
qc_results = analyzer.check_specifications()

# Generate visualizations
viz = UniformityVisualizer()
interpolated_data = analyzer.interpolate_uniformity_map(parameter='deltaE')
fig = viz.plot_uniformity_heatmap(interpolated_data, save_path='uniformity_map.png')

print(f"Result: {'PASS' if qc_results['pass'] else 'FAIL'}")
print(f"Mean ΔE: {metrics['deltaE_mean']:.2f}")
```

### Output

#### CIEDE2000 Results (Current Implementation)
```
Result: PASS
Mean ΔE*00: 0.51
Max ΔE*00: 1.33 at position (7.5, 2.5) mm
Std ΔE*00: 0.31
```

#### Comparison: ΔE*ab vs CIEDE2000
Measured on synthetic waveguide data with corrected CIE D65 illuminant (November 2025):

| Quality | Metric | ΔE*ab | CIEDE2000 |
|---------|--------|-------|-----------|
| **Excellent** | Mean | 0.07 | 0.08 |
| | Max | 0.14 | 0.18 |
| **Good** | Mean | 0.76 | 0.51 |
| | Max | 2.11 | 1.33 |
| **Defect** | Mean | 1.87 | 1.18 |
| | Max | 10.79 | 6.49 |

**Result:** CIEDE2000 produces lower ΔE values (58-66% lower on defects). Both metrics agree on pass/fail but CIEDE2000 better represents perceptual uniformity.

#### Why CIEDE2000?
- **Industry Standard**: Used by display manufacturers and AR/VR companies
- **Better Blue Accuracy**: Critical for AR waveguides (450-480nm range)
- **Perceptual Uniformity**: Equal ΔE = equal perceived color difference
- **ISO Compliance**: Meets international colorimetry standards

The tool now uses CIEDE2000 for production QC applications.

### Analyze Your Own Measurement Data

For real QC applications, load your own spectral measurement data:

#### 1. Format Your Data as CSV

Your CSV file must have these columns:
- `position_x_mm`, `position_y_mm` - Measurement coordinates in millimeters
- `wl_XXX.XXnm` - Spectral data columns (wavelength in nm, e.g., `wl_400.00nm`)

```csv
position_x_mm,position_y_mm,wl_400.00nm,wl_500.00nm,wl_600.00nm,wl_700.00nm
0.0,0.0,0.85,0.90,0.88,0.82
1.0,0.0,0.86,0.89,0.87,0.83
0.0,1.0,0.84,0.91,0.89,0.81
1.0,1.0,0.85,0.90,0.88,0.82
```

#### 2. Configure the Tool

```python
# In end_to_end_analysis.py, change these settings:
DATA_SOURCE = "file"  # Instead of "synthetic"
CUSTOM_DATA_FILE = "path/to/your/measurement_data.csv"  # Can be absolute or relative path
```

**Path Examples:**
```python
CUSTOM_DATA_FILE = "/Users/john/lab_data/waveguide_measurements.csv"  # Absolute path
CUSTOM_DATA_FILE = "../spectrometer_data/session_2025_01_15.csv"       # Relative path
CUSTOM_DATA_FILE = "C:/spectrometer_data/output.csv"                   # Windows path
CUSTOM_DATA_FILE = "sample_data/example_custom_data.csv"               # Test with included example
```

#### 3. Run Analysis

```bash
python end_to_end_analysis.py
```

The tool will load your data and perform the same uniformity analysis, generating visualizations and reports based on your measurements.

**Quick Test:** Run `python test_with_example_data.py` to try the analysis with the included example data.

**Example file:** See `sample_data/example_custom_data.csv` for a complete 5×5 grid example.

### Testing with Your Data (For Recruiters/Evaluators)

To evaluate the tool with your own measurement data:

#### 1. Prepare Your CSV File
Format your spectral measurements with the required columns (see format above).

#### 2. Configure the Tool
Edit `end_to_end_analysis.py` and change:
```python
DATA_SOURCE = "file"  # Change from "synthetic"
CUSTOM_DATA_FILE = "/path/to/your/data.csv"  # Set your file path
```

#### 3. Run the Analysis
```bash
python end_to_end_analysis.py
```

#### 4. Evaluate Results
The tool will:
- Load your measurement data
- Perform CIEDE2000 uniformity analysis
- Generate QC reports and visualizations
- Display PASS/FAIL results

#### Expected Output:
```
STEP 1: Data Generation/Loading
    ✓ Loaded X measurement positions
    ✓ Wavelength range: XXX - XXX nm

[... analysis steps ...]

RESULT: PASS/FAIL
Generated files:
  Reports: output_reports/analysis_report_*.csv, *.pdf
  Visualizations: images_cie2000/*.png
```

#### Validation Checklist:
- [ ] Data loads without errors
- [ ] ΔE values are reasonable (0.1-10 range typical)
- [ ] Visualizations show expected spatial patterns
- [ ] Reports contain complete metadata and measurements
- [ ] QC pass/fail logic works as expected

---

## 🧪 Technical Details

### Color Science Implementation

- **CIE 1931 2° Standard Observer** - Interpolated to 1nm resolution (380-780nm).
- **Color Matching Functions** - x̄(λ), ȳ(λ), z̄(λ) values.
- **XYZ Tristimulus Values** - Numerical integration via trapezoidal rule.
- **CIELAB Color Space** - Perceptually uniform with white point normalization.
- **ΔE Metrics**:
  - ΔE*ab - Euclidean distance in Lab space.
  - CIEDE2000 (ΔE00) - Industry standard with lightness, chroma, and hue weighting.

### Synthetic Data Generator

Simulates realistic waveguide defects:
- Center-to-edge uniformity gradients.
- Localized defects (contamination, scratches).
- Wavelength shifts (thickness variations).
- Interference fringes (thin-film effects).
- Measurement noise (Gaussian).

### Quality Control Metrics

**Pass/Fail Criteria:**
- Mean ΔE across surface.
- Maximum ΔE at any point.
- Standard deviation (uniformity requirement).
- Chromaticity tolerance in xy space.
- Optional lightness (L*) range limits.

**Traceability:**
- Operator name.
- Instrument model and serial number.
- Calibration date and certificate number.
- Environmental conditions (temperature, humidity).
- Timestamps (ISO format).

---

## 📁 Project Structure

```
SpectralUniformityAnalyzer/
├── color_converter.py        # CIE color space conversions
├── data_generator.py          # Synthetic waveguide data generator
├── uniformity_analyzer.py     # QC analysis engine
├── visualizer.py              # Visualization suite
├── end_to_end_analysis.py     # Complete workflow demo
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🔬 Background

This project applies colorimetry to manufacturing quality control. I built it to translate color science knowledge into practical QC tools for optical devices.

The methodology comes from working with display color characterization back in 2017, where I used polynomial and masking models for LCD and AMOLED devices. That work focused on how screens reproduce colors. This project takes those same colorimetry principles and applies them to waveguide uniformity analysis - checking if optical components have consistent color transmission across their surface.

The implementation uses CIE 1931 color standards and CIEDE2000 color difference calculations, which are industry standards for quantifying color variations that humans can perceive.

**Reference:** The 2017 display work is documented in *"Colour Characterisation of a LCD and Mobile Display Using Polynomial and Masking Models"* (Mohamed, Balogun, Das 2017).

---

## 🛠️ Requirements

- **Python:** 3.9 or higher (developed and tested on 3.13)
- **Dependencies:**
  - numpy >= 1.21.0
  - scipy >= 1.7.0
  - matplotlib >= 3.4.0
  - pandas >= 1.3.0
  - seaborn >= 0.11.0

See `requirements.txt` for complete list.

---

## 📈 Performance

- **Analysis time:** ~1-2 seconds per device (25 measurement points).
- **Repeatability:** ΔE std < 0.03 (excellent instrument precision).
- **Scalability:** Tested up to 100 measurement points (10×10 grid).

---

## 🎓 Use Cases

- **Manufacturing QC:** Waveguide, LED, display uniformity testing.
- **Research:** Spatial color characterization of photonic devices.
- **Calibration:** Display color accuracy verification.
- **Education:** Teaching CIE color science and colorimetry.

---

## 📝 License

MIT License - See LICENSE file for details.

---

## 👤 Author

**Olufemi Balogun**

- 📧 Email: phemmsbalo@gmail.com.

---

## 🙏 Acknowledgments

- CIE color matching functions and standards from CIE publications.
- CIEDE2000 implementation based on published specifications.
- Methodology builds on display color characterization work from 2017.

---

## 📊 Example Results (CIEDE2000)

### Excellent Quality Device ✅
```
Mean ΔE*00: 0.08  (Limit: 3.0) ✓
Max ΔE*00:  0.18  (Limit: 5.0) ✓
Std ΔE*00:  0.05  (Limit: 2.0) ✓
Result: PASS
```

### Good Quality Device ✅
```
Mean ΔE*00: 0.51  (Limit: 3.0) ✓
Max ΔE*00:  1.33  (Limit: 5.0) ✓
Std ΔE*00:  0.31  (Limit: 2.0) ✓
Result: PASS
```

### Defect Quality Device ❌
```
Mean ΔE*00: 1.18  (Limit: 3.0) ✓
Max ΔE*00:  6.49  (Limit: 5.0) ✗ FAIL
Std ΔE*00:  1.24  (Limit: 2.0) ✓
Result: FAIL - Defect detected at position (7.5, 2.5) mm
```

*Results shown are from CIEDE2000 analysis. Sample outputs are available in `images_cie2000/` and `output_reports/` folders.*

---

*Built with Python, NumPy, SciPy, and Matplotlib.*

