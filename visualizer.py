"""
visualizer.py
Visualisation tools for colour uniformity analysis

Creates plots of:
- 2D uniformity heatmaps
- Chromaticity diagrams
- Spectral profiles
- Statistical distributions

Author: Olufemi Balogun
Date: October 2025
"""

import numpy as np
from typing import Tuple, Optional, Dict
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


class UniformityVisualizer:
    """
    Creates visualisation tools for colour uniformity analysis

    Plotting with labels, colourbars and styling
    """
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize the visualizer
        :param style: Matplotlib style
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')

        #set nice defaults
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['legend.fontsize'] = 19

    def plot_uniformity_heatmap(self, interpolated_data: Dict,
                                spec_limit: Optional[float] = None,
                                title: str = 'Colour Uniformity Map',
                                cmap: str = 'RdYlGn_r',
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot a 2D uniformity heatmap
        :param interpolated_data: Dictionary from interpolate_uniformity_map()
        :param spec_limit: Optional specification limit line
        :param title: Plot title
        :param cmap: Colourmap ('RdYlGn_r' = red-yellow-green-reversed)
        :param save_path: Optional path to saved figure
        :return: Figure object
        """
        X = interpolated_data['X']
        Y = interpolated_data['Y']
        Z = interpolated_data['Z']
        parameter = interpolated_data['parameter']
        original_pos = interpolated_data['original_positions']
        original_val = interpolated_data['original_values']

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap
        im = ax.contourf(X, Y, Z, levels=20, cmap=cmap)

        # Add colourbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(self._get_parameter_label(parameter), rotation=270, labelpad=20)

        # Add original positions
        scatter = ax.scatter(original_pos[:, 0], original_pos[:, 1], c=original_val, cmap=cmap, s=100, edgecolors='black', linewidths=2, zorder=5)

        # Add specification limit contour if provided
        if spec_limit is not None:
            contour = ax.contour(X, Y, Z, levels=[spec_limit], colors='red', linestyles='--', linewidths=2)
            ax.clabel(contour, inline=True, fmt=f'Limit: {spec_limit:.2f}', fontsize=10)

        ax.set_xlabel('X Position (mm)', fontweight='bold')
        ax.set_ylabel('Y Position (mm)', fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to {save_path}")

        return fig

    def plot_chromaticity_diagram(self, xy_values: np.ndarray,
                                  positions: Optional[np.ndarray] = None,
                                  title: str = 'Chromaticity Diagram',
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot chromaticity coordinates on CIE 1931 diagram
        :param xy_values: Array of (x, y) chromaticity coordinates
        :param positions: Optional position array for colouring
        :param title: Plot title
        :param save_path: Optional path to saved figure
        :return: Figure object
        """
        fig, ax = plt.subplots(figsize=(9, 9))

        # Draw CIE 1931 horseshoe
        self._draw_cie_horseshoe(ax)

        # Plot measurement points
        if positions is not None:
            # Colour by distance from centre
            centre = np.mean(positions, axis=0)
            distances = np.linalg.norm(positions - centre, axis=1)
            scatter = ax.scatter(xy_values[:, 0], xy_values[:, 1], c=distances, cmap='viridis', s=100, edgecolors='black', linewidths=1.5, zorder=5)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Distance from Centre (mm)', rotation=270, labelpad=20)
        else:
            ax.scatter(xy_values[:, 0], xy_values[:, 1], s=100, edgecolors='black', linewidths=1.5, zorder=5, color='red', label='Measurement Points')

        # Plot mean and std ellipses
        mean_xy = np.mean(xy_values, axis=0)
        std_xy = np.std(xy_values, axis=0)

        ax.plot(mean_xy[0], mean_xy[1], 'k*', markersize=20, label=f'Mean: {mean_xy[0]:.4f}, {mean_xy[1]:.4f}', zorder=6)

        # Draw 2-sigma ellipse
        ellipse = patches.Ellipse(mean_xy, width=4*std_xy[0], height=2*std_xy[1], fill=False, edgecolor='red', linewidth=2, linestyle='--', label='2-sigma boundary', zorder=6)
        ax.add_patch(ellipse)

        ax.set_xlabel('x', fontweight='bold', fontsize=12)
        ax.set_ylabel('y', fontweight='bold', fontsize=12)
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.legend(loc='upper right')
        ax.set_xlim(0, 0.8)
        ax.set_ylim(0, 0.9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)


        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chromaticity diagram saved to {save_path}")

        return fig

    def plot_spectral_profile(self, wavelengths: np.ndarray, spectra: np.ndarray, positions: np.ndarray, selected_indices: Optional[list] = None, title: str = 'Spectral Measurements', save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot multiple spectra from grid
        :param wavelengths: Wavelength array
        :param spectra: Spectra array (N, M)
        :param positions: Position array (N, 2)
        :param selected_indices: Indices to plot (None = plot all)
        :param title: Plot title
        :param save_path: Optional save path
        :return: Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        if selected_indices is None:
            # Plot all spectra with transparent fill
            for i in range(len(spectra)):
                ax.plot(wavelengths, spectra[i], alpha=0.3, linewidth=1)

            # Highlight mean
            mean_spectrum = np.mean(spectra, axis=0)
            ax.plot(wavelengths, mean_spectrum, 'k-', linewidth=3, label='Mean', zorder=10)

            # Show ±1 std band
            std_spectrum = np.std(spectra, axis=0)
            ax.fill_between(wavelengths, mean_spectrum - std_spectrum, mean_spectrum + std_spectrum, color='gray', alpha=0.3, label='±1 std')
        else:
            # Plot selected spectra
            for idx in selected_indices:
                label = f"Position ({positions[idx, 0]:.1f}, {positions[idx, 1]:.1f})"
                ax.plot(wavelengths, spectra[idx], linewidth=2, label=label)

        ax.set_xlabel('Wavelength (nm)', fontweight='bold')
        ax.set_ylabel('Transmission', fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(wavelengths[0], wavelengths[-1])
        ax.set_ylim(0, 1.1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Spectral profile saved to {save_path}")

        return fig

    def plot_deltaE_distribution(self, deltaE_values: np.ndarray, spec_limits: Optional[float] = None, title: str = 'Delta E Distribution', save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot a histogram of deltaE values

        :param deltaE_values: Array of deltaE values
        :param spec_limit: Dictionary with 'acceptable' and 'marginal' limits
        :param title: Plot title
        :param save_path: Optional save path
        :return: Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        n, bins, patches = ax1.hist(deltaE_values, bins=20, edgecolor='black', alpha=0.7, color='skyblue')

        # Colour bars by spec
        if spec_limits:
            for i, patch in enumerate(patches):
                bin_centre = (bins[i] + bins[i+1]) / 2
                if bin_centre < spec_limits.get('acceptable', 3.0):
                    patch.set_facecolor('green')
                    patch.set_alpha(0.7)
                elif bin_centre < spec_limits.get('marginal', 5.0):
                    patch.set_facecolor('yellow')
                    patch.set_alpha(0.7)
                else:
                    patch.set_facecolor('red')
                    patch.set_alpha(0.7)

            # Add limit lines
            ax1.axvline(spec_limits.get('acceptable', 3.0), color='green', linestyle='--', linewidth=2, label='Acceptable')
            ax1.axvline(spec_limits.get('marginal', 5.0), color='red', linestyle='--', linewidth=2, label='Marginal')

        ax1.set_xlabel('Delta E*ab', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.set_title('DeltaE', fontweight='bold', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot
        bp = ax2.boxplot(deltaE_values, vert=False, patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor('skyblue')
        bp['boxes'][0].set_alpha(0.7)

        # Add spec lines
        if spec_limits:
            ax2.axvline(spec_limits.get('acceptable', 3.0), color='green', linestyle='--', linewidth=2, label='Acceptable')
            ax2.axvline(spec_limits.get('marginal', 5.0), color='red', linestyle='--', linewidth=2, label='Marginal')

        # Add stats text
        stats_text = f"Mean: {np.mean(deltaE_values):.2f}\n"
        stats_text += f"Median: {np.median(deltaE_values):.2f}\n"
        stats_text += f"Std: {np.std(deltaE_values):.2f}\n"
        stats_text += f"Min: {np.min(deltaE_values):.2f}\n"
        stats_text += f"Max: {np.max(deltaE_values):.2f}\n"

        ax2.text(1.3, np.max(deltaE_values), stats_text, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), verticalalignment='top')

        ax2.set_ylabel('Delta E*ab', fontweight='bold')
        ax2.set_title('Box Plot', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xticklabels([''])

        plt.suptitle(title, fontweight='bold', fontsize=14)
        plt.tight_layout()


        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distribution plot saved to {save_path}")

        return fig

    def create_qc_report_figure(self, analysis_results: Dict, qc_results: Dict, grid_data: Dict, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a QC report figure
        :param analysis_results: Dictionary from calculate_uniformity_metric(
        :param qc_results: Dictionary from check_specifications()
        :param grid_data: Dictionary from load_grid_measurements()
        :param save_path: Optional save path
        :return: Figure object
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # DeltaE heatmap (top left, large)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        positions = grid_data['positions']
        deltaE = analysis_results['deltaE_values']

        scatter = ax1.scatter(positions[:, 0], positions[:, 1], c=deltaE, cmap='RdYlGn_r', s=200, edgecolors='black', linewidths=2)
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Delta E*ab', rotation=270, labelpad=20)

        ax1.set_xlabel('X Position (mm)', fontweight='bold')
        ax1.set_ylabel('Y Position (mm)', fontweight='bold')
        ax1.set_title('Spatial DeltaE Distribution', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')

        # 2. Chromaticity (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        xy_values = grid_data['xy_values']
        ax2.scatter(xy_values[:, 0], xy_values[:, 1], s=50, edgecolors='black', alpha=0.6)

        mean_xy = np.mean(xy_values, axis=0)
        ax2.plot(mean_xy[0], mean_xy[1], 'r*', markersize=15)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('Chromaticity', fontweight='bold', fontsize=10)
        ax2.grid(True, alpha=0.3)

        # 3. DeltaE histogram (middle right)
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.hist(deltaE, bins=15, edgecolor='black', alpha=0.7)
        ax3.axvline(analysis_results['deltaE_mean'], color='red', linestyle='--', label='Mean')
        ax3.set_xlabel('Delta E*ab')
        ax3.set_ylabel('Count')
        ax3.set_title('DeltaE Distribution', fontweight='bold', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8)

        # 4. Spectra (bottom, span 2 columns)
        ax4 = fig.add_subplot(gs[2, 0:2])
        wavelengths = grid_data['wavelengths']
        spectra = grid_data['spectra']

        for i in range(len(spectra)):
            ax4.plot(wavelengths, spectra[i], alpha=0.2, linewidth=0.5)

        mean_spectrum = np.mean(spectra, axis=0)
        ax4.plot(wavelengths, mean_spectrum, 'k-', linewidth=2, label='Mean')
        ax4.set_xlabel('Wavelength (nm)', fontweight='bold')
        ax4.set_ylabel('Transmission')
        ax4.set_title('Spectral Distribution', fontweight='bold', fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        # 5. QC summary (bottom right)
        ax5 = fig.add_subplot(gs[2, 2])
        ax5.axis('off')

        # Create summary text
        summary_text = "QC SUMMARY\n" + "="*30 + "\n\n"

        if qc_results['pass']:
            summary_text += "Result: PASS\n\n"
            text_color = 'green'
        else:
            summary_text += "Result: FAIL\n\n"
            text_color = 'red'

        summary_text += f"Mean Delta E: {analysis_results['deltaE_mean']:.2f}\n"
        summary_text += f"Max Delta E: {analysis_results['deltaE_max']:.2f}\n\n"
        summary_text += f"Std Delta E: {analysis_results['deltaE_std']:.2f}\n\n"

        summary_text += f"Measurements: {len(positions)}\n"
        summary_text += f"Wavelengths: {wavelengths[0]:.0f}-{wavelengths[-1]:.0f} nm\n\n"

        if qc_results['failures']:
            summary_text += "Failures:\n"
            for failure in qc_results['failures'][:3]:  # show first 3
                summary_text += f"- {failure[:40]}...\n"

        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), family='monospace')

        # Main title
        timestamp = qc_results['metadata']['timestamp']
        fig.suptitle(f"QC Report - {timestamp}", fontsize=16, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"QC report figure saved to {save_path}")

        return fig

    def _draw_cie_horseshoe(self, ax):
        """Draw CIE 1931 chromaticity diagram horseshoe"""
        # Wavelength locus (380-700 nm)
        wl_locus = np.array([
            [0.1741, 0.0050], [0.1740, 0.0050], [0.1738, 0.0049],
            [0.1736, 0.0049], [0.1733, 0.0048], [0.1730, 0.0048],
            [0.1726, 0.0048], [0.1721, 0.0048], [0.1714, 0.0051],
            [0.1703, 0.0058], [0.1689, 0.0069], [0.1669, 0.0086],
            [0.1644, 0.0109], [0.1611, 0.0138], [0.1566, 0.0177],
            [0.1510, 0.0227], [0.1440, 0.0297], [0.1355, 0.0399],
            [0.1241, 0.0578], [0.1096, 0.0868], [0.0913, 0.1327],
            [0.0687, 0.2007], [0.0454, 0.2950], [0.0235, 0.4127],
            [0.0082, 0.5384], [0.0039, 0.6548], [0.0139, 0.7502],
            [0.0389, 0.8120], [0.0743, 0.8338], [0.1142, 0.8262],
            [0.1547, 0.8059], [0.1929, 0.7816], [0.2296, 0.7543],
            [0.2658, 0.7243], [0.3016, 0.6923], [0.3373, 0.6589],
            [0.3731, 0.6245], [0.4087, 0.5896], [0.4441, 0.5547],
            [0.4788, 0.5202], [0.5125, 0.4866], [0.5448, 0.4544],
            [0.5752, 0.4242], [0.6029, 0.3965], [0.6270, 0.3725],
            [0.6482, 0.3514], [0.6658, 0.3340], [0.6801, 0.3197],
            [0.6915, 0.3083], [0.7006, 0.2993], [0.7079, 0.2920],
            [0.7140, 0.2859], [0.7190, 0.2809], [0.7230, 0.2770],
            [0.7260, 0.2740], [0.7283, 0.2717], [0.7300, 0.2700]
        ])

        # Close the horseshoe with a purple line
        purple_line = np.array([[wl_locus[-1, 0], wl_locus[-1, 1]], [wl_locus[0, 0], wl_locus[0, 1]]])

        ax.plot(wl_locus[:, 0], wl_locus[:, 1], 'k-', linewidth=2)
        ax.plot(purple_line[:, 0], purple_line[:, 1], 'k--', linewidth=2)

        # Fill with colour (optional - simplified)
        ax.fill(wl_locus[:, 0], wl_locus[:, 1], color='lightgray', alpha=0.2)

    def _get_parameter_label(self, parameter: str) -> str:
        """Get label for parameter"""
        labels = {
            'deltaE': 'Delta E*ab',
            'L': 'L* (Lightness)',
            'a': 'a* (red-green)',
            'b': 'b* (yellow-blue)',
            'xy_deviation': 'Chromaticity Deviation'
        }
        return labels.get(parameter, parameter)

# Test code
if __name__ == "__main__":
    print("=== Visualizer Test===")

    # Create data for testing
    np.random.seed(42)
    n_points = 25

    # Grid positions
    x = np.linspace(0,10, 5)
    y = np.linspace(0,10, 5)
    X, Y = np.meshgrid(x, y)
    positions = np.column_stack([X.ravel(), Y.ravel()])

    # Synthetic deltaE values (higher at the edges)
    centre = np.array([5, 5])
    distances = np.linalg.norm(positions - centre, axis=1)
    deltaE_values = 1.0 + 2.0 * (distances / np.max(distances)) + np.random.normal(0, 0.3, n_points)
    deltaE_values = np.clip(deltaE_values, 0, 5)

    # Create interpolated data
    Xi, Yi = np.meshgrid(np.linspace(0, 10, 50), np.linspace(0, 10, 50))
    from scipy.interpolate import griddata
    Zi = griddata(positions, deltaE_values, (Xi, Yi), method='cubic')

    interpolated_data = {
        'X': Xi,
        'Y': Yi,
        'Z': Zi,
        'parameter': 'deltaE',
        'original_positions': positions,
        'original_values': deltaE_values
    }

    # Initialize visualizer
    viz = UniformityVisualizer()

    print("1. Creating uniformity heatmap...")
    fig1 = viz.plot_uniformity_heatmap(interpolated_data, spec_limit=3.0, save_path='test_heatmap.png')

    print("\n2. Creating deltaE distribution plot...")
    spec_limits = {'acceptable': 3.0, 'marginal': 5.0}
    fig2 = viz.plot_deltaE_distribution(deltaE_values, spec_limits=spec_limits, save_path='test_distribution.png')

    print("\n3. Creating chromaticity diagram...")
    # Synthetic xy values (greenish region)
    xy_values = np.random.normal([0.3, 0.6], [0.02, 0.03], (n_points, 2))
    fig3 = viz.plot_chromaticity_diagram(xy_values, positions=positions, save_path='test_chromaticity.png')

    plt.show()

    print("\n=== Test Complete ===")
    print("Generated test images:")
    print("- test_heatmap.png")
    print("- test_distribution.png")
    print("- test_chromaticity.png")