"""
color_converter.py
Spectral to CIE color space conversion for optical QC

Implements conversions from spectral data to XYZ, Lab and other color spaces
Based on the CIE 1931 color space and the CIE 1960 color space

This implementation builds on colorimetry methodology developed in:
"Colour Characterisation of a LCD and Mobile Display Using Polynomial and 
Masking Models" (Mohamed, Balogun, Das, 2017)

Author: Olufemi Balogun
Date: October 2025
"""

import numpy as np
from typing import Tuple, Optional, Dict
from scipy import interpolate


class ColorConverter:
    """
    Convert spectral data to CIE spaces and calculate color differences

    Supports:
    Spectral radiance/reflectance to XYZ tristimulus values
    XYZ to Lab (CIELAB)
    XYZ to xy chromaticity coordinates
    -ΔE color difference calculations (ΔE*ab, ΔE00)
    """

    def __init__(self):
        """Initialize with CIE standard observer and illuminants"""
        self.cmf_wavelengths = np.arange(380, 781, 1) # 1nm resolution
        self.cmf = self._load_cie_cmf() # Color matching functions
        self.illuminants = self._load_illuminants()


    def _load_cie_cmf(self) -> np.ndarray:
        """
        Load CIE 1931 2-degree standard observer color matching functions
        :return: Array of shape (401,3) for wavelengths at 1nm intervals
                Columns are x̄(λ), ȳ(λ), z̄(λ)
        """

        # official values of CIE 1931 2-degree standard observer ( 380-780nm, 5nm intervals)
        wavelengths_cie = np.array([
            380, 385, 390, 395, 400, 405, 410, 415, 420, 425, 430, 435, 440, 445, 450,
            455, 460, 465, 470, 475, 480, 485, 490, 495, 500, 505, 510, 515, 520, 525,
            530, 535, 540, 545, 550, 555, 560, 565, 570, 575, 580, 585, 590, 595, 600,
            605, 610, 615, 620, 625, 630, 635, 640, 645, 650, 655, 660, 665, 670, 675,
            680, 685, 690, 695, 700, 705, 710, 715, 720, 725, 730, 735, 740, 745, 750,
            755, 760, 765, 770, 775, 780
        ])

        # x̄(λ) - red sensitivity
        x_bar = np.array([
            0.0014, 0.0022, 0.0042, 0.0076, 0.0143, 0.0232, 0.0435, 0.0776, 0.1344, 0.2148,
            0.2839, 0.3285, 0.3483, 0.3481, 0.3362, 0.3187, 0.2908, 0.2511, 0.1954, 0.1421,
            0.0956, 0.0580, 0.0320, 0.0147, 0.0049, 0.0024, 0.0093, 0.0291, 0.0633, 0.1096,
            0.1655, 0.2257, 0.2904, 0.3597, 0.4334, 0.5121, 0.5945, 0.6784, 0.7621, 0.8425,
            0.9163, 0.9786, 1.0263, 1.0567, 1.0622, 1.0456, 1.0026, 0.9384, 0.8544, 0.7514,
            0.6424, 0.5419, 0.4479, 0.3608, 0.2835, 0.2187, 0.1649, 0.1212, 0.0874, 0.0636,
            0.0468, 0.0329, 0.0227, 0.0158, 0.0114, 0.0081, 0.0058, 0.0041, 0.0029, 0.0020,
            0.0014, 0.0010, 0.0007, 0.0005, 0.0003, 0.0002, 0.0002, 0.0001, 0.0001, 0.0001, 0.0000
        ])

        # ȳ(λ) - green sensitivity (also luminosity function)
        y_bar = np.array([
            0.0000, 0.0001, 0.0001, 0.0002, 0.0004, 0.0006, 0.0012, 0.0022, 0.0040, 0.0073,
            0.0116, 0.0168, 0.0230, 0.0298, 0.0380, 0.0480, 0.0600, 0.0739, 0.0910, 0.1126,
            0.1390, 0.1693, 0.2080, 0.2586, 0.3230, 0.4073, 0.5030, 0.6082, 0.7100, 0.7932,
            0.8620, 0.9149, 0.9540, 0.9803, 0.9950, 1.0000, 0.9950, 0.9786, 0.9520, 0.9154,
            0.8700, 0.8163, 0.7570, 0.6949, 0.6310, 0.5668, 0.5030, 0.4412, 0.3810, 0.3210,
            0.2650, 0.2170, 0.1750, 0.1382, 0.1070, 0.0816, 0.0610, 0.0446, 0.0320, 0.0232,
            0.0170, 0.0119, 0.0082, 0.0057, 0.0041, 0.0029, 0.0021, 0.0015, 0.0010, 0.0007,
            0.0005, 0.0004, 0.0002, 0.0002, 0.0001, 0.0001, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000
        ])

        # z̄(λ) - blue sensitivity
        z_bar = np.array([
            0.0065, 0.0105, 0.0201, 0.0362, 0.0679, 0.1102, 0.2074, 0.3713, 0.6456, 1.0391,
            1.3856, 1.6230, 1.7471, 1.7826, 1.7721, 1.7441, 1.6692, 1.5281, 1.2876, 1.0419,
            0.8130, 0.6162, 0.4652, 0.3533, 0.2720, 0.2123, 0.1582, 0.1117, 0.0782, 0.0573,
            0.0422, 0.0298, 0.0203, 0.0134, 0.0087, 0.0057, 0.0039, 0.0027, 0.0021, 0.0018,
            0.0017, 0.0014, 0.0011, 0.0010, 0.0008, 0.0006, 0.0003, 0.0002, 0.0002, 0.0001,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
        ])

        # Interpolate CMF to 1nm resolution
        wl_target = np.arange(380, 781, 1)

        x_interp = interpolate.interp1d(wavelengths_cie, x_bar, kind='cubic', fill_value=0, bounds_error=False)
        y_interp = interpolate.interp1d(wavelengths_cie, y_bar, kind='cubic', fill_value=0, bounds_error=False)
        z_interp = interpolate.interp1d(wavelengths_cie, z_bar, kind='cubic', fill_value=0, bounds_error=False)

        cmf = np.column_stack([x_interp(wl_target),
                               y_interp(wl_target),
                               z_interp(wl_target)
                               ])
        return cmf

    def _load_illuminants(self) -> Dict[str, np.ndarray]:
        """
        Load standard CIE illuminants
        :return: Dictionary of illuminants SPDs (380-780nm, 1nm resolution)
        """
        # D65 (daylight, 6500K) - most common
        # Simplified approximation - in production, use official values
        wl = self.cmf_wavelengths

        # D65 approximation using Planck's law and corrections
        T = 6500 # Kelvin
        c1 = 3.74183e-16 # 2πhc²
        c2 = 1.4388e-2 # hc/k

        # Planck's law
        D65 = c1 / ((wl * 1e-9)**5 * (np.exp(c2 / (wl * 1e-9 * T)) - 1))
        D65 = D65 / np.max(D65) # Normalize

        # A (tungsten filament) - 2856K
        T_a = 2856
        A = c1 / ((wl * 1e-9)**5 * (np.exp(c2 / (wl * 1e-9 * T_a)) - 1))
        A = A / np.max(A)

        return {
            'D65': D65,
            'A': A,
            'E': np.ones_like(wl) # Equal energy
        }

    def spectrum_to_xyz(self,wavelengths: np.ndarray, spectrum: np.ndarray, illuminant: Optional[str] = None, mode: str = 'radiance') -> np.ndarray:
        """
        Convert spectral data to XYZ tristimulus values
        :param spectrum: Spectral radiance or reflectance
        :param illuminant: 'D65', 'A', 'E' or None (for self-luminous)
        :param mode: 'radiance' (self-luminous) or 'reflectance' (reflecting surface)
        :return: XYZ tristimulus values as arrays
        """
        spectrum_interp = interpolate.interp1d(wavelengths, spectrum,
                                               kind='linear', fill_value=0,
                                               bounds_error=False
                                               )(self.cmf_wavelengths)

        if mode == 'reflectance' and illuminant is not None:
            # Multiply reflectance by illuminant
            illuminant_spd = self.illuminants[illuminant]
            spectrum_interp = spectrum_interp * illuminant_spd

        # Calculate tristimulus values using trapezoidal integration
        # XYZ = k * ∫ S(λ) * CMF(λ) dλ
        # k = 100 for reflectance, (normalized to Y=100 for perfect white

        if mode == 'radiance':
            k = 683 # Photometric constant
        else:
            k = 100 # Reflectance constant
            # Normalize by illuminant integral
            illum_spd = self.illuminants[illuminant]
            k = k / np.trapezoid(illum_spd * self.cmf, self.cmf_wavelengths)

        X = k * np.trapezoid(spectrum_interp * self.cmf[:,0], self.cmf_wavelengths)
        Y = k * np.trapezoid(spectrum_interp * self.cmf[:,1], self.cmf_wavelengths)
        Z = k * np.trapezoid(spectrum_interp * self.cmf[:,2], self.cmf_wavelengths)

        return np.array([X, Y, Z])

    def XYZ_to_Lab(self,XYZ: np.ndarray, white_point: Optional[np.ndarray] = None, illuminant: str = None) -> np.ndarray:
        """
        Convert XYZ tristimulus values to CIE Lab color space
        :param XYZ: XYZ tristimulus values as arrays
        :param white_point: Reference white point as array [Xn, Yn, Zn]. if None, uses illuminant
        :param illuminant: 'D65', 'A' for standard white point
        :return: Lab color space values as arrays
        """
        # Get white point if not provided
        if white_point is None:
            if illuminant == 'D65':
                white_point = np.array([95.047, 100.000, 108.883])
            elif illuminant == 'A':
                white_point = np.array([109.850, 100.000, 35.585])
            else:
                white_point = np.array([100.000, 100.000, 100.000])

        # Normalize by white point
        xyz = XYZ / white_point

        # Apply f(t) function
        # f(t) = t^(1/3) if t > (6/29)^3, otherwise f(t) = (1/3)*((29/6)^3)*t + 4/29

        epsilon = (6/29)**3
        kappa = 29/6

        def f(t):
            """Nonlinear transformation for Lab"""
            mask = t > epsilon
            result = np.zeros_like(t)
            result[mask] = np.power(t[mask], 1/3)
            result[~mask] = (kappa * t[~mask] + 16) / 116
            return result

        fx, fy, fz = f(xyz[0]), f(xyz[1]), f(xyz[2])

        # Calculate Lab
        L_star = 116 * fy - 16
        a_star = 500 * (fx - fy)
        b_star = 200 * (fy - fz)

        return np.array([L_star, a_star, b_star])

    def XYZ_to_xy(self, XYZ: np.ndarray) -> np.ndarray:
        """
        Convert XYZ to xy chromaticity coordinates
        :param XYZ: XYZ tristimulus values [X,Y,Z]
        :return: xy chromaticity coordinates [x,y]
        """
        sum_XYZ = np.sum(XYZ)

        if sum_XYZ == 0:
            return np.array([0.0,0.0])

        x = XYZ[0] / sum_XYZ
        y = XYZ[1] / sum_XYZ

        return np.array([x,y])

    def calculate_deltaE_ab(self, Lab1: np.ndarray, Lab2: np.ndarray) -> float:
        """
        Calculate CIE deltaE_ab color difference

        ΔE*ab = √[(ΔL*)² + (Δa*)² + (Δb*)²]
        :param Lab1: First Lab color [L*, a*, b*]
        :param Lab2: Second Lab color [L*, a*, b*]
        :return: Color difference ΔE*ab
        """
        dL = Lab1[0] - Lab2[0]
        da = Lab1[1] - Lab2[1]
        db = Lab1[2] - Lab2[2]

        deltaE = np.sqrt(dL**2 + da**2 + db**2)
        return deltaE

    def calculate_deltaE_00(self, Lab1: np.ndarray, Lab2: np.ndarray,
                            kL: float = 1.0, kC: float = 1.0, kH: float = 1.0) -> float:
        """
        Calculate CIE deltaE_00 color difference (CIEDE 2000)

        More perceptually uniform than deltaE_ab and accounts for
        - Lightness weighting
        - Chroma weighting
        - Hue weighting
        - Rotation term for blue region

        :param Lab1: First Lab color [L*, a*, b*]
        :param Lab2: Second Lab color [L*, a*, b*]
        :param kL: Lightness scale factor
        :param kC: Chroma scale factor
        :param kH: Hue scale factor
        :return: Color difference ΔE*00
        """
        L1, a1,b1 = Lab1
        L2, a2,b2 = Lab2

        # Calculate C and h
        C1 = np.sqrt(a1**2 + b1**2)
        C2 = np.sqrt(a2**2 + b2**2)
        C_bar = (C1 + C2) / 2

        # a' correction
        G = 0.5 * (1 - np.sqrt(C_bar**7 / (C_bar**7 + 25**7)))
        a1_prime = (1 + G) * a1
        a2_prime = (1 + G) * a2

        C1_prime = np.sqrt(a1_prime**2 + b1**2)
        C2_prime = np.sqrt(a2_prime**2 + b2**2)
        C_bar_prime = (C1_prime + C2_prime) / 2

        h1_prime = np.arctan2(b1,a1_prime) * 180 / np.pi
        if h1_prime < 0:
            h1_prime += 360
        h2_prime = np.arctan2(b2,a2_prime) * 180 / np.pi
        if h2_prime < 0:
            h2_prime += 360

        # Differences
        delta_LPrime = L2 - L1
        delta_CPrime = C2_prime - C1_prime

        # dH' calculation
        if C1_prime * C2_prime == 0:
            delta_Hprime = 0
        else:
            delta_Hprime = h2_prime - h1_prime
            if delta_Hprime > 180:
                delta_Hprime -= 360
            elif delta_Hprime < -180:
                delta_Hprime += 360

        delta_HPrime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(delta_Hprime * np.pi / 360)

        # Weighting functions
        L_bar_prime = (L1 + L2) / 2

        if C1_prime * C2_prime == 0:
            H_bar_prime = h1_prime + h2_prime
        else:
            H_bar_prime = (h1_prime + h2_prime) / 2
            if np.abs(h1_prime - h2_prime) > 180:
                if H_bar_prime < 180:
                    H_bar_prime += 180
                else:
                    H_bar_prime -= 180

        T = (1 - 0.17 * np.cos((H_bar_prime - 30) * np.pi / 180) +
             0.24 * np.cos(2 * H_bar_prime * np.pi / 180) -
             0.32 * np.cos((3 * H_bar_prime + 6) * np.pi / 180) -
             0.20 * np.cos((4 * H_bar_prime - 63) * np.pi / 180))

        SL = 1 + ((0.015 * (L_bar_prime - 50)**2) /
                  np.sqrt(20 + (L_bar_prime - 50)**2))
        SC = 1 + 0.045 * C_bar_prime
        SH = 1 + 0.015 * C_bar_prime * T

        # Rotation term
        dtheta = 30 * np.exp(-((H_bar_prime - 275) / 25)**2)
        RC = 2 * np.sqrt(C_bar_prime**7 / (C_bar_prime**7 + 25**7))
        RT = -RC * np.sin(2 * dtheta * np.pi / 180)

        # Final ΔE00
        delta_E00 = np.sqrt(
            (delta_LPrime / (kL * SL))**2 +
            (delta_CPrime / (kC * SC))**2 +
            (delta_HPrime / (kH * SH))**2 +
            RT * (delta_CPrime / (kC * SC)) * (delta_HPrime / (kH * SH))
        )

        return delta_E00

    def black_correction(self, spectrum: np.ndarray, dark_spectrum: np.ndarray) -> np.ndarray:
        """
        Apply black correction to spectral data

        Subtracts dark measurement and sets negative values to zero
        :param: Spectrum: Measured spectrum
        :param: Dark spectrum: Dark measurement (offset)
        :return: Corrected spectrum
        """
        corrected = spectrum - dark_spectrum
        corrected[corrected < 0] = 0
        return corrected

# Test code
if __name__ == "__main__":
    print("==Color converter test==\n")

    converter = ColorConverter()

    # Test 1: Convert a synthetic spectrum to XYZ
    print("1. Testing Spectrum to XYZ conversion:")
    wavelengths = np.linspace(400, 700, 100)
    # Simulate greenish spectrum ( peak at 550nm)
    spectrum = np.exp(-((wavelengths-550)**2) / (2 * 30**2))

    XYZ = converter.spectrum_to_xyz(wavelengths, spectrum, mode='radiance')
    print(f"    XYZ: {XYZ}")

    print (f"   (Should have high Y for green)")

    # Test 2: Convert XYZ to Lab
    print("\n2. Testing XYZ to Lab conversion:")
    Lab = converter.XYZ_to_Lab(XYZ)
    print(f"    Lab: {Lab}")
    print(f"    L*: {Lab[0]:.1f} (lightness)")
    print(f"    a*: {Lab[1]:.1f} (negative = green)")
    print(f"    b*: {Lab[2]:.1f} (positive = yellow)")

    # Test 3: Calculate deltaE_ab
    print("\n3. Testing deltaE_ab calculation:")
    Lab1 = np.array([50, 20, 30])
    Lab2 = np.array([55, 25, 35])

    deltaE_ab = converter.calculate_deltaE_ab(Lab1, Lab2)
    deltaE_00 = converter.calculate_deltaE_00(Lab1, Lab2)

    print(f"    deltaE_ab: {deltaE_ab:.2f}")
    print(f"    deltaE_00: {deltaE_00:.2f}")
    print(f"    (deltaE_00 is usually smaller, more perceptually uniform)")

    # Test 4: Chromaticity
    print("\n4. Testing chromaticity conversion:")
    xy = converter.XYZ_to_xy(XYZ)
    print(f"    xy: {xy}")
    print(f"    (Should be in green region of chromaticity diagram)")

    print("\n=== Test completed ===")