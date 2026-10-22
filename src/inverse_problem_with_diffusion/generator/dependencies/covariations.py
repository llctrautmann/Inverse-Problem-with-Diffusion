# Covariation Helper Functions
import numpy as np

class CovariationPatterns:
    """Collection of covariation functions for ellipse generation and intensity patterns"""
    
    # ============= LEVEL 1: LINEAR =============
    
    @staticmethod
    def linear_area_aspect(Af, k=2.0, Ar_min=1.5):
        """Larger ellipses are more elongated"""
        return Ar_min + k * Af
    
    @staticmethod
    def linear_position_rotation(cx, W, theta_min=0, theta_max=np.pi, rng=None, jitter=0.3):
        """Rotation increases with x-position, with optional seeded jitter"""
        theta = theta_min + (theta_max - theta_min) * (cx / W)
        if rng is not None:
            theta += rng.uniform(-jitter, jitter)
        return theta
    
    @staticmethod
    def linear_position_intensity(cx, W, I_min=40, I_max=255, **kwargs):
        """Intensity increases with x-position"""
        return I_min + (I_max - I_min) * (cx / W)
    
    # ============= LEVEL 2: NONLINEAR SINGLE-VAR =============
    
    @staticmethod
    def radial_area_decay(cx, cy, W, H, Af_max=0.15, k=1.0):
        """Ellipses get smaller away from center"""
        r_norm = np.sqrt(((cx - W/2) / (W/2))**2 + ((cy - H/2) / (H/2))**2)
        return Af_max * np.exp(-k * r_norm**2)
    
    @staticmethod
    def sinusoidal_aspect(cx, W, Ar_mean=2.0, amplitude=0.8, n_cycles=3, phase=0):
        """Aspect ratio oscillates with position"""
        return Ar_mean + amplitude * np.sin(2 * np.pi * n_cycles * cx / W + phase)
    
    @staticmethod
    def radial_rotation_tangent(cx, cy, W, H):
        """Ellipses aligned tangent to radial direction"""
        return np.arctan2(cy - H/2, cx - W/2) + np.pi/2
    
    @staticmethod
    def radial_rotation_radial(cx, cy, W, H):
        """Ellipses aligned with radial direction (sunburst)"""
        return np.arctan2(cy - H/2, cx - W/2)
    
    @staticmethod
    def quadratic_area_aspect(Af, a=10, b=-3, c=1.5):
        """Parabolic relationship between area and aspect"""
        return a * Af**2 + b * Af + c
    
    # ============= LEVEL 3: MULTI-VARIABLE =============
    
    @staticmethod
    def position_modulated_area_aspect(Af, cy, H, k=0.5, c=1.0):
        """Area-aspect relationship varies by y-position"""
        return (1 + k * cy / H) * Af + c
    
    @staticmethod
    def geometric_mean_intensity(Af, Ar, theta, k=500, **kwargs):
        """Intensity from multiple geometric properties"""
        return k * np.sqrt(Af * Ar) * (1 + np.cos(theta)) / 2
    
    @staticmethod
    def distance_modulated_rotation(cx, cy, W, H, k=2.0):
        """Rotation with spiral-like modulation"""
        r_norm = np.sqrt(((cx - W/2) / (W/2))**2 + ((cy - H/2) / (H/2))**2)
        base_angle = np.arctan2(cy - H/2, cx - W/2)
        return base_angle + k * r_norm
    
    # ============= LEVEL 4: CONSTRAINED CURVES =============
    

    @staticmethod
    def sinusoidal_constraint(Af, Ar_mean=2.0, amplitude=1.0, k=10, noise_std=0.1, rng=None):
        """Ar constrained to curve in (Af, Ar) space"""
        base = Ar_mean + amplitude * np.sin(k * Af)
        if rng is not None:
            base += rng.normal(0, noise_std)
        return base
    
    # ============= LEVEL 5: REGION-DEPENDENT =============
    

    @staticmethod
    def bifurcating_aspect(Af, cx, W, k1=2.0, k2=-1.0):
        """Different area-aspect relationships in different regions"""
        if cx < W/2:
            return 1.5 + k1 * Af  # left side: positive correlation
        else:
            return 3.0 + k2 * Af  # right side: negative correlation

    # ============= INTENSITY PATTERNS =============

    @staticmethod
    def eccentricity_intensity(a, b, I_min=50, I_max=255, **kwargs):
        """Intensity increases with ellipse eccentricity (how elongated)"""
        if a == 0:
            return I_min
        ecc = np.sqrt(1 - (b / a)**2) if a >= b else np.sqrt(1 - (a / b)**2)
        return I_min + (I_max - I_min) * ecc

    @staticmethod
    def diagonal_distance_intensity(cx, cy, W, H, I_min=80, I_max=200, **kwargs):
        """Intensity based on distance from main diagonal (x-y relationship)"""
        dist = abs(cx - (W - cy)) / np.sqrt(2)
        max_dist = np.sqrt(W**2 + H**2) / np.sqrt(2)
        normalized = dist / max_dist
        return I_min + (I_max - I_min) * normalized

    @staticmethod
    def orientation_alignment_intensity(cx, cy, theta, W, H, I_max=255, **kwargs):
        """Intensity based on how well ellipse orientation matches position angle"""
        pos_angle = np.arctan2(cy - H/2, cx - W/2)
        angle_diff = abs(theta - pos_angle)
        # Normalize to 0-1 (180 degree = 1.0)
        normalized = angle_diff / np.pi
        return I_max * (1 - normalized)

    @staticmethod
    def circularity_intensity(a, b, I_min=100, I_max=180, **kwargs):
        """Intensity increases with circularity (compactness measure)"""
        if a == 0 or b == 0:
            return I_min
        area = np.pi * a * b
        perimeter = 4 * np.pi * np.sqrt((a**2 + b**2) / 2)
        circularity = 4 * np.pi * area / (perimeter**2 + 1e-6)
        return I_min + (I_max - I_min) * circularity

    @staticmethod
    def quadrant_phase_intensity(cx, cy, W, H, phase=0, I_base=128, amplitude=80, **kwargs):
        """Intensity based on quadrant with phase shifts (checkerboard-like)"""
        quad_x = int(cx > W/2)
        quad_y = int(cy > H/2)
        quad = quad_x * 2 + quad_y
        phase_factor = np.sin(phase + quad * np.pi / 2)
        return I_base + amplitude * phase_factor

    @staticmethod
    def radial_energy_intensity(cx, cy, a, b, theta, W, H, **kwargs):
        """Intensity based on 'rotational energy' concept (moment of inertia)"""
        center_x, center_y = W/2, H/2
        r = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
        I_rot = r**2 * (a**2 + b**2) / 4
        max_val = (np.sqrt(W**2 + H**2)/2)**2 * (max(a, b)**2)
        intensity = 255 * (I_rot / (max_val + 1e-6))
        return np.clip(intensity, 50, 255)

    @staticmethod
    def aspect_oscillation_intensity(Ar, I_mean=140, amplitude=60, n_cycles=3, **kwargs):
        """Intensity oscillates with aspect ratio (waves in Ar space)"""
        return I_mean + amplitude * np.sin(n_cycles * Ar)

    @staticmethod
    def perimeter_density_intensity(a, b, cx, cy, W, H, I_min=60, I_max=220, **kwargs):
        """Intensity based on perimeter-to-position ratio (shape efficiency)"""
        perimeter = 4 * np.pi * np.sqrt((a**2 + b**2) / 2)
        center_dist = np.sqrt((cx - W/2)**2 + (cy - H/2)**2) + 1
        density = perimeter / center_dist
        normalized = np.clip(density / 50, 0, 1)
        return I_min + (I_max - I_min) * normalized

    @staticmethod
    def angle_resonance_intensity(theta, a, b, resonance_freq=2.0, I_base=130, amplitude=60, **kwargs):
        """Intensity based on 'resonance' between orientation and shape"""
        shape_param = a / (a + b + 1e-6)
        resonance = np.sin(resonance_freq * theta) * shape_param
        return I_base + amplitude * resonance

    @staticmethod
    def fibonacci_spiral_intensity(cx, cy, W, H, n_spirals=3, **kwargs):
        """Intensity along Fibonacci spiral pattern"""
        x = cx - W/2
        y = cy - H/2
        r = np.sqrt(x**2 + y**2)
        angle = np.arctan2(y, x)
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        spiral_phase = n_spirals * angle / (2 * np.pi)
        intensity = 128 + 80 * np.sin(spiral_phase)
        return np.clip(intensity, 50, 210)

