import numpy as np
from itertools import combinations


# Design contrast threshold (CIEDE2000)
# Based on color science standards:
# - < 1: imperceptible
# - 1-2: barely perceptible
# - 2-10: perceptible
# - 10-50: clearly different
# For good design contrast, recommend 25+ (clearly distinguishable)
MIN_CONTRAST_THRESHOLD = 40.0

# RGB difference threshold for "rainbow colors" perception
# Euclidean distance in RGB space (0-255 scale)
# - < 30: very similar, hard to distinguish
# - 30-50: similar, noticeable difference
# - 50-100: clearly different colors
# - 100+: very different, distinct colors (rainbow-like)
# For distinct rainbow colors perception, recommend 75+ (human eye perception)
MIN_RGB_RAINBOW_THRESHOLD = 120.0


def ciede2000(lab1, lab2):
    """
    Calculate the CIEDE2000 color difference between two LAB colors.
    
    Parameters:
    -----------
    lab1 : array-like
        LAB color 1 as [L*, a*, b*]
    lab2 : array-like
        LAB color 2 as [L*, a*, b*]
    
    Returns:
    --------
    float
        CIEDE2000 color difference
    """
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)
    
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    
    # Step 1: Calculate C* (chroma)
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    Cbar = (C1 + C2) / 2
    
    # Step 2: Calculate G and apply to a*
    G = 0.5 * (1 - np.sqrt(Cbar**7 / (Cbar**7 + 25**7)))
    a1_prime = (1 + G) * a1
    a2_prime = (1 + G) * a2
    
    # Step 3: Recalculate C* with modified a*
    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)
    
    # Step 4: Calculate h (hue angle)
    h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360
    h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360
    
    # Step 5: Calculate differences
    dL_prime = L2 - L1
    dC_prime = C2_prime - C1_prime
    
    # Hue difference
    dh_prime_abs = np.abs(h1_prime - h2_prime)
    if dh_prime_abs <= 180:
        dH_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(dh_prime_abs / 2))
    else:
        dH_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians((360 - dh_prime_abs) / 2))
    
    # Step 6: Calculate CIEDE2000
    Lbar_prime = (L1 + L2) / 2
    Cbar_prime = (C1_prime + C2_prime) / 2
    
    # Mean hue angle
    if dh_prime_abs <= 180:
        hbar_prime = (h1_prime + h2_prime) / 2
    else:
        if (h1_prime + h2_prime) < 360:
            hbar_prime = (h1_prime + h2_prime + 360) / 2
        else:
            hbar_prime = (h1_prime + h2_prime - 360) / 2
    
    # Weighting factors
    T = (1 - 0.17 * np.cos(np.radians(hbar_prime - 30)) +
         0.24 * np.cos(np.radians(2 * hbar_prime)) +
         0.32 * np.cos(np.radians(3 * hbar_prime + 6)) -
         0.20 * np.cos(np.radians(4 * hbar_prime - 63)))
    
    dtheta = 30 * np.exp(-((hbar_prime - 275) / 25)**2)
    Rc = 2 * np.sqrt(Cbar_prime**7 / (Cbar_prime**7 + 25**7))
    
    # Lightness, chroma, and hue weighting
    Sl = 1 + (0.015 * (Lbar_prime - 50)**2) / np.sqrt(20 + (Lbar_prime - 50)**2)
    Sc = 1 + 0.045 * Cbar_prime
    Sh = 1 + 0.015 * Cbar_prime * T
    
    # Final CIEDE2000 calculation
    Delta_E = np.sqrt(
        (dL_prime / (Sl))**2 +
        (dC_prime / (Sc))**2 +
        (dH_prime / (Sh))**2 +
        Rc * (dC_prime / (Sc)) * (dH_prime / (Sh))
    )
    
    return Delta_E


def rgb_distance(rgb1, rgb2):
    """
    Calculate the Euclidean distance between two RGB colors.
    
    Parameters:
    -----------
    rgb1 : array-like
        RGB color 1 as [R, G, B] (0-255 scale)
    rgb2 : array-like
        RGB color 2 as [R, G, B] (0-255 scale)
    
    Returns:
    --------
    float
        Euclidean RGB distance (0-255 scale)
    """
    rgb1 = np.asarray(rgb1, dtype=np.float64)
    rgb2 = np.asarray(rgb2, dtype=np.float64)
    
    distance = np.sqrt(np.sum((rgb1 - rgb2) ** 2))
    return distance


def calculate_palette_metrics(palette_lab, palette_rgb=None):
    """
    Calculate CIEDE2000 distances and RGB differences between all pairs of colors in a palette.
    
    Parameters:
    -----------
    palette_lab : list of array-like
        List of LAB colors, each as [L*, a*, b*]
    palette_rgb : list of array-like, optional
        List of RGB colors, each as [R, G, B] (0-255 scale)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'distances': 2D array of pairwise CIEDE2000 distances
        - 'min_distance': minimum CIEDE2000 distance
        - 'max_distance': maximum CIEDE2000 distance
        - 'mean_distance': mean CIEDE2000 distance
        - 'rgb_distances': 2D array of pairwise RGB distances (if palette_rgb provided)
        - 'min_rgb_distance': minimum RGB distance
        - 'max_rgb_distance': maximum RGB distance
        - 'mean_rgb_distance': mean RGB distance
        - 'pairs': list of (color_idx1, color_idx2, ciede2000_distance) tuples
        - 'rgb_pairs': list of (color_idx1, color_idx2, rgb_distance) tuples (if palette_rgb provided)
    """
    n_colors = len(palette_lab)
    distances_matrix = np.zeros((n_colors, n_colors))
    pairs = []
    
    # Calculate CIEDE2000 distances
    for i in range(n_colors):
        for j in range(i + 1, n_colors):
            distance = ciede2000(palette_lab[i], palette_lab[j])
            distances_matrix[i, j] = distance
            distances_matrix[j, i] = distance
            pairs.append((i, j, distance))
    
    metrics = {}
    
    if pairs:
        distances_flat = [d for _, _, d in pairs]
        metrics['distances'] = distances_matrix
        metrics['min_distance'] = min(distances_flat)
        metrics['max_distance'] = max(distances_flat)
        metrics['mean_distance'] = np.mean(distances_flat)
        metrics['pairs'] = pairs
    else:
        metrics['distances'] = distances_matrix
        metrics['min_distance'] = None
        metrics['max_distance'] = None
        metrics['mean_distance'] = None
        metrics['pairs'] = []
    
    # Calculate RGB distances if RGB palette is provided
    if palette_rgb is not None:
        rgb_distances_matrix = np.zeros((n_colors, n_colors))
        rgb_pairs = []
        
        for i in range(n_colors):
            for j in range(i + 1, n_colors):
                rgb_dist = rgb_distance(palette_rgb[i], palette_rgb[j])
                rgb_distances_matrix[i, j] = rgb_dist
                rgb_distances_matrix[j, i] = rgb_dist
                rgb_pairs.append((i, j, rgb_dist))
        
        if rgb_pairs:
            rgb_distances_flat = [d for _, _, d in rgb_pairs]
            metrics['rgb_distances'] = rgb_distances_matrix
            metrics['min_rgb_distance'] = min(rgb_distances_flat)
            metrics['max_rgb_distance'] = max(rgb_distances_flat)
            metrics['mean_rgb_distance'] = np.mean(rgb_distances_flat)
            metrics['rgb_pairs'] = rgb_pairs
        else:
            metrics['rgb_distances'] = rgb_distances_matrix
            metrics['min_rgb_distance'] = None
            metrics['max_rgb_distance'] = None
            metrics['mean_rgb_distance'] = None
            metrics['rgb_pairs'] = []
    
    return metrics


def print_palette_metrics(palette_lab, palette_hex=None, palette_rgb=None):
    """
    Print formatted metrics for a palette with CIEDE2000 and RGB contrast analysis.
    
    Parameters:
    -----------
    palette_lab : list of array-like
        List of LAB colors
    palette_hex : list, optional
        List of HEX color strings for display
    palette_rgb : list, optional
        List of RGB tuples for display
    """
    metrics = calculate_palette_metrics(palette_lab, palette_rgb=palette_rgb)
    
    print("PALETTE METRICS (CIEDE2000 & RGB Color Differences)")
    print(" "*70)
    
    # Print pairwise CIEDE2000 distances
    print("\nCIEDE2000 Distances:")
    print(" "*70)
    
    for idx1, idx2, distance in metrics['pairs']:
        color1_str = f"Color {idx1}"
        color2_str = f"Color {idx2}"
        
        if palette_hex:
            color1_str += f" ({palette_hex[idx1]})"
            color2_str += f" ({palette_hex[idx2]})"
        
        print(f"  {color1_str} <-> {color2_str}: {distance:.2f}")
    
    # Print pairwise RGB distances if available
    if palette_rgb is not None and 'rgb_pairs' in metrics:
        print("\nRGB Distances:")
        print(" "*70)
        
        # Calculate standard deviation of RGB distances
        rgb_std_deviation = None
        if metrics['rgb_pairs']:
            rgb_distances_list = [d for _, _, d in metrics['rgb_pairs']]
            rgb_mean_distance = metrics['mean_rgb_distance']
            rgb_std_deviation = np.std(rgb_distances_list)

            print(f"Mean Distance: {rgb_mean_distance:.2f}")
            print(f"Std Deviation: {rgb_std_deviation:.2f}")
            print(" "*70)
        
        for idx1, idx2, rgb_dist in metrics['rgb_pairs']:
            color1_str = f"Color {idx1}"
            color2_str = f"Color {idx2}"
            
            if palette_hex:
                color1_str += f" ({palette_hex[idx1]})"
                color2_str += f" ({palette_hex[idx2]})"
            
            # Determine if distance is above or below standard deviation
            if rgb_std_deviation is not None:
                if rgb_dist >= rgb_std_deviation:
                    std_status = "↑"  # Above standard
                    diff_from_std = rgb_dist - rgb_std_deviation
                else:
                    std_status = "↓"  # Below standard
                    diff_from_std = rgb_std_deviation - rgb_dist
            else:
                std_status = "="
                diff_from_std = 0
            
            # Determine if contrast is enough for rainbow colors
            if rgb_dist >= MIN_RGB_RAINBOW_THRESHOLD:
                rainbow_status = "✓"
            else:
                rainbow_status = "✗"
            
            print(f"  {std_status} {rainbow_status} {color1_str} <-> {color2_str}: {rgb_dist:.2f} (diff from std: {diff_from_std:.2f})")
    
    # Print summary statistics for CIEDE2000
    print("\n" + " "*70)
    print("Summary Statistics (CIEDE2000):")
    print(f"  Minimum distance: {metrics['min_distance']:.2f}")
    print(f"  Maximum distance: {metrics['max_distance']:.2f}")
    print(f"  Mean distance:    {metrics['mean_distance']:.2f}")
    
    # Print summary statistics for RGB if available
    if palette_rgb is not None and 'mean_rgb_distance' in metrics:
        print("\nSummary Statistics (RGB):")
        print(f"  Minimum distance: {metrics['min_rgb_distance']:.2f}")
        print(f"  Maximum distance: {metrics['max_rgb_distance']:.2f}")
        print(f"  Mean distance:    {metrics['mean_rgb_distance']:.2f}")
        
        # Calculate and display standard deviation
        if metrics['rgb_pairs']:
            rgb_distances_list = [d for _, _, d in metrics['rgb_pairs']]
            rgb_std_deviation = np.std(rgb_distances_list)
            print(f"  Std Deviation:    {rgb_std_deviation:.2f}")
    
    # Print CIEDE2000 contrast analysis
    print("\n" + " "*70)
    print(f"CIEDE2000 Contrast Analysis (Threshold: {MIN_CONTRAST_THRESHOLD:.2f}):")
    mean_distance = metrics['mean_distance']
    
    if mean_distance >= MIN_CONTRAST_THRESHOLD:
        status = "GOOD"
        difference = mean_distance - MIN_CONTRAST_THRESHOLD
        print(f"  Status: {status} - Average contrast is HIGHER than threshold")
        print(f"  Difference: +{difference:.2f} (Palette has good color separation)")
    else:
        status = "POOR"
        difference = MIN_CONTRAST_THRESHOLD - mean_distance
        print(f"  Status: {status} - Average contrast is LOWER than threshold")
        print(f"  Difference: -{difference:.2f} (Colors may appear too similar)")
    
    # Print RGB rainbow color analysis if available
    if palette_rgb is not None and 'mean_rgb_distance' in metrics:
        print("\n" + " "*70)
        print(f"RGB Rainbow Color Analysis (Threshold: {MIN_RGB_RAINBOW_THRESHOLD:.2f}):")
        mean_rgb_distance = metrics['mean_rgb_distance']
        
        # Calculate standard deviation for comparison
        if metrics['rgb_pairs']:
            rgb_distances_list = [d for _, _, d in metrics['rgb_pairs']]
            rgb_std_deviation = np.std(rgb_distances_list)
            
            # Count pairs above and below standard deviation
            above_std = sum(1 for _, _, d in metrics['rgb_pairs'] if d >= rgb_std_deviation)
            below_std = sum(1 for _, _, d in metrics['rgb_pairs'] if d < rgb_std_deviation)
            total_pairs = len(metrics['rgb_pairs'])
            
            print(f"  Pairs above standard deviation: {above_std}/{total_pairs}")
            print(f"  Pairs below standard deviation: {below_std}/{total_pairs}")
        
        if mean_rgb_distance >= MIN_RGB_RAINBOW_THRESHOLD:
            status = "EXCELLENT"
            difference = mean_rgb_distance - MIN_RGB_RAINBOW_THRESHOLD
            print(f"  Status: {status} - Colors appear as distinct RAINBOW colors")
            print(f"  Difference: +{difference:.2f} (Perfect for diverse color palettes)")
        elif mean_rgb_distance >= 50:
            status = "GOOD"
            difference = mean_rgb_distance - 50
            print(f"  Status: {status} - Colors are clearly different")
            print(f"  Difference: +{difference:.2f} from 'clearly different' threshold")
        else:
            status = "POOR"
            difference = 50 - mean_rgb_distance
            print(f"  Status: {status} - Colors are too similar for rainbow perception")
            print(f"  Difference: -{difference:.2f} from 'clearly different' threshold")
    
    
    return metrics
