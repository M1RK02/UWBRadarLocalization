#!/usr/bin/env python3
"""
FoV and Range Bin Visualization Script for EEAI Multi-Person Localization.
Visualizes how the 6 UWB radar sensors cover the 4.8m x 7.2m room,
explaining how the 120 range bins map to physical space and how bin
truncation can be performed without losing any coverage.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Wedge

# Constants
ROOM_WIDTH = 4.8  # meters (x-axis)
ROOM_HEIGHT = 7.2  # meters (y-axis)
BIN_RESOLUTION = 0.15  # meters per bin
TOTAL_BINS = 120
MAX_RANGE = TOTAL_BINS * BIN_RESOLUTION  # 18.0 m
DIRECT_COUPLING_BINS = 5
DIRECT_COUPLING_RANGE = DIRECT_COUPLING_BINS * BIN_RESOLUTION  # 0.75 m

# Radar configurations
# Positions: (x, y, z) in meters
# Facing/Boresight angles in degrees:
#   Up = 90, Left = 180, Down = 270, Right = 0
RADARS = {
    "SR250_1": {"pos": (2.4, 0.0), "angle": 90, "color": "#1f77b4"},
    "SR250_2": {"pos": (4.8, 1.8), "angle": 180, "color": "#ff7f0e"},
    "SR250_3": {"pos": (4.8, 5.4), "angle": 180, "color": "#2ca02c"},
    "SR250_4": {"pos": (2.4, 7.2), "angle": 270, "color": "#d62728"},
    "SR250_5": {"pos": (0.0, 5.4), "angle": 0, "color": "#9467bd"},
    "SR250_6": {"pos": (0.0, 1.8), "angle": 0, "color": "#8c564b"},
}


def get_angle_diff(angle1, angle2):
    """Calculate the absolute difference between two angles in degrees."""
    diff = np.abs(angle1 - angle2) % 360
    return np.minimum(diff, 360 - diff)


def analyze_sensor_coverage(name, config):
    """
    Determine the maximum range and bin index that falls inside the room
    and within the sensor's 120-degree FoV.
    """
    xr, yr = config["pos"]
    boresight = config["angle"]

    # Grid search over the room area to find the furthest point inside the FoV
    xs = np.linspace(0, ROOM_WIDTH, 1000)
    ys = np.linspace(0, ROOM_HEIGHT, 1000)
    xv, yv = np.meshgrid(xs, ys)

    # Distances from sensor
    dx = xv - xr
    dy = yv - yr
    dists = np.sqrt(dx**2 + dy**2)

    # Angles from sensor (in degrees, -180 to 180)
    angles = np.degrees(np.arctan2(dy, dx))

    # Calculate angular difference to boresight
    angle_diffs = get_angle_diff(angles, boresight)

    # Filter points within FoV (120 degrees total, so +/- 60 degrees)
    in_fov = angle_diffs <= 60.0

    # Find max distance within FoV inside the room
    dists_in_fov = dists[in_fov]
    if len(dists_in_fov) > 0:
        max_dist_in_room = np.max(dists_in_fov)
        max_bin_in_room = int(np.ceil(max_dist_in_room / BIN_RESOLUTION))
    else:
        max_dist_in_room = 0.0
        max_bin_in_room = 0

    return max_dist_in_room, max_bin_in_room


def generate_visualization():
    """Generate and show the FoV and Range Bin visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 11), sharex=True, sharey=True)
    axes = axes.flatten()

    # Analyze each radar and plot
    analysis_results = {}
    for idx, (name, config) in enumerate(RADARS.items()):
        ax = axes[idx]
        xr, yr = config["pos"]
        boresight = config["angle"]

        # Calculate coverage boundaries
        max_dist, max_bin = analyze_sensor_coverage(name, config)
        analysis_results[name] = {
            "max_dist": max_dist,
            "max_bin": max_bin,
            "min_bin": DIRECT_COUPLING_BINS,
        }

        # Draw the room
        room_rect = Rectangle(
            (0, 0),
            ROOM_WIDTH,
            ROOM_HEIGHT,
            edgecolor="black",
            facecolor="none",
            linewidth=2,
            linestyle="--",
            label="Room Boundary",
        )
        ax.add_patch(room_rect)

        # Draw the Field of View Wedge for the entire 120 bins (18m)
        # Wedge arguments: center, radius, theta1, theta2
        theta1 = boresight - 60
        theta2 = boresight + 60

        # 1. Wasted/Out-of-Room Region (from max room distance to 18m)
        wasted_wedge = Wedge(
            (xr, yr),
            MAX_RANGE,
            theta1,
            theta2,
            facecolor="#d3d3d3",
            alpha=0.3,
            label="Wasted/Out-of-Room (> Max Room Range)",
        )
        ax.add_patch(wasted_wedge)

        # 2. Useful Room Region (from direct coupling range 0.75m to max room range)
        useful_wedge = Wedge(
            (xr, yr),
            max_dist,
            theta1,
            theta2,
            facecolor="#2ca02c",
            alpha=0.3,
            label="Useful Range Bins",
        )
        ax.add_patch(useful_wedge)

        # 3. Direct Coupling Region (bins 0-4, range 0 to 0.75m)
        direct_wedge = Wedge(
            (xr, yr),
            DIRECT_COUPLING_RANGE,
            theta1,
            theta2,
            facecolor="#d62728",
            alpha=0.4,
            label="Direct Coupling (Bins 0-4)",
        )
        ax.add_patch(direct_wedge)

        # Draw range bin rings (concentric arcs at intervals of 10 bins)
        for b in range(10, TOTAL_BINS + 1, 10):
            r = b * BIN_RESOLUTION
            # Only draw the arc within the 120-degree FoV
            # Generate points for the arc
            theta_span = np.linspace(np.radians(theta1), np.radians(theta2), 100)
            arc_x = xr + r * np.cos(theta_span)
            arc_y = yr + r * np.sin(theta_span)

            # Use solid line if inside the room/useful range, dashed if outside
            if r <= max_dist:
                ax.plot(
                    arc_x,
                    arc_y,
                    color="darkgreen",
                    linestyle=":",
                    alpha=0.5,
                    linewidth=0.8,
                )
                if b % 20 == 0 or b == 50:
                    # Label the bin distance
                    label_angle = np.radians(boresight - 15)
                    ax.text(
                        xr + (r - 0.2) * np.cos(label_angle),
                        yr + (r - 0.2) * np.sin(label_angle),
                        f"B{b}",
                        fontsize=12,
                        color="darkgreen",
                        ha="center",
                        va="center",
                    )
            else:
                ax.plot(
                    arc_x, arc_y, color="grey", linestyle="--", alpha=0.3, linewidth=0.8
                )

        # Plot sensor location and direction
        ax.plot(xr, yr, "o", color=config["color"], markersize=10, label="_nolegend_")
        ax.tick_params(axis="both", which="major", labelsize=12)
        # Draw boresight arrow
        arrow_len = 0.5
        ax.arrow(
            xr,
            yr,
            arrow_len * np.cos(np.radians(boresight)),
            arrow_len * np.sin(np.radians(boresight)),
            head_width=0.2,
            head_length=0.2,
            fc=config["color"],
            ec=config["color"],
        )

        # Titles and Labels
        ax.set_title(
            f"{name} ({boresight}° Boresight)\nUseful Bins: {DIRECT_COUPLING_BINS} to {max_bin} ({max_dist:.2f}m)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlim(-2, ROOM_WIDTH + 6)
        ax.set_ylim(-2, ROOM_HEIGHT + 2)
        ax.set_aspect("equal")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)

    # Place a single legend in the bottom-right space or coordinate it
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=14,
        bbox_to_anchor=(0.5, 0.02),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10)

    # Save the figure
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    out_dir = os.path.join(project_root, "report/images")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fov_visualization.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved visualization to {out_path}")

    return analysis_results


if __name__ == "__main__":
    print("Analyzing sensor coverage and generating visualization...")
    results = generate_visualization()

    print("\nAnalysis Summary Table:")
    print("-" * 80)
    print(
        f"{'Radar':<10} | {'Boresight':<10} | {'Max Room Dist':<15} | {'Max Active Bin':<15} | {'Truncation Saved':<15}"
    )
    print("-" * 80)
    for name, res in results.items():
        radar_angle = RADARS[name]["angle"]
        saved_bins = TOTAL_BINS - res["max_bin"]
        saved_pct = (saved_bins / TOTAL_BINS) * 100
        print(
            f"{name:<10} | {radar_angle:<9}° | {res['max_dist']:<13.2f}m | {res['max_bin']:<15} | {saved_bins} ({saved_pct:.1f}%)"
        )
    print("-" * 80)
