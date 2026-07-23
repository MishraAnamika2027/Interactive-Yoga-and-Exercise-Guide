"""Analyze training pose images to extract joint angle statistics.

Usage:
    python analyze_training_data.py [--input Our_trained_data] [--output-prefix training_analysis_results]

Outputs:
    - <output-prefix>.json : Statistical summary of joint angles per pose type.
    - <output-prefix>.csv  : Flattened table of joint angle statistics per pose type.

Requirements:
    - opencv-python
    - mediapipe
    - numpy

This script processes static training images for T Pose and Tree Pose, computes
pose landmarks using MediaPipe, extracts key joint angles, and summarizes the
results. The statistics can be used to tune the handcrafted thresholds in
`classifyPose()` within the Flask application.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import mediapipe as mp
import numpy as np

# Type aliases for clarity
AngleName = str
PoseName = str

mp_pose = mp.solutions.pose


def calculate_angle(point1: np.ndarray, point2: np.ndarray, point3: np.ndarray) -> float:
    """Calculate the angle formed by three points in degrees.

    The angle is computed at ``point2`` using the vector from ``point2`` to
    ``point1`` and the vector from ``point2`` to ``point3``. The result is
    normalized to the range [0, 360).

    Args:
        point1: First landmark as a NumPy array of shape (3,).
        point2: Vertex landmark where the angle is measured.
        point3: Third landmark.

    Returns:
        Angle in degrees between 0 (inclusive) and 360 (exclusive).
    """

    vector1 = point1 - point2
    vector2 = point3 - point2

    radians = math.atan2(vector2[1], vector2[0]) - math.atan2(vector1[1], vector1[0])
    angle = math.degrees(radians)
    angle = angle + 360 if angle < 0 else angle
    return angle


def load_training_images(folder_path: Path) -> Dict[PoseName, List[Path]]:
    """Load training images grouped by pose type based on filename prefixes."""

    pose_images: Dict[PoseName, List[Path]] = {"T_Pose": [], "Tree_Pose": []}

    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Training data folder not found: {folder_path}")

    for pose_type in pose_images.keys():
        pose_images[pose_type] = sorted(folder_path.glob(f"{pose_type}_*.jpg"))

    total_images = sum(len(paths) for paths in pose_images.values())
    if total_images == 0:
        raise FileNotFoundError(
            f"No training images found in {folder_path}. Expected files like 'T_Pose_0.jpg'."
        )

    return pose_images


def detect_pose_landmarks(image_path: Path, pose_detector: "mp.solutions.pose.Pose") -> Optional[np.ndarray]:
    """Detect pose landmarks for a static image using MediaPipe Pose."""

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise IOError(f"Unable to read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(image_rgb)

    if not results.pose_landmarks:
        return None

    height, width, _ = image_bgr.shape
    landmarks = []
    for landmark in results.pose_landmarks.landmark:
        landmarks.append(np.array([landmark.x * width, landmark.y * height, landmark.z * width]))
    return np.array(landmarks)


def extract_joint_angles(landmarks: np.ndarray) -> Dict[AngleName, Optional[float]]:
    """Extract key joint angles from pose landmarks."""

    def get_landmark(name: "mp.solutions.pose.PoseLandmark") -> Optional[np.ndarray]:
        index = name.value
        if index >= len(landmarks):
            return None
        return landmarks[index]

    joint_map = {
        "left_elbow": (
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_ELBOW,
            mp_pose.PoseLandmark.LEFT_WRIST,
        ),
        "right_elbow": (
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_WRIST,
        ),
        "left_shoulder": (
            mp_pose.PoseLandmark.LEFT_ELBOW,
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_HIP,
        ),
        "right_shoulder": (
            mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
        ),
        "left_knee": (
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.LEFT_KNEE,
            mp_pose.PoseLandmark.LEFT_ANKLE,
        ),
        "right_knee": (
            mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.RIGHT_KNEE,
            mp_pose.PoseLandmark.RIGHT_ANKLE,
        ),
    }

    angles: Dict[AngleName, Optional[float]] = {}

    for angle_name, (idx1, idx2, idx3) in joint_map.items():
        point1, point2, point3 = get_landmark(idx1), get_landmark(idx2), get_landmark(idx3)
        if point1 is None or point2 is None or point3 is None:
            angles[angle_name] = None
            continue
        angles[angle_name] = calculate_angle(point1, point2, point3)

    return angles


def calculate_statistics(samples: Iterable[float]) -> Dict[str, Optional[float]]:
    """Compute summary statistics for a collection of angle samples."""

    values = np.array([value for value in samples if value is not None], dtype=np.float64)
    if values.size == 0:
        return {"min": None, "max": None, "mean": None, "std": None, "median": None, "count": 0}

    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "median": float(np.median(values)),
        "count": int(values.size),
    }


def analyze_training_data(folder_path: Path, verbose: bool = True) -> Dict[PoseName, Dict[str, object]]:
    """Process all training images and compute joint angle statistics."""

    pose_images = load_training_images(folder_path)
    if verbose:
        print(f"Found {len(pose_images['T_Pose'])} T Pose images and {len(pose_images['Tree_Pose'])} Tree Pose images.")

    angle_names = [
        "left_elbow",
        "right_elbow",
        "left_shoulder",
        "right_shoulder",
        "left_knee",
        "right_knee",
    ]

    results: Dict[PoseName, Dict[str, object]] = {}

    with mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5) as pose_detector:
        for pose_type, images in pose_images.items():
            if verbose:
                print(f"\nAnalyzing {pose_type.replace('_', ' ')} images...")

            collected_angles: Dict[AngleName, List[Optional[float]]] = {name: [] for name in angle_names}
            for index, image_path in enumerate(images, start=1):
                if verbose:
                    print(f"  Processing {image_path.name}... [{index}/{len(images)}]")

                try:
                    landmarks = detect_pose_landmarks(image_path, pose_detector)
                except Exception as exc:  # broad catch to continue processing other images
                    print(f"    Error processing {image_path.name}: {exc}")
                    continue

                if landmarks is None:
                    print(f"    No pose detected in {image_path.name}.")
                    continue

                joint_angles = extract_joint_angles(landmarks)
                for angle_name, value in joint_angles.items():
                    collected_angles[angle_name].append(value)

            stats = {angle: calculate_statistics(values) for angle, values in collected_angles.items()}
            results[pose_type] = {
                "image_count": len(images),
                "angles": stats,
                "samples": collected_angles,
            }

    return results


def export_to_json(results: Dict[PoseName, Dict[str, object]], output_path: Path) -> None:
    """Persist analysis results to a JSON file."""

    serializable_results = {}
    for pose_type, data in results.items():
        serializable_results[pose_type] = {
            "image_count": data["image_count"],
            "angles": data["angles"],
        }

    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(serializable_results, json_file, indent=4)
    print(f"JSON results exported to: {output_path}")


def export_to_csv(results: Dict[PoseName, Dict[str, object]], output_path: Path) -> None:
    """Persist analysis results to a CSV file."""

    fieldnames = ["Pose_Type", "Joint", "Min", "Max", "Mean", "Std", "Median", "Sample_Count"]
    with output_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for pose_type, data in results.items():
            for joint, stats in data["angles"].items():
                writer.writerow(
                    {
                        "Pose_Type": pose_type,
                        "Joint": joint,
                        "Min": stats["min"],
                        "Max": stats["max"],
                        "Mean": stats["mean"],
                        "Std": stats["std"],
                        "Median": stats["median"],
                        "Sample_Count": stats["count"],
                    }
                )
    print(f"CSV results exported to: {output_path}")


def generate_comparison_report(results: Dict[PoseName, Dict[str, object]], verbose: bool = True) -> None:
    """Print a comparison between measured angles and current classification thresholds."""

    if not verbose:
        return

    current_thresholds = {
        "T_Pose": {
            "left_knee": (165, 195),
            "right_knee": (165, 195),
            "left_elbow": (130, 180),
            "right_elbow": (175, 220),
            "left_shoulder": (100, 200),
            "right_shoulder": (50, 130),
        },
        "Tree_Pose": {
            "left_knee": (165, 195),
            "right_knee": (25, 45),
            "left_elbow": None,
            "right_elbow": None,
            "left_shoulder": None,
            "right_shoulder": None,
        },
    }

    print("\n================ Pose Threshold Comparison ================")

    for pose_type, data in results.items():
        print(f"\n=== {pose_type.replace('_', ' ').upper()} ===")
        angles = data.get("angles", {})
        thresholds = current_thresholds.get(pose_type, {})

        for joint, stats in angles.items():
            current_range = thresholds.get(joint)
            print(f"  Joint: {joint.replace('_', ' ').title()}")
            if current_range is not None:
                print(f"    Current threshold: {current_range[0]}° - {current_range[1]}°")
            else:
                print("    Current threshold: Not defined / not enforced")

            if stats["count"] == 0:
                print("    No detected samples to analyze.")
                continue

            measured_range = f"{stats['min']:.2f}° - {stats['max']:.2f}°"
            print(f"    Detected range : {measured_range} (mean: {stats['mean']:.2f}°, std: {stats['std']:.2f}°)")

            recommended_low = stats["mean"] - 2 * stats["std"] if stats["std"] is not None else None
            recommended_high = stats["mean"] + 2 * stats["std"] if stats["std"] is not None else None

            if recommended_low is not None and recommended_high is not None:
                print(
                    "    Suggested threshold (mean +/- 2*std): "
                    f"{recommended_low:.2f}° - {recommended_high:.2f}°"
                )

            if current_range is not None and recommended_low is not None:
                if stats["min"] < current_range[0] or stats["max"] > current_range[1]:
                    print("    Recommendation: Consider adjusting thresholds to cover detected range.")
                else:
                    print("    Recommendation: Current thresholds encompass detected samples.")
            else:
                print("    Recommendation: Review thresholds based on collected data.")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Analyze training pose image angles.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Our_trained_data"),
        help="Path to folder containing training images (default: Our_trained_data)",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("training_analysis_results"),
        help="Prefix for output files (default: training_analysis_results)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for analysis workflow."""

    args = parse_arguments()
    input_path: Path = args.input
    output_prefix: Path = args.output_prefix
    verbose = not args.quiet

    if verbose:
        print(f"Loading training images from {input_path.resolve()}...")

    try:
        results = analyze_training_data(input_path, verbose=verbose)
    except FileNotFoundError as exc:
        print(exc)
        return

    total_images = sum(data["image_count"] for data in results.values())

    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")

    export_to_json(results, json_path)
    export_to_csv(results, csv_path)
    generate_comparison_report(results, verbose=verbose)

    if verbose:
        print(
            f"\nAnalysis complete! Processed {total_images} images "
            f"({results['T_Pose']['image_count']} T Pose, {results['Tree_Pose']['image_count']} Tree Pose)."
        )
        print(f"Results saved to: {json_path.resolve()} and {csv_path.resolve()}")

if __name__ == "__main__":
    main()
