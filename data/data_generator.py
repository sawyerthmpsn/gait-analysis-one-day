import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)


def generate_gait_dataset(n_subjects=500, n_trials_per_subject=3):
    """
    Generate a comprehensive synthetic human gait analysis dataset

    Parameters:
    n_subjects: Number of unique subjects
    n_trials_per_subject: Number of gait trials per subject
    """

    data = []
    subject_id = 1

    # Define gait conditions
    gait_conditions = ['normal', 'fast', 'slow', 'pathological']
    pathology_types = ['healthy', 'stroke', 'parkinsons', 'cerebral_palsy', 'amputee', 'arthritis']

    for subject in range(n_subjects):
        # Generate subject demographics
        age = np.random.normal(45, 20)
        age = max(18, min(85, age))  # Constrain age between 18-85

        sex = random.choice(['M', 'F'])
        height = np.random.normal(170 if sex == 'M' else 160, 8)  # cm
        weight = np.random.normal(75 if sex == 'M' else 65, 12)  # kg
        bmi = weight / (height / 100) ** 2

        # Determine if subject has pathology
        has_pathology = random.choice([True, False])
        pathology = random.choice(pathology_types[1:]) if has_pathology else 'healthy'

        # Generate multiple trials for each subject
        for trial in range(n_trials_per_subject):
            # Select gait condition
            if pathology != 'healthy':
                condition = random.choice(['normal', 'slow', 'pathological'])
            else:
                condition = random.choice(['normal', 'fast', 'slow'])

            # Generate temporal parameters
            if condition == 'fast':
                base_cadence = np.random.normal(125, 10)
            elif condition == 'slow':
                base_cadence = np.random.normal(95, 8)
            elif condition == 'pathological':
                base_cadence = np.random.normal(85, 15)
            else:  # normal
                base_cadence = np.random.normal(110, 10)

            # Adjust for age and pathology
            age_factor = 1 - (age - 20) * 0.003 if age > 20 else 1
            pathology_factor = 0.8 if has_pathology else 1

            cadence = base_cadence * age_factor * pathology_factor
            cadence = max(60, min(180, cadence))  # Physiological limits

            step_time = 60 / cadence  # seconds
            stride_time = step_time * 2

            # Generate spatial parameters
            leg_length = height * 0.53  # Approximate leg length

            if condition == 'fast':
                stride_length = np.random.normal(leg_length * 0.85, leg_length * 0.05)
            elif condition == 'slow':
                stride_length = np.random.normal(leg_length * 0.65, leg_length * 0.05)
            elif condition == 'pathological':
                stride_length = np.random.normal(leg_length * 0.55, leg_length * 0.08)
            else:  # normal
                stride_length = np.random.normal(leg_length * 0.75, leg_length * 0.05)

            step_length = stride_length / 2
            step_width = np.random.normal(8, 2)  # cm
            walking_speed = stride_length * cadence / 120  # m/s

            # Generate kinematic data (joint angles in degrees)
            # Hip angles
            hip_flexion_max = np.random.normal(30, 5)
            hip_extension_max = np.random.normal(-10, 3)
            hip_abduction_max = np.random.normal(5, 2)

            # Knee angles
            knee_flexion_max = np.random.normal(60, 8)
            knee_extension_max = np.random.normal(5, 3)

            # Ankle angles
            ankle_dorsiflexion_max = np.random.normal(15, 4)
            ankle_plantarflexion_max = np.random.normal(-20, 5)

            # Adjust for pathology
            if pathology == 'stroke':
                knee_flexion_max *= 0.7
                ankle_dorsiflexion_max *= 0.6
            elif pathology == 'parkinsons':
                hip_flexion_max *= 0.8
                step_length *= 0.8
            elif pathology == 'cerebral_palsy':
                knee_flexion_max *= 1.3
                ankle_plantarflexion_max *= 1.2

            # Generate kinetic data (forces and moments)
            body_weight = weight * 9.81  # Convert to Newtons

            # Ground reaction forces (normalized to body weight)
            vertical_grf_max = np.random.normal(1.1, 0.1) * body_weight
            anterior_grf_max = np.random.normal(0.15, 0.03) * body_weight
            posterior_grf_max = np.random.normal(-0.18, 0.03) * body_weight
            medial_grf_max = np.random.normal(0.05, 0.02) * body_weight

            # Joint moments (Nm/kg)
            hip_moment_max = np.random.normal(1.2, 0.2)
            knee_moment_max = np.random.normal(0.8, 0.15)
            ankle_moment_max = np.random.normal(1.5, 0.25)

            # Calculate additional derived parameters
            double_support_time = np.random.normal(0.12, 0.02)  # seconds
            single_support_time = (stride_time - 2 * double_support_time) / 2
            swing_time = stride_time - (single_support_time + double_support_time)
            stance_time = single_support_time + double_support_time

            # Symmetry indices
            step_length_symmetry = np.random.normal(1.0, 0.05)
            step_time_symmetry = np.random.normal(1.0, 0.03)

            # Add some noise and pathological variations
            if has_pathology:
                step_length_symmetry *= np.random.normal(1.0, 0.1)
                step_time_symmetry *= np.random.normal(1.0, 0.08)

            # Compile all data for this trial
            trial_data = {
                'subject_id': subject_id,
                'trial_number': trial + 1,
                'age': round(age, 1),
                'sex': sex,
                'height_cm': round(height, 1),
                'weight_kg': round(weight, 1),
                'bmi': round(bmi, 1),
                'pathology': pathology,
                'gait_condition': condition,

                # Temporal parameters
                'cadence_steps_per_min': round(cadence, 1),
                'step_time_s': round(step_time, 3),
                'stride_time_s': round(stride_time, 3),
                'stance_time_s': round(stance_time, 3),
                'swing_time_s': round(swing_time, 3),
                'double_support_time_s': round(double_support_time, 3),
                'single_support_time_s': round(single_support_time, 3),

                # Spatial parameters
                'step_length_cm': round(step_length, 1),
                'stride_length_cm': round(stride_length, 1),
                'step_width_cm': round(step_width, 1),
                'walking_speed_m_s': round(walking_speed, 2),

                # Kinematic parameters (degrees)
                'hip_flexion_max_deg': round(hip_flexion_max, 1),
                'hip_extension_max_deg': round(hip_extension_max, 1),
                'hip_abduction_max_deg': round(hip_abduction_max, 1),
                'knee_flexion_max_deg': round(knee_flexion_max, 1),
                'knee_extension_max_deg': round(knee_extension_max, 1),
                'ankle_dorsiflexion_max_deg': round(ankle_dorsiflexion_max, 1),
                'ankle_plantarflexion_max_deg': round(ankle_plantarflexion_max, 1),

                # Kinetic parameters
                'vertical_grf_max_N': round(vertical_grf_max, 1),
                'anterior_grf_max_N': round(anterior_grf_max, 1),
                'posterior_grf_max_N': round(posterior_grf_max, 1),
                'medial_grf_max_N': round(medial_grf_max, 1),
                'hip_moment_max_Nm_kg': round(hip_moment_max, 2),
                'knee_moment_max_Nm_kg': round(knee_moment_max, 2),
                'ankle_moment_max_Nm_kg': round(ankle_moment_max, 2),

                # Symmetry and variability
                'step_length_symmetry_ratio': round(step_length_symmetry, 3),
                'step_time_symmetry_ratio': round(step_time_symmetry, 3),

                # Additional metrics
                'gait_deviation_index': round(np.random.normal(100 if pathology == 'healthy' else 120, 15), 1),
                'energy_cost_J_kg_m': round(np.random.normal(3.2 if pathology == 'healthy' else 4.1, 0.5), 2),

                # Study metadata
                'collection_date': (datetime.now() - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d'),
                'lab_id': f"LAB_{random.randint(1, 5):02d}",
                'equipment_type': random.choice(['Vicon', 'OptiTrack', 'Xsens', 'AMTI'])
            }

            data.append(trial_data)

        subject_id += 1

    return pd.DataFrame(data)


# Generate the dataset
print("Generating synthetic gait analysis dataset...")
df = generate_gait_dataset(n_subjects=500, n_trials_per_subject=3)

print(f"Dataset generated with {len(df)} trials from {df['subject_id'].nunique()} subjects")
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {len(df.columns)}")

# Display basic statistics
print("\n" + "=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print(f"\nSubject Demographics:")
print(f"Age range: {df['age'].min():.1f} - {df['age'].max():.1f} years")
print(f"Sex distribution: {df['sex'].value_counts().to_dict()}")
print(f"Pathology distribution:")
for pathology, count in df['pathology'].value_counts().items():
    print(f"  {pathology}: {count} trials")

print(f"\nGait Condition Distribution:")
for condition, count in df['gait_condition'].value_counts().items():
    print(f"  {condition}: {count} trials")

print(f"\nKey Gait Parameters (Mean ± SD):")
print(f"Walking speed: {df['walking_speed_m_s'].mean():.2f} ± {df['walking_speed_m_s'].std():.2f} m/s")
print(f"Cadence: {df['cadence_steps_per_min'].mean():.1f} ± {df['cadence_steps_per_min'].std():.1f} steps/min")
print(f"Stride length: {df['stride_length_cm'].mean():.1f} ± {df['stride_length_cm'].std():.1f} cm")
print(f"Step width: {df['step_width_cm'].mean():.1f} ± {df['step_width_cm'].std():.1f} cm")

# Save to CSV
filename = 'synthetic_gait_analysis_dataset.csv'
df.to_csv(filename, index=False)
print(f"\nDataset saved as '{filename}'")

# Display first few rows
print(f"\nFirst 5 rows of the dataset:")
print(df.head())

# Show column information
print(f"\nColumn Information:")
print(df.dtypes)
