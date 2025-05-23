import matplotlib.pyplot as plt
import numpy as np
import json
import os
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def plot_feature_distributions(results_file, output_dir):
    """
    Plot distributions of features for cry vs non-cry samples
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Separate features by cry and non-cry
    cry_features = {
        'energy_ratio_main': [],
        'energy_ratio_low': [],
        'energy_ratio_mid': [],
        'energy_ratio_high': [],
        'energy_ratio_distribution': [],
        'energy_ratio_peak_freq': [],
        'rhythm_regularity': [],
        'duration': [],
        'amplitude_modulation': [],
        'noise_ratio': [],
        'music_ratio': [],
        'spectral_flux': [],
        'spectral_contrast': [],
        'harmonicity': [],
        'percussiveness': [],
        'spectral_centroid_mean': [],
        'spectral_bandwidth_mean': [],
        'spectral_rolloff_mean': [],
        'spectral_flatness_mean': [],
        'zcr_mean': [],
        'pattern_score': [],
        'rhythm_stability': [],
        'rhythm_complexity': [],
        'peak_prominence': [],
        'peak_ratio_mean': []
    }
    
    non_cry_features = {
        'energy_ratio_main': [],
        'energy_ratio_low': [],
        'energy_ratio_mid': [],
        'energy_ratio_high': [],
        'energy_ratio_distribution': [],
        'energy_ratio_peak_freq': [],
        'rhythm_regularity': [],
        'duration': [],
        'amplitude_modulation': [],
        'noise_ratio': [],
        'music_ratio': [],
        'spectral_flux': [],
        'spectral_contrast': [],
        'harmonicity': [],
        'percussiveness': [],
        'spectral_centroid_mean': [],
        'spectral_bandwidth_mean': [],
        'spectral_rolloff_mean': [],
        'spectral_flatness_mean': [],
        'zcr_mean': [],
        'pattern_score': [],
        'rhythm_stability': [],
        'rhythm_complexity': [],
        'peak_prominence': [],
        'peak_ratio_mean': []
    }
    
    # Collect features
    for dir_name, files in results.items():
        is_cry_dir = dir_name.startswith('Cry-')
        for file_result in files:
            features = file_result['features']
            target_dict = cry_features if is_cry_dir else non_cry_features
            for feature_name in target_dict.keys():
                try:
                    target_dict[feature_name].append(features.get(feature_name, 0))
                except (KeyError, TypeError):
                    print(f"Warning: Feature {feature_name} not found in results, using default value 0")
                    target_dict[feature_name].append(0)
    
    # Create plots
    feature_names = list(cry_features.keys())
    n_features = len(feature_names)
    
    # Calculate grid dimensions
    n_cols = 3  # Changed from 2 to 3 columns
    n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division
    
    # Create a grid of subplots
    fig = plt.figure(figsize=(20, 5 * n_rows))  # Adjusted figure height based on number of rows
    gs = GridSpec(n_rows, n_cols, figure=fig)
    
    # Plot feature distributions
    for i, feature in enumerate(feature_names):
        row = i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        # Plot histograms with KDE
        sns.histplot(data=cry_features[feature], label='Cry', alpha=0.5, kde=True)
        sns.histplot(data=non_cry_features[feature], label='No Cry', alpha=0.5, kde=True)
        
        # Add threshold line if applicable
        if feature in ['spectral_centroid_mean', 'zcr_mean', 'energy_ratio_main']:
            threshold = 0.25  # Current threshold
            ax.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
        
        ax.set_title(f'{feature.replace("_", " ").title()}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_distributions.png'))
    plt.close()

def plot_decision_boundaries(results_file, output_dir):
    """
    Plot decision boundaries using pairs of features and highlight misclassified points.
    """
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Updated feature combinations to match cry_detection.py logic
    feature_combinations = [
        ['energy_ratio_main', 'rhythm_regularity', 'amplitude_modulation'],
        ['energy_ratio_main', 'amplitude_modulation', 'music_ratio'],
        ['music_ratio', 'noise_ratio', 'amplitude_modulation']
    ]

    for i, features in enumerate(feature_combinations):
        cry_points = []
        non_cry_points = []
        misclassified_points = []
        misclassified_labels = []

        for dir_name, files in results.items():
            true_label = 'Cry' if dir_name.startswith('Cry-') else 'No Cry'
            for file_result in files:
                file_features = file_result['features']
                predicted_label = 'Cry' if file_result['is_cry'] else 'No Cry'
                point = [file_features.get(f, 0) for f in features]
                
                if predicted_label == true_label:
                    if predicted_label == 'Cry':
                        cry_points.append(point)
                    else:
                        non_cry_points.append(point)
                else:
                    misclassified_points.append(point)
                    misclassified_labels.append(file_result.get('filename', ''))

        cry_points = np.array(cry_points)
        non_cry_points = np.array(non_cry_points)
        misclassified_points = np.array(misclassified_points)

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        if len(cry_points) > 0:
            ax.scatter(cry_points[:, 0], cry_points[:, 1], cry_points[:, 2], 
                      c='r', marker='o', label='Cry (Correct)')
        if len(non_cry_points) > 0:
            ax.scatter(non_cry_points[:, 0], non_cry_points[:, 1], non_cry_points[:, 2], 
                      c='b', marker='x', label='No Cry (Correct)')
        if len(misclassified_points) > 0:
            ax.scatter(misclassified_points[:, 0], misclassified_points[:, 1], misclassified_points[:, 2], 
                      c='yellow', marker='*', s=180, edgecolor='k', label='Misclassified')
            # Annotate misclassified points
            for i, label in enumerate(misclassified_labels):
                ax.text(misclassified_points[i, 0], misclassified_points[i, 1], misclassified_points[i, 2], 
                        label, color='black', fontsize=8)

        ax.set_xlabel(features[0].replace('_', ' ').title())
        ax.set_ylabel(features[1].replace('_', ' ').title())
        ax.set_zlabel(features[2].replace('_', ' ').title())
        ax.set_title(f'Decision Boundaries in Feature Space {i+1}\n(Misclassifications Highlighted)')
        ax.legend()

        plt.savefig(os.path.join(output_dir, f'decision_boundaries_{i+1}.png'))
        plt.close()

def plot_feature_importance(results_file, output_dir):
    """
    Plot feature importance based on weights used in detection
    """
    # Updated weights to match cry_detection.py
    weights = {
        'energy_ratio_main': 0.25,
        'rhythm_regularity': 0.2,
        'duration': 0.15,
        'amplitude_modulation': 0.2,
        'noise_ratio': 0.1,
        'music_ratio': 0.1
    }
    
    plt.figure(figsize=(12, 6))
    features = list(weights.keys())
    importance = list(weights.values())
    
    # Create bar plot with custom colors
    colors = ['#2ecc71' if w >= 0.15 else '#3498db' for w in importance]
    bars = plt.bar(features, importance, color=colors)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.title('Feature Importance in Cry Detection')
    plt.xlabel('Features')
    plt.ylabel('Weight')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()

def plot_category_accuracy(results_file, output_dir):
    """
    Plot accuracy for each category
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    categories = []
    accuracies = []
    
    for dir_name, files in results.items():
        total = len(files)
        correct = sum(1 for f in files if (dir_name.startswith('Cry-') and f['is_cry']) or 
                     (not dir_name.startswith('Cry-') and not f['is_cry']))
        accuracy = correct / total if total > 0 else 0
        categories.append(dir_name)
        accuracies.append(accuracy)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(categories, accuracies)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}',
                ha='center', va='bottom')
    
    plt.title('Accuracy by Category')
    plt.xlabel('Category')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)  # Set y-axis limit to show percentages clearly
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'category_accuracy.png'))
    plt.close()

def create_visualization_report():
    """
    Create all visualizations and save them to the results directory
    """
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    results_file = os.path.join(results_dir, "all_results.json")
    
    # Create visualizations
    plot_feature_distributions(results_file, results_dir)
    plot_decision_boundaries(results_file, results_dir)
    plot_feature_importance(results_file, results_dir)
    plot_category_accuracy(results_file, results_dir)
    
    print("Visualization report generated in the results directory.")

if __name__ == "__main__":
    create_visualization_report() 