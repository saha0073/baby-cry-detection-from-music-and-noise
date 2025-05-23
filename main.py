import os
import json
import numpy as np
from cry_detection import CryDetector

def normalize_feature(value, min_val, max_val):
    """Normalize a feature value to [0, 1] range"""
    return (value - min_val) / (max_val - min_val) if max_val > min_val else 0

def convert_to_serializable(obj):
    """Convert objects to JSON serializable format"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, bool):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj

def main():
    # Define input directories
    input_directories = [
        "Cry-NoNoise-Music",
        "NoCry-NoNoise-Music",
        "Cry-NoNoise-NoMusic",
        "Cry-Noise-NoMusic",
        "NoCry-Noise-NoMusic"
    ]
    
    # Create results directory
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Initialize cry detector
    detector = CryDetector()
    
    # Process each directory
    all_results = {}
    total_files = 0
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    
    # Track results by category
    category_results = {
        'Cry-NoNoise-Music': {'correct': 0, 'total': 0},
        'NoCry-NoNoise-Music': {'correct': 0, 'total': 0},
        'Cry-NoNoise-NoMusic': {'correct': 0, 'total': 0},
        'Cry-Noise-NoMusic': {'correct': 0, 'total': 0},
        'NoCry-Noise-NoMusic': {'correct': 0, 'total': 0}
    }
    
    for input_dir in input_directories:
        print(f"\nProcessing directory: {input_dir}")
        print("-" * 50)
        
        # Get path to features file
        features_file = os.path.join("extracted_features", f"{input_dir}_features.json")
        dir_results = []
        
        try:
            # Load features for this directory
            print(f"Loading features from: {features_file}")
            with open(features_file, 'r') as f:
                features_dict = json.load(f)
            print(f"Loaded {len(features_dict)} entries from {features_file}")
            
            # Process each file's features
            for file_entry in features_dict:
                filename = file_entry['filename']
                features = file_entry['features']
                # Pass the full features dict to the detector
                detection_result = detector.detect_cry_from_features(features)
                # Store results (without scores dictionary)
                result = {
                    'filename': filename,
                    'is_cry': int(detection_result['is_cry']),  # Convert bool to int
                    'confidence': detection_result['confidence'],
                    'features': features
                }
                dir_results.append(result)
                
                # Update accuracy metrics
                total_files += 1
                expected_cry = input_dir.startswith('Cry-')
                predicted_cry = detection_result['is_cry']
                
                # Update category results
                category_results[input_dir]['total'] += 1
                if expected_cry == predicted_cry:
                    category_results[input_dir]['correct'] += 1
                
                if expected_cry and predicted_cry:
                    true_positives += 1
                elif not expected_cry and not predicted_cry:
                    true_negatives += 1
                elif not expected_cry and predicted_cry:
                    false_positives += 1
                else:  # expected_cry and not predicted_cry
                    false_negatives += 1
                
                print(f"File: {filename}")
                print(f"Directory: {input_dir}")
                print(f"Expected: {'Cry' if expected_cry else 'No Cry'}")
                print(f"Predicted: {'Cry' if predicted_cry else 'No Cry'}")
                print(f"Confidence: {detection_result['confidence']:.2f}")
                print("Individual Scores:")
                for score_name, score_value in detection_result['scores'].items():
                    print(f"  {score_name}: {score_value}")
                
                # Add detailed logging output
                if 'detailed_log' in detection_result:
                    print("\nDetailed Feature Analysis:")
                    print("Energy Conditions:")
                    for condition in detection_result['detailed_log']['energy_conditions']:
                        print(f"  {condition}")
                    
                    print("\nRhythm Conditions:")
                    for condition in detection_result['detailed_log']['rhythm_conditions']:
                        print(f"  {condition}")
                    
                    print("\nModulation Conditions:")
                    for condition in detection_result['detailed_log']['modulation_conditions']:
                        print(f"  {condition}")
                    
                    print("\nSpectral Conditions:")
                    for condition in detection_result['detailed_log']['spectral_conditions']:
                        print(f"  {condition}")
                    
                    print("\nMFCC Conditions:")
                    for condition in detection_result['detailed_log']['mfcc_conditions']:
                        print(f"  {condition}")
                    
                    print("\nScore Modifications:")
                    for modification in detection_result['detailed_log']['score_modifications']:
                        print(f"  {modification}")
                    
                    print("\nFeature Values:")
                    print("  Energy Ratios:")
                    for name, value in detection_result['detailed_log']['feature_values']['energy_ratios'].items():
                        print(f"    {name}: {value}")
                    
                    print("  Rhythm Features:")
                    for name, value in detection_result['detailed_log']['feature_values']['rhythm_features'].items():
                        print(f"    {name}: {value}")
                    
                    print("  Modulation Features:")
                    for name, value in detection_result['detailed_log']['feature_values']['modulation_features'].items():
                        print(f"    {name}: {value}")
                    
                    print("  Interference Features:")
                    for name, value in detection_result['detailed_log']['feature_values']['interference'].items():
                        print(f"    {name}: {value}")
                
                print("-" * 30)
            
            all_results[input_dir] = dir_results
            
            # Save results for this directory
            results_file = os.path.join(results_dir, f"{input_dir}_results.json")
            with open(results_file, 'w') as f:
                json.dump(dir_results, f, indent=4)
            
            print(f"Successfully processed all files in {input_dir}")
            
        except Exception as e:
            print(f"Error processing {input_dir}: {str(e)}")
    
    # Save all results
    with open(os.path.join(results_dir, "all_results.json"), 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # Calculate and save accuracy metrics
    accuracy = (true_positives + true_negatives) / total_files if total_files > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'total_files': total_files,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'category_results': category_results
    }
    
    with open(os.path.join(results_dir, "accuracy_metrics.json"), 'w') as f:
        json.dump(convert_to_serializable(metrics), f, indent=4)
    
    print("\nProcessing complete!")
    print(f"Results saved in: {results_dir}")
    print("\nAccuracy Metrics:")
    print(f"Total files processed: {total_files}")
    print(f"True Positives: {true_positives}")
    print(f"True Negatives: {true_negatives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1_score:.2%}")
    
    print("\nCategory-wise Results:")
    for category, results in category_results.items():
        accuracy = results['correct'] / results['total'] if results['total'] > 0 else 0
        print(f"{category}: {accuracy:.2%} ({results['correct']}/{results['total']})")

if __name__ == "__main__":
    main() 