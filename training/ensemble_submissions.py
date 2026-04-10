import numpy as np
from sklearn.metrics import roc_auc_score
import os
import zipfile
import argparse

def evaluate_weighted_ensemble(submission_files, weights, output_file):
    try:
            
        # 2. Read the predictions from all submission files
        all_preds = []
        for file_path in submission_files:
            with open(file_path, 'r') as f:
                preds = [float(line.strip()) for line in f if line.strip()]
                all_preds.append(preds)
                
        # Convert to a 2D numpy array (shape: number of files x number of lines)
        all_preds = np.array(all_preds)

        # 4. Calculate the weighted average
        # np.average automatically handles dividing by the sum of the weights
        weighted_preds = np.average(all_preds, axis=0, weights=weights)
        
        # 5. Round to 1 decimal place (using Round Half to Even for math)
        rounded_preds = np.round(weighted_preds, 1)
        
        # 7. (Optional) Save the rounded ensemble predictions ready for submission
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        if output_file:
            with open(output_file, 'w') as f:
                for p in rounded_preds:
                    # Using the strict string formatting rule for the text file
                    f.write(f"{float(p):.1f}\n")
            print(f"Saved ensemble predictions to: {output_file}")

    except Exception as e:
        print(f"Error during ensemble calculation: {e}")
        return None

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a weighted ensemble from multiple submission files.")
    
    # nargs='+' allows you to pass multiple file paths separated by spaces
    parser.add_argument(
        "--sub_files", 
        nargs='+', 
        required=True, 
        help="List of paths to the submission.txt files."
    )
    
    # Also making weights an argument, defaulting to your [2, 1, 2] if not provided
    parser.add_argument(
        "--weights", 
        nargs='+', 
        type=float, 
        default=[2.0, 1.0, 2.0], 
        help="Weights for the ensemble (must match the number of sub_files)."
    )
    
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="/raid/dtle/NTIRE26-DeepfakeDetection/submissions_test/ensemble_submission/submission.txt", 
        help="Path to save the ensembled submission.txt"
    )

    args = parser.parse_args()

    # Safety check
    if len(args.sub_files) != len(args.weights):
        raise ValueError(f"You provided {len(args.sub_files)} files but {len(args.weights)} weights. They must match.")

    # Run the evaluation
    evaluate_weighted_ensemble(args.sub_files, args.weights, args.output_file)

    # Zip the submission.txt file and save the ensemble predictions
    zip_path = os.path.join(os.path.dirname(args.output_file), 'ensemble_submission.zip')
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(args.output_file, os.path.basename(args.output_file))
        
    print(f"Saved zip file to: {zip_path}")
