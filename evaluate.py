"""
Comprehensive Evaluation Script for FaceNet Implementation
Compares old vs new implementation and generates detailed reports
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
from datetime import datetime
from tqdm import tqdm
import json

# Import new FaceNet implementation
from facenet import FaceNetModel


class FaceNetEvaluator:
    """Comprehensive evaluator for FaceNet models"""
    
    def __init__(self, model_path=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Evaluation Device: {self.device}")
        
        # Initialize model
        self.model = FaceNetModel(device=device)
        
        # Load model if provided
        if model_path and os.path.exists(model_path):
            self.model.load_model(model_path)
            print(f"✅ Model loaded from: {model_path}")
        
        self.results = {}
    
    def evaluate_on_directory(self, test_dir, threshold=0.6, max_images=None):
        """
        Evaluate model on a test directory
        Args:
            test_dir: Directory containing test images
            threshold: Similarity threshold for classification
            max_images: Maximum number of images to test (None = all)
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n{'='*80}")
        print(f"🧪 EVALUATING ON: {test_dir}")
        print(f"{'='*80}")
        
        # Get all test images
        test_images = []
        for root, dirs, files in os.walk(test_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    test_images.append(os.path.join(root, file))
        
        if max_images:
            test_images = test_images[:max_images]
        
        print(f"📊 Total test images: {len(test_images)}")
        
        # Predictions
        predictions = []
        ground_truths = []
        similarities = []
        no_face_count = 0
        unknown_count = 0
        
        print(f"\n🔍 Running predictions...\n")
        
        for img_path in tqdm(test_images, desc="Processing"):
            # Get ground truth (folder name)
            gt_label = os.path.basename(os.path.dirname(img_path))
            
            # Predict
            pred_name, similarity = self.model.predict(img_path, threshold=threshold)
            
            # Track results
            if pred_name == "No Face Detected":
                no_face_count += 1
                continue
            elif pred_name == "Unknown":
                unknown_count += 1
            
            predictions.append(pred_name)
            ground_truths.append(gt_label)
            similarities.append(similarity)
        
        # Calculate metrics
        results = {
            'total_images': len(test_images),
            'processed_images': len(predictions),
            'no_face_detected': no_face_count,
            'unknown_predictions': unknown_count,
            'threshold': threshold,
            'predictions': predictions,
            'ground_truths': ground_truths,
            'similarities': similarities
        }
        
        # Calculate accuracy, precision, recall, f1
        if len(predictions) > 0:
            accuracy = accuracy_score(ground_truths, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                ground_truths, predictions, average='weighted', zero_division=0
            )
            
            results.update({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'avg_similarity': np.mean(similarities) if similarities else 0,
                'std_similarity': np.std(similarities) if similarities else 0
            })
        
        self.results = results
        return results
    
    def evaluate_with_validation_split(self, data_dir, test_size=0.2, threshold=0.6):
        """
        Evaluate using train/val split
        Args:
            data_dir: Root directory with person folders
            test_size: Validation split ratio
            threshold: Similarity threshold
        Returns:
            Evaluation results
        """
        print(f"\n{'='*80}")
        print(f"🧪 EVALUATION WITH VALIDATION SPLIT")
        print(f"{'='*80}")
        
        # Collect all images
        all_images = []
        all_labels = []
        
        for person_name in os.listdir(data_dir):
            person_dir = os.path.join(data_dir, person_name)
            if not os.path.isdir(person_dir):
                continue
            
            for img_file in os.listdir(person_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_images.append(os.path.join(person_dir, img_file))
                    all_labels.append(person_name)
        
        # Split
        train_images, val_images, train_labels, val_labels = train_test_split(
            all_images, all_labels,
            test_size=test_size,
            random_state=42,
            stratify=all_labels if len(set(all_labels)) > 1 else None
        )
        
        print(f"\n📊 Data Split:")
        print(f"   Training: {len(train_images)} images")
        print(f"   Validation: {len(val_images)} images")
        print(f"   Classes: {len(set(all_labels))}")
        
        # Evaluate on validation set
        predictions = []
        ground_truths = []
        similarities = []
        no_face = 0
        unknown = 0
        
        print(f"\n🔍 Evaluating on validation set...\n")
        
        for img_path, gt_label in tqdm(zip(val_images, val_labels), total=len(val_images), desc="Validating"):
            pred_name, similarity = self.model.predict(img_path, threshold=threshold)
            
            if pred_name == "No Face Detected":
                no_face += 1
                continue
            elif pred_name == "Unknown":
                unknown += 1
            
            predictions.append(pred_name)
            ground_truths.append(gt_label)
            similarities.append(similarity)
        
        # Metrics
        results = {
            'validation_size': len(val_images),
            'processed': len(predictions),
            'no_face': no_face,
            'unknown': unknown,
            'threshold': threshold
        }
        
        if len(predictions) > 0:
            accuracy = accuracy_score(ground_truths, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                ground_truths, predictions, average='weighted', zero_division=0
            )
            
            results.update({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'avg_similarity': np.mean(similarities),
                'std_similarity': np.std(similarities),
                'predictions': predictions,
                'ground_truths': ground_truths,
                'similarities': similarities
            })
        
        self.results = results
        return results
    
    def print_results(self):
        """Print evaluation results"""
        if not self.results:
            print("⚠️  No results to display. Run evaluation first.")
            return
        
        print(f"\n{'='*80}")
        print(f"📊 EVALUATION RESULTS")
        print(f"{'='*80}")
        
        for key, value in self.results.items():
            if key in ['predictions', 'ground_truths', 'similarities']:
                continue
            
            if isinstance(value, float):
                if 'accuracy' in key.lower() or 'precision' in key.lower() or \
                   'recall' in key.lower() or 'f1' in key.lower():
                    print(f"   {key}: {value*100:.2f}%")
                else:
                    print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
        
        print(f"{'='*80}")
    
    def plot_confusion_matrix(self, save_path='confusion_matrix.png'):
        """Plot confusion matrix"""
        if 'predictions' not in self.results or 'ground_truths' not in self.results:
            print("⚠️  No predictions to plot.")
            return
        
        preds = self.results['predictions']
        gts = self.results['ground_truths']
        
        # Get unique labels
        labels = sorted(set(gts + preds))
        
        # Compute confusion matrix
        cm = confusion_matrix(gts, preds, labels=labels)
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to: {save_path}")
        plt.show()
    
    def plot_similarity_distribution(self, save_path='similarity_distribution.png'):
        """Plot similarity score distribution"""
        if 'similarities' not in self.results:
            print("⚠️  No similarity scores to plot.")
            return
        
        similarities = self.results['similarities']
        
        plt.figure(figsize=(10, 6))
        plt.hist(similarities, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(self.results['threshold'], color='red', linestyle='--', 
                   linewidth=2, label=f"Threshold: {self.results['threshold']}")
        plt.xlabel('Similarity Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Similarity Score Distribution', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Similarity distribution saved to: {save_path}")
        plt.show()
    
    def save_results(self, save_path='evaluation_results.json'):
        """Save results to JSON"""
        if not self.results:
            print("⚠️  No results to save.")
            return
        
        # Prepare results for JSON (remove non-serializable items)
        json_results = {k: v for k, v in self.results.items() 
                       if k not in ['predictions', 'ground_truths', 'similarities']}
        
        json_results['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(save_path, 'w') as f:
            json.dump(json_results, f, indent=4)
        
        print(f"✅ Results saved to: {save_path}")


def main():
    """Main evaluation pipeline"""
    print("="*80)
    print("🔬 FACENET COMPREHENSIVE EVALUATION")
    print("="*80)
    
    # ==================== CONFIGURATION ====================
    MODEL_PATH = "./models/facenet_model_20251201_144047.pkl"
    DATA_DIR = "Train/train"
    THRESHOLD = 0.6
    DEVICE = 'cuda'
    
    # Evaluation mode: 'validation_split' or 'test_directory'
    EVAL_MODE = 'validation_split'
    # =======================================================
    
    print(f"\n⚙️  Configuration:")
    print(f"   Model Path: {MODEL_PATH}")
    print(f"   Data Directory: {DATA_DIR}")
    print(f"   Test Directory: {TEST_DIR}")
    print(f"   Threshold: {THRESHOLD}")
    print(f"   Device: {DEVICE}")
    print(f"   Evaluation Mode: {EVAL_MODE}")
    
    # Initialize evaluator
    evaluator = FaceNetEvaluator(model_path=MODEL_PATH, device=DEVICE)
    
    # Run evaluation
    if EVAL_MODE == 'validation_split':
        results = evaluator.evaluate_with_validation_split(
            DATA_DIR, 
            test_size=0.2, 
            threshold=THRESHOLD
        )
    elif EVAL_MODE == 'test_directory':
        results = evaluator.evaluate_on_directory(
            threshold=THRESHOLD
        )
    else:
        print(f"❌ Invalid EVAL_MODE: {EVAL_MODE}")
        return
    
    # Print results
    evaluator.print_results()
    
    # Plot visualizations
    if len(results.get('predictions', [])) > 0:
        print(f"\n📊 Generating visualizations...")
        evaluator.plot_confusion_matrix('facenet_confusion_matrix.png')
        evaluator.plot_similarity_distribution('facenet_similarity_distribution.png')
    
    # Save results
    evaluator.save_results('facenet_evaluation_results.json')
    
    print(f"\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()
