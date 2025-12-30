import os
import time
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from features import EdgeFeatureExtractor

def train_models(data_dir, output_dir):
    extractor = EdgeFeatureExtractor()
    
    # Store raw features separately to avoid re-reading images
    X_hog = []
    X_glcm = []
    y = []
    
    classes = ['clean', 'dust', 'bird_droppings', 'moss']
    class_map = {name: i for i, name in enumerate(classes)}
    
    print("🚀 Starting Feature Extraction (HOG & GLCM).")
    
    # 1. Extract all features once
    total_images = 0
    for class_name in classes:
        class_path = Path(data_dir) / class_name
        if not class_path.exists():
            print(f"⚠️ Warning: {class_name} folder not found in {data_dir}. Skipping.")
            continue
            
        print(f"📦 Processing class: {class_name}")
        class_images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.jpeg")) + list(class_path.glob("*.png"))
        total_images += len(class_images)
        
        for i, img_path in enumerate(class_images):
            if i % 50 == 0:
                print(f"  - Progress: {i}/{len(class_images)}")
            
            # Extract both separately
            hog_feat = extractor.extract(img_path, feature_types=['hog'])
            glcm_feat = extractor.extract(img_path, feature_types=['glcm'])
            
            if hog_feat is not None and glcm_feat is not None:
                X_hog.append(hog_feat)
                X_glcm.append(glcm_feat)
                y.append(class_map[class_name])
                
    X_hog = np.array(X_hog)
    X_glcm = np.array(X_glcm)
    y = np.array(y)
    
    if len(y) == 0:
        print("❌ Error: No images found to train on.")
        return

    print(f"\n✅ Extraction Complete. Total samples: {len(y)}")

    # Define the 4 configurations
    configs = [
        {'name': 'solar_svm_hog',  'model_type': 'svm', 'features': 'hog'},
        {'name': 'solar_svm_glcm', 'model_type': 'svm', 'features': 'glcm'},
        {'name': 'solar_rf_hog',   'model_type': 'rf',  'features': 'hog'},
        {'name': 'solar_rf_glcm',  'model_type': 'rf',  'features': 'glcm'},
    ]
    
    results = []
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 2. Train and Evaluate Loop
    for config in configs:
        print(f"\n⚙️  Training Configuration: {config['name']} ({config['model_type'].upper()} + {config['features'].upper()})")
        
        # Select Data
        X = X_hog if config['features'] == 'hog' else X_glcm
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Initialize Model
        if config['model_type'] == 'svm':
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('svc', SVC(kernel='linear', probability=True, random_state=42))
            ])
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate & Measure Time
        start_time = time.time()
        y_pred = model.predict(X_test)
        end_time = time.time()
        
        # Metrics
        avg_proc_time_ms = ((end_time - start_time) / len(X_test)) * 1000
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"   Accuracy: {acc:.4f} | F1: {f1:.4f} | Proc Time: {avg_proc_time_ms:.4f}ms/img")
        
        # Save
        save_path = Path(output_dir) / f"{config['name']}.pkl"
        joblib.dump(model, save_path)
        
        results.append({
            'Model': config['model_type'].upper(),
            'Features': config['features'].upper(),
            'Accuracy': f"{acc:.4f}",
            'Precision': f"{prec:.4f}",
            'Recall': f"{rec:.4f}",
            'F1_Score': f"{f1:.4f}",
            'Proc_Time_ms': f"{avg_proc_time_ms:.4f}ms"
        })

    # 3. Final Report
    print("\n" + "="*95)
    print("📊 COMPARATIVE RESULTS (Raspberry Pi Stability Assessment)")
    print("="*95)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("="*95)
    print(f"💾 All models saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/train", help="Directory with class subfolders")
    parser.add_argument("--output_dir", default="backend_edge/app/models", help="Where to save models")
    args = parser.parse_args()
    
    train_models(args.data_dir, args.output_dir)
