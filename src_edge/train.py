import os
import argparse
import joblib
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from features import EdgeFeatureExtractor

def train_models(data_dir, output_dir):
    extractor = EdgeFeatureExtractor()
    X = []
    y = []
    
    classes = ['clean', 'dust', 'bird_droppings', 'moss']
    class_map = {name: i for i, name in enumerate(classes)}
    
    print("🚀 Starting Feature Extraction...")
    for class_name in classes:
        class_path = Path(data_dir) / class_name
        if not class_path.exists():
            print(f"⚠️ Warning: {class_name} folder not found in {data_dir}. Skipping.")
            continue
            
        print(f"📦 Processing class: {class_name}")
        class_images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.jpeg")) + list(class_path.glob("*.png"))
        for i, img_path in enumerate(class_images):
            if i % 20 == 0:
                print(f"  - Progress: {i}/{len(class_images)}")
            features = extractor.extract(img_path)
            if features is not None:
                X.append(features)
                y.append(class_map[class_name])
                
    X = np.array(X)
    y = np.array(y)
    
    if len(X) == 0:
        print("❌ Error: No images found to train on.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 1. Train SVM
    print("🧠 Training SVM (Support Vector Machine)...")
    svm_model = SVC(kernel='linear', probability=True, random_state=42, verbose=True)
    svm_model.fit(X_train, y_train)
    print("✅ SVM Training Complete.")
    y_pred_svm = svm_model.predict(X_test)
    print(f"✅ SVM Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
    
    # 2. Train Random Forest
    print("🌲 Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    print(f"✅ RF Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    
    # Save models
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(svm_model, Path(output_dir) / "solar_svm_model.pkl")
    joblib.dump(rf_model, Path(output_dir) / "solar_rf_model.pkl")
    print(f"💾 Models saved to {output_dir}")

    # Output detailed report for SVM
    print("\n📄 SVM Classification Report:")
    print(classification_report(y_test, y_pred_svm, target_names=[c for c in classes if class_map[c] in np.unique(y)]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/train", help="Directory with class subfolders")
    parser.add_argument("--output_dir", default="backend_edge/app/models", help="Where to save models")
    args = parser.parse_args()
    
    train_models(args.data_dir, args.output_dir)
