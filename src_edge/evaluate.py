import joblib
import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from features import EdgeFeatureExtractor

def evaluate(data_dir, model_dir):
    extractor = EdgeFeatureExtractor()
    classes = ['clean', 'dust', 'bird_droppings', 'moss']
    class_map = {name: i for i, name in enumerate(classes)}
    
    X_test = []
    y_test = []
    
    print("🚀 Extracting features for evaluation...")
    for class_name in classes:
        class_path = Path(data_dir) / class_name
        if not class_path.exists(): continue
        
        for img_path in class_path.glob("*.jpg"):
            features = extractor.extract(img_path)
            if features is not None:
                X_test.append(features)
                y_test.append(class_map[class_name])
                
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    models = {
        "SVM": Path(model_dir) / "solar_svm_model.pkl",
        "Random Forest": Path(model_dir) / "solar_rf_model.pkl"
    }
    
    print("\n" + "="*40)
    print("📊 MODEL PERFORMANCE COMPARISON")
    print("="*40)
    
    for name, path in models.items():
        if not path.exists():
            print(f"❌ {name} model not found at {path}")
            continue
            
        model = joblib.load(path)
        y_pred = model.predict(X_test)
        
        print(f"\n🔹 {name}:")
        print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
        print(f"  Precision: {precision_score(y_test, y_pred, average='macro'):.4f}")
        print(f"  Recall:    {recall_score(y_test, y_pred, average='macro'):.4f}")
        print(f"  F1 Score:  {f1_score(y_test, y_pred, average='macro'):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/test", help="Directory with class subfolders")
    parser.add_argument("--model_dir", default="backend_edge/app/models", help="Directory where models are saved")
    args = parser.parse_args()
    
    evaluate(args.data_dir, args.model_dir)
