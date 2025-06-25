# train_model.py

print("🚀 Script started...")

try:
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score
    import joblib
    import os

    # Step 1: Load the dataset
    print("📂 Loading dataset...")
    df = pd.read_excel("Liver_Cirrhosis_Dataset.xlsx")  # Ensure this file is in the same folder

    # Step 2: Encode Gender
    print("🔠 Encoding Gender...")
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])  # Male = 1, Female = 0

    # Step 3: Split features and target
    print("🧪 Splitting features and target...")
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Step 4: Normalize data
    print("📊 Normalizing data...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 5: Split into train/test
    print("🧩 Splitting data into training and testing...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Step 6: Train models
    print("🤖 Training models...")
    rf = RandomForestClassifier(random_state=42)
    knn = KNeighborsClassifier()
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    rf.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    # Step 7: Evaluate models
    print("📈 Evaluating models...")
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    knn_acc = accuracy_score(y_test, knn.predict(X_test))
    xgb_acc = accuracy_score(y_test, xgb.predict(X_test))

    print(f"✅ Random Forest Accuracy: {rf_acc:.2f}")
    print(f"✅ KNN Accuracy: {knn_acc:.2f}")
    print(f"✅ XGBoost Accuracy: {xgb_acc:.2f}")

    # Step 8: Choose best model
    print("🏆 Choosing best model...")
    best_model = rf
    best_name = "Random Forest"
    best_acc = rf_acc

    if knn_acc > best_acc:
        best_model = knn
        best_name = "KNN"
        best_acc = knn_acc

    if xgb_acc > best_acc:
        best_model = xgb
        best_name = "XGBoost"
        best_acc = xgb_acc

    print(f"🥇 Best Model: {best_name} (Accuracy: {best_acc:.2f})")

    # Step 9: Save best model and scaler
    print("💾 Saving model and normalizer...")
    os.makedirs("model", exist_ok=True)
    joblib.dump(best_model, "model/best_model.pkl")
    joblib.dump(scaler, "model/normalizer.pkl")

    print("🎉 All done! Model and scaler saved in /model folder.")

except Exception as e:
    print("❌ An error occurred while running the script:")
    print(e)
