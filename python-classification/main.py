import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

def cap_outliers_iqr(df_column):

    q1 = df_column.quantile(0.25)
    q3 = df_column.quantile(0.75)
    IQR = q3 - q1
    lower_bound = q1 - 1.5*IQR
    higher_bound = q3+ 1.5*IQR

    capped_column = np.where(df_column < lower_bound, lower_bound, df_column)
    print(capped_column)
    capped_column = np.where(capped_column > higher_bound, higher_bound, capped_column)
    return pd.Series(capped_column, index = df_column.index)

def evaluation_model(model_name,model,y_test, X_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(f'Model : {model_name}')
    print(cm)
    print(f'Accuracy Score: {accuracy_score(y_test, y_pred)}')
    print(f'Precision Score: {precision_score(y_test, y_pred)}')
    print(f'Recall Score: {recall_score(y_test,y_pred)}')
    print(f'F1 Score: {f1_score(y_test, y_pred)}')
    print(classification_report(y_test,y_pred,target_names=['NO CHD (0)', 'CHF (1)']))
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability of the positive class (1)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC AUC Score: {roc_auc:.4f}")
    else:
        print("ROC AUC Score: Not available (model does not provide probabilities directly).")
    print(f"{'='*50}\n")


def main():
    try :
        df = pd.read_csv('framingham.csv', decimal=',', sep=',', header =0)
    except FileNotFoundError:
        print("Error: 'framingham.csv' not found. Please try again!")
        return

    df.dropna(inplace=True)
    print(df['TenYearCHD'].value_counts(normalize=True))
    df['sysBP'] = pd.to_numeric(df['sysBP'])


    features = ['male', 'age', 'cigsPerDay' , 'totChol', 'sysBP', 'glucose' ]
    y = df['TenYearCHD']
    X_processed = df[features].copy()
    for col in features:
            X_processed[col] = cap_outliers_iqr(X_processed[col])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)
    df_X_scaled = pd.DataFrame(X_scaled, columns =X_processed.columns)
    X_trained, X_test, y_trained, y_test = train_test_split(df_X_scaled, y, test_size = 0.2, train_size=0.8, random_state = 42)
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Support Vector Machine': SVC(random_state=42, probability=True)
    }
    for model_name, model in models.items():
        model.fit(X_trained, y_trained)

        evaluation_model(model_name, model, y_test,X_test)


if __name__ == "__main__":
    main()
