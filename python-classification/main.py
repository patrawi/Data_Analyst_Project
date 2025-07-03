import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

def evaluation_model(model_name, model,y_test,X_test):
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

def logistic_regression_model(df_X_scaled, y):
    X_trained, X_test, y_trained, y_test = train_test_split(df_X_scaled, y, test_size = 0.2, train_size=0.8, random_state = 42)

    lr = LogisticRegression(random_state = 42, max_iter= 1000)
    lr.fit(X_trained, y_trained)

    evaluation_model('Logistic Regression',lr,y_test, X_test)
    return

def decision_tree_model(df_X_scaled, y):
    X_trained, X_test, y_trained, y_test = train_test_split(df_X_scaled, y, test_size = 0.2, train_size=0.8, random_state = 42)

    clf = DecisionTreeClassifier(random_state=42)


    clf.fit(X_trained,y_trained)

    evaluation_model('Decision Tree',clf, y_test, X_test)
    return

def random_forest_model(df_X_scaled, y):

    X_trained, X_test, y_trained, y_test = train_test_split(df_X_scaled, y, test_size = 0.2, train_size=0.8, random_state = 42)

    rfc = RandomForestClassifier(random_state = 42)
    rfc.fit(X_trained, y_trained)

    evaluation_model('Random Forest',rfc, y_test, X_test)
    return

def knn_model(df_X_scaled, y):
    X_trained, X_test, y_trained, y_test = train_test_split(df_X_scaled, y, test_size = 0.2, train_size=0.8, random_state = 42)

    knn = KNeighborsClassifier()
    knn.fit(X_trained,y_trained)

    evaluation_model('K-nearest neighbourhood', knn ,y_test, X_test)
    return

def svc_model(df_X_scaled, y):
    X_trained, X_test,  y_trained, y_test = train_test_split(df_X_scaled, y, test_size = 0.2, train_size=0.8, random_state = 42)
    svc = SVC(random_state = 42, probability=True)
    svc.fit(X_trained, y_trained)

    evaluation_model('Support Vector Machines', svc,y_test, X_test)
    return




def main():
    df = pd.read_csv('framingham.csv', decimal=',', sep=',', header =0)
    df.dropna(inplace=True)
    print(df['TenYearCHD'].value_counts(normalize=True))
    y = df['TenYearCHD']
    X = df[['male', 'age', 'cigsPerDay' , 'totChol', 'sysBP', 'glucose' ]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    df_X_scaled = pd.DataFrame(X_scaled, columns =X.columns)

    logistic_regression_model(df_X_scaled, y)
    decision_tree_model(df_X_scaled,y)
    random_forest_model(df_X_scaled,y)
    knn_model(df_X_scaled, y)
    svc_model(df_X_scaled,y)


if __name__ == "__main__":
    main()
