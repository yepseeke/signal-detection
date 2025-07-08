import os
import json
import seaborn as sns
import numpy as np

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid

from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix


def load_catboost_data(npz_path: str):
    data = np.load(npz_path)
    X = data['features']
    y = data['labels']

    feature_names = data["feature_names"].tolist()
    class_mapping = json.loads(str(data["class_mapping"]))

    return X, y, feature_names, class_mapping


def shuffle_data(X, y, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    return X[indices], y[indices]


def run_catboost_hyperparam_tuning(
        train_path: str,
        valid_path: str,
        param_grid: dict,
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose=100,
        random_seed: int = 42,
        save_best_model_path: str = None
):
    X_train, y_train, feature_names, class_mapping = load_catboost_data(train_path)
    X_train, y_train = shuffle_data(X_train, y_train, seed=random_seed)
    train_pool = Pool(X_train, y_train)

    X_valid, y_valid, _, _ = load_catboost_data(valid_path)
    valid_pool = Pool(X_valid, y_valid)

    best_score = 0
    best_model = None
    best_params = None

    for params in ParameterGrid(param_grid):
        params = params.copy()
        params['random_seed'] = random_seed
        params['task_type'] = 'GPU'  # Включаем GPU
        params.setdefault('loss_function', 'MultiClass')
        params.setdefault('eval_metric', 'Accuracy')

        print(f"Training with params: {params}")

        model = CatBoostClassifier(iterations=num_boost_round, **params)
        model.fit(
            train_pool,
            eval_set=valid_pool,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose
        )

        preds = model.predict(valid_pool)
        acc = accuracy_score(valid_pool.get_label(), preds)
        f1 = f1_score(valid_pool.get_label(), preds, average='weighted')
        print(f"Validation accuracy: {acc:.4f}, weighted F1: {f1:.4f}")

        # Подбор по accuracy (можно поменять на f1, если хочешь)
        if acc > best_score:
            best_score = acc
            best_model = model
            best_params = params
            if save_best_model_path:
                os.makedirs(os.path.dirname(save_best_model_path), exist_ok=True)
                best_model.save_model(save_best_model_path)
                print(f"Best model saved to {save_best_model_path}")

    print(f"\nBest validation accuracy: {best_score:.4f}")
    print(f"Best params: {best_params}")

    return best_model, best_params


def plot_feature_importance(model, feature_names=None, max_num_features=20):
    importances = model.get_feature_importance()

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    sorted_idx = importances.argsort()[::-1][:max_num_features]
    sorted_importances = importances[sorted_idx]
    sorted_features = [feature_names[i] for i in sorted_idx]

    for f, imp in zip(sorted_features, sorted_importances):
        print(f"{f}: {imp:.4f}")

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features[::-1], sorted_importances[::-1])
    plt.xlabel("Feature Importance")
    plt.title("Top Feature Importances")
    plt.show()


def apply_model_to_test(test_path: str, model_path: str):
    X_test, y_test, feature_names, class_mapping = load_catboost_data(test_path)
    test_pool = Pool(X_test, y_test)

    model = CatBoostClassifier()
    model.load_model(model_path)

    preds = model.predict(test_pool)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')

    print(f"Test accuracy: {acc:.4f}")
    print(f"Test weighted F1 score: {f1:.4f}")

    return preds


def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False, cmap='Blues'):
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if class_names is None:
        num_classes = cm.shape[0]
        class_names = [str(i) for i in range(num_classes)]
    elif isinstance(class_names, dict):
        class_names = [class_names[str(i)] for i in range(len(class_names))]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    scalar_features = r'D:\Projects\Python\drone-detection-c\dataset\scalar_features'
    train_path = os.path.join(scalar_features, 'train.npz')
    valid_path = os.path.join(scalar_features, 'valid.npz')
    test_path = os.path.join(scalar_features, 'test.npz')

    model_path = r'D:\Projects\github\signal-detection\runs\catboost\catboost_model_hyper.cbm'

    # model = CatBoostClassifier()
    # model.load_model(model_path)
    #
    # X_test, y_test, feature_names, class_mapping = load_catboost_data(test_path)
    # preds = apply_model_to_test(test_path, model_path)
    # print(len(preds), len(y_test))
    # print(class_mapping)
    # plot_confusion_matrix(y_test, preds, class_mapping)
    # plot_feature_importance(model, max_num_features = 92)

    param_grid = {
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5],
        'border_count': [32, 64, 128],
    }

    # apply_model_to_test(test_path, )

    best_model, best_params = run_catboost_hyperparam_tuning(
        train_path=train_path,
        valid_path=valid_path,
        param_grid=param_grid,
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose=100,
        random_seed=42,
        save_best_model_path=r'D:\Projects\github\signal-detection\runs\catboost\catboost_model_hyper.cbm'
    )

