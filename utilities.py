"""
Utility functions for data analysis, visualization, and machine learning.

This module provides a collection of functions for handling missing values,
detecting outliers, visualizing correlations, checking multicollinearity,
and processing numerical features. Additionally, it includes utilities for
model evaluation, hyperparameter tuning, feature engineering,
and result visualization.
"""
from typing import List, Tuple, Dict

from IPython.display import display, Markdown
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.api import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    precision_score,
    recall_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.compose import make_column_transformer


def find_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies and counts missing values in a DataFrame,
    including zeroes, empty strings, and NaN values.

    Args:
        df: The DataFrame to analyze.

    Returns:
        A DataFrame with counts of zeroes, empty strings,
        and NaN values for each column.
    """
    zeroes = (df == 0).sum()
    empty_strings = (df.replace(r"^\s*$", "", regex=True) == "").sum()
    nas = df.isna().sum()
    combined_counts = pd.DataFrame(
        {"Zeroes": zeroes, "Empty Strings": empty_strings, "NaN": nas}
    )
    return combined_counts


def find_outliers(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Detects outliers in multiple features using the IQR method.

    Args:
        df: DataFrame containing the data.
        features: List of features to detect outliers in.

    Returns:
        DataFrame containing the outliers for each feature and a DataFrame
        containing analysis for each feature (outlier count, percentage, IQR bounds,
        and flagged values).
    """
    outlier_reports = []
    outlier_indices = set()
    total_rows = len(df)

    for feature in features:
        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        mask = (df[feature] < lower_bound) | (df[feature] > upper_bound)
        outlier_count = mask.sum()
        outlier_percentage = (outlier_count / total_rows) * 100

        flagged_values = "None"
        if outlier_count > 0:
            flagged_values = (
                f"[{df[feature][mask].min():.2f}, {df[feature][mask].max():.2f}]"
            )
            outlier_indices.update(df[mask].index)

        outlier_reports.append(
            {
                "Feature Name": feature,
                "Outliers": outlier_count,
                "Percentage": f"{outlier_percentage:.2f}%",
                "IQR Bounds": f"[{lower_bound:.2f}, {upper_bound:.2f}]",
                "Flagged Values": flagged_values,
            }
        )

    outlier_reports_df = pd.DataFrame(outlier_reports)
    outlier_reports_df = outlier_reports_df[outlier_reports_df["Outliers"] > 0]

    if not outlier_reports_df.empty:
        display(Markdown("**Feature-wise Outlier Analysis**"))
        display(outlier_reports_df)
        display(Markdown("**All Outliers**"))
        outliers = df.loc[list(outlier_indices), features]
        display(outliers)
    else:
        print("**No features with outliers detected**")


def plot_corr_matrix(df: pd.DataFrame) -> None:
    """
    Plots a heatmap of the correlation matrix for the numerical
    features in the DataFrame.

    Args:
        df: The input DataFrame containing numerical features.

    Returns:
        None
    """
    correlation = df.corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, mask=mask, vmax=1, vmin=-1, cmap="vlag", annot=True)
    plt.title("Correlations heatmap")
    plt.show()


def check_vif(df: pd.DataFrame, features: List[str]) -> None:
    """
    Calculates and prints the Variance Inflation Factor (VIF) for each feature
    in the dataset, excluding the specified features.

    Args:
        df: The input DataFrame containing the dataset.
        features: A list of column names to exclude from the VIF calculation.

    Returns:
        None
    """
    X = df.drop(columns=features)
    X_with_const = add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X_with_const.values, i)
        for i in range(1, X_with_const.shape[1])
    ]
    vif_data["VIF"] = vif_data["VIF"].apply(lambda x: f"{x:.2f}")
    display(vif_data)


def find_binary_numerical_features(df: pd.DataFrame) -> Tuple[list[str], list[str]]:
    """
    Identifies binary and numerical features in the DataFrame.

    Args:
        df: The input DataFrame.

    Returns:
        A tuple containing a list of binary features and a list of numerical features.
    """
    binary_features = [col for col in df.columns if set(df[col].unique()) == {0, 1}]

    numerical_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numerical_features = [
        col for col in numerical_features if col not in binary_features
    ]

    return (binary_features, numerical_features)


def model_performance_comparison(
    scaling_required_models: dict[str, object],
    no_scaling_models: dict[str, object],
    scaler: object,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    cv: StratifiedKFold,
    scorer: str,
) -> None:
    """
    Evaluates models using cross-validation and prints the mean precision-recall AUC score.

    Args:
        scaling_required_models: Dictionary of models that require feature scaling.
        no_scaling_models: Dictionary of models that do not require scaling.
        scaler: Scaler for feature scaling.
        X_train: Training feature data.
        y_train: Training labels.
        cv: StratifiedKFold or number of cross-validation folds.
        scorer: Scoring metric.

    Returns:
        None
    """
    _, numerical_features = find_binary_numerical_features(X_train)
    preprocessor = make_column_transformer((scaler, numerical_features))

    for name, _ in {**scaling_required_models, **no_scaling_models}.items():
        if name in scaling_required_models:
            pipeline = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("model", scaling_required_models[name]),
                ]
            )
        else:
            pipeline = no_scaling_models[name]

        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scorer)
        print(f"{name} Mean PR AUC: {np.mean(scores):.4f}")


def create_poly_features(df: pd.DataFrame, poly: PolynomialFeatures) -> pd.DataFrame:
    """
    Creates polynomial features from a given DataFrame using
    a PolynomialFeatures transformer.

    Args:
        df: The input DataFrame.
        poly: The polynomial feature generator.

    Returns:
        A DataFrame containing the polynomial features.
    """
    poly_features = poly.fit_transform(df)
    poly_feature_names = poly.get_feature_names_out()
    df_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
    return df_poly


def plot_insured_previously_traveled(
    conf_int_travelled: Tuple[float, float],
    conf_int_not_travelled: Tuple[float, float],
    prop_travelled_insured: float,
    prop_not_travelled_insured: float,
) -> None:
    """
    Plots the proportion of insured individuals who have traveled abroad vs. those who have not,
    including 95% confidence intervals.

    Args:
        conf_int_travelled: Confidence interval for those who traveled.
        conf_int_not_travelled: Confidence interval for those who did not travel.
        prop_travelled_insured: Proportion insured among those who traveled.
        prop_not_travelled_insured: Proportion insured among those who did not travel.

    Returns:
        None
    """
    lower_bounds = [conf_int_travelled[0], conf_int_not_travelled[0]]
    upper_bounds = [conf_int_travelled[1], conf_int_not_travelled[1]]

    df_plot = pd.DataFrame(
        {
            "Travel Status": ["Travelled Abroad", "Did Not Travel Abroad"],
            "Proportion Insured": [prop_travelled_insured, prop_not_travelled_insured],
            "Lower Error": [
                prop_travelled_insured - lower_bounds[0],
                prop_not_travelled_insured - lower_bounds[1],
            ],
            "Upper Error": [
                upper_bounds[0] - prop_travelled_insured,
                upper_bounds[1] - prop_not_travelled_insured,
            ],
        }
    )

    ax = sns.barplot(
        data=df_plot,
        x="Travel Status",
        y="Proportion Insured",
        capsize=0.1,
        hue="Travel Status",
    )

    ax.errorbar(
        df_plot["Travel Status"],
        df_plot["Proportion Insured"],
        yerr=[df_plot["Lower Error"], df_plot["Upper Error"]],
        fmt="none",
        c="#0571b0",
        capsize=5,
    )

    plt.ylabel("Proportion Insured")
    plt.title("Proportion of Insured by Travel Status with 95% CIs")
    plt.show()


def model_hyperparameter_tuning(
    scaling_required_models: Dict[str, object],
    no_scaling_models: Dict[str, object],
    scaler: object,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    cv: StratifiedKFold,
    scorer: str,
    param_grids: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    """
    Performs hyperparameter tuning using RandomizedSearchCV on a set of models.

    Args:
        scaling_required_models: Models requiring feature scaling.
        no_scaling_models: Models not requiring scaling.
        scaler: Preprocessing scaler.
        X_train: Training feature set.
        y_train: Training target values.
        cv: StratifiedKFold or number of cross-validation folds.
        scorer: Scoring metric for model evaluation.
        param_grids: Hyperparameter grids for models.

    Returns:
        Dictionary containing the best models after tuning.
    """
    _, numerical_features = find_binary_numerical_features(X_train)
    preprocessor = make_column_transformer((scaler, numerical_features))
    best_models = {}

    for name, _ in {**scaling_required_models, **no_scaling_models}.items():
        if name in scaling_required_models:
            pipeline = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("model", scaling_required_models[name]),
                ]
            )
        else:
            pipeline = Pipeline([("model", no_scaling_models[name])])

        params = {f"model__{k}": v for k, v in param_grids[name].items()}
        randomized_search = RandomizedSearchCV(
            pipeline,
            params,
            cv=cv,
            scoring=scorer,
            n_iter=20,
            n_jobs=-1,
            random_state=42,
        )
        randomized_search.fit(X_train, y_train)

        best_models[name] = randomized_search.best_estimator_
        best_params = {
            k.replace("model__", ""): v
            for k, v in randomized_search.best_params_.items()
        }
        best_score = randomized_search.best_score_

        print(f"Best {name}: {best_params}, PR AUC: {best_score:.4f}")
    return best_models


def plot_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, title: str) -> None:
    """
    Plots a confusion matrix.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        title: Title for the plot.

    Returns:
        None
    """
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.show()


def plot_feature_importances(
    feature_names: List, feature_importances: np.ndarray
) -> None:
    """
    Plots feature importances for a Random Forest model.

    Args:
        feature_names: List of feature names.
        feature_importances: Feature importance values.

    Returns:
        None
    """
    sorted_idx = np.argsort(feature_importances)[::-1]
    sns.barplot(
        x=feature_importances[sorted_idx], y=np.array(feature_names)[sorted_idx]
    )
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance for Random Forest")
    plt.show()


def make_dummy_classifier(
    strategy: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Trains and evaluates a DummyClassifier.

    Args:
        strategy: Dummy classifier strategy.
        X_train: Training features.
        y_train: Training labels.
        X_test): Test features.
        y_test: Test labels.

    Returns:
        None
    """
    dummy = DummyClassifier(strategy=strategy, random_state=42)
    dummy.fit(X_train, y_train)
    y_pred = dummy.predict(X_test)
    y_probs = dummy.predict_proba(X_test)[:, 1]
    print(
        f"\n{strategy} Precision:",
        round(precision_score(y_test, y_pred, zero_division=0), 4),
    )
    print(
        f"{strategy} Recall:", round(recall_score(y_test, y_pred, zero_division=0), 4)
    )
    print(f"{strategy} PR AUC:", round(average_precision_score(y_test, y_probs), 4))


def plot_income_insurance_status(
    mean_insured: float,
    mean_not_insured: float,
    ci_insured: Tuple[float, float],
    ci_not_insured: Tuple[float, float],
) -> None:
    """
    Plots mean annual income by insurance status with 95% confidence intervals.

    Args:
        mean_insured: Mean income of insured individuals.
        mean_not_insured: Mean income of non-insured individuals.
        ci_insured: Confidence interval for insured.
        ci_not_insured: Confidence interval for non-insured.

    Returns:
        None
    """
    lower_bounds = [mean_insured - ci_insured[0], mean_not_insured - ci_not_insured[0]]
    upper_bounds = [ci_insured[1] - mean_insured, ci_not_insured[1] - mean_not_insured]

    means = [mean_insured, mean_not_insured]
    insurance_status = ["Insured", "Not Insured"]

    ax = sns.barplot(x=insurance_status, y=means, hue=insurance_status)

    ax.errorbar(
        insurance_status,
        means,
        yerr=[lower_bounds, upper_bounds],
        fmt="none",
        c="#0571b0",
        capsize=5,
    )

    plt.ylabel("Mean Annual Income")
    plt.title("Mean Annual Income by Insurance Status with 95% CIs")
    plt.show()


def plot_original_duplicate_distributions(
    df: pd.DataFrame, duplicates: pd.DataFrame
) -> None:
    """
    Plots the distributions of features in the original dataset and its duplicate subset.

    Args:
        df: The full dataset.
        duplicates: The subset containing duplicate rows.

    Returns:
        None
    """
    rows = len(df.columns)
    fig, axes = plt.subplots(rows, 2, figsize=(10, rows * 4))

    for i, column in enumerate(df.columns):
        sns.histplot(
            data=df,
            x=column,
            hue="TravelInsurance",
            edgecolor="black",
            bins=15,
            ax=axes[i, 0],
        )
        sns.histplot(
            data=duplicates,
            x=column,
            hue="TravelInsurance",
            edgecolor="black",
            bins=15,
            ax=axes[i, 1],
        )

    fig.suptitle(
        "Feature distributions of full dataset (left) vs. only duplicates (right)",
        y=1,
    )
    plt.tight_layout()
    plt.show()
