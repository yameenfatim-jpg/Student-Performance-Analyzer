
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
import io
import warnings


# Load custom CSS
from pathlib import Path

CSS_FILE = Path(__file__).parent / "style.css"

st.set_page_config(
    page_title="Student Performance Analyzer",
    page_icon="🎓",
    layout="wide"
)

if CSS_FILE.exists():
    st.markdown(
        f"<style>{CSS_FILE.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True
    )


from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    cross_validate
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    LabelEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")




st.markdown("""
<div style="
    background: linear-gradient(120deg, #ffe3ef, #f0eaff);
    border: 1px solid #f1c3dc;
    border-radius: 20px;
    padding: 26px;
    margin-bottom: 18px;
">
    <div style="font-size: 15px; color: #bb4c83;">
        ✿ MY COLLEGE PROJECT · DATA + MACHINE LEARNING
    </div>
    <h1 style="margin: 10px 0; color: #344563;">
        🎓 Student<span
        <span style="color: #d94d8b;"> Performance Analyzer</span>
    </h1>
    <p style="font-size: 17px; color: #52617d; margin-bottom: 0;">
        Explore student data, discover interesting patterns,
        and experiment with machine-learning models.
    </p>
</div>
""", unsafe_allow_html=True)

st.info(
    "This app adapts to the uploaded dataset. Available analyses and "
    "prediction targets depend on its columns and data."
)


# --------------------------------------------------
# 1. DATASET UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is None:
    st.markdown("### Getting started")
    st.write("1. Upload a CSV or Excel dataset.")
    st.write("2. Explore its columns, missing values, and patterns.")
    st.write("3. Select a target column for machine-learning experiments.")
    st.write("4. Compare models using test-set metrics and cross-validation.")
    st.stop()


# --------------------------------------------------
# 2. READ THE DATASET
# --------------------------------------------------

def load_dataset(file):
    filename = file.name.lower()

    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)

    raw = file.getvalue()

    # Try automatic delimiter detection first.
    try:
        return pd.read_csv(
            io.BytesIO(raw),
            sep=None,
            engine="python"
        )
    except Exception:
        pass

    # Fallback for common CSV separators.
    for separator in [",", ";", "\t", "|"]:
        try:
            data = pd.read_csv(
                io.BytesIO(raw),
                sep=separator
            )
            if data.shape[1] > 1:
                return data
        except Exception:
            continue

    raise ValueError(
        "The file could not be read. Check its format and delimiter."
    )


try:
    df = load_dataset(uploaded_file)
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()


if df.empty:
    st.error("The uploaded dataset contains no rows.")
    st.stop()

# Clean column names and remove completely empty rows/columns.
df.columns = [
    str(column).strip()
    for column in df.columns
]
df = df.dropna(how="all").dropna(axis=1, how="all")

if df.empty or len(df.columns) == 0:
    st.error("No usable data was found in this file.")
    st.stop()

# Avoid ambiguous column names.
if df.columns.duplicated().any():
    st.warning(
        "Duplicate column names were found. They will be renamed "
        "so the app can process them safely."
    )
    counts = {}
    new_columns = []

    for column in df.columns:
        counts[column] = counts.get(column, 0) + 1
        if counts[column] == 1:
            new_columns.append(column)
        else:
            new_columns.append(f"{column}_{counts[column]}")

    df.columns = new_columns


st.success("Dataset uploaded successfully!")

st.caption(f"File: {uploaded_file.name}")

# Keep the original uploaded data available for analysis.
original_df = df.copy()

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = df.select_dtypes(
    include=["object", "category", "bool", "string"]
).columns.tolist()

# --------------------------------------------------
# 3. OVERVIEW
# --------------------------------------------------

overview_tab, quality_tab, charts_tab, ml_tab, download_tab = st.tabs([
    "📊 Overview",
    "🧹 Data Quality",
    "📈 Visualizations",
    "🤖 ML Training & Comparison",
    "⬇️ Downloads"
])

with overview_tab:
    st.subheader("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", f"{df.shape[1]:,}")
    col3.metric("Numeric columns", len(numeric_columns))
    col4.metric("Categorical columns", len(categorical_columns))

    st.markdown("#### Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("#### Column information")

    column_info = pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(df[c].dtype) for c in df.columns],
        "Missing values": [int(df[c].isna().sum()) for c in df.columns],
        "Missing (%)": [
            round(df[c].isna().mean() * 100, 2)
            for c in df.columns
        ],
        "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns]
    })

    st.dataframe(column_info, use_container_width=True)

    st.markdown("#### Statistical summary")
    st.dataframe(
        df.describe(include="all").T,
        use_container_width=True
    )


# --------------------------------------------------
# 4. DATA QUALITY
# --------------------------------------------------

with quality_tab:
    st.subheader("Data Quality Analysis")

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    st.markdown("#### Missing values")

    if missing.empty:
        st.success("No missing values were detected.")
    else:
        missing_df = pd.DataFrame({
            "Column": missing.index,
            "Missing count": missing.values,
            "Missing (%)": [
                round(df[column].isna().mean() * 100, 2)
                for column in missing.index
            ]
        })
        st.dataframe(missing_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        missing_df.plot(
            x="Column",
            y="Missing count",
            kind="bar",
            ax=ax,
            legend=False
        )
        ax.set_title("Missing Values by Column")
        ax.set_ylabel("Missing count")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("#### Duplicate rows")
    duplicate_count = int(df.duplicated().sum())
    st.metric("Duplicate rows", duplicate_count)

    if duplicate_count:
        st.dataframe(
            df[df.duplicated(keep=False)].head(50),
            use_container_width=True
        )

    st.markdown("#### Constant columns")
    constant_columns = [
        c for c in df.columns
        if df[c].nunique(dropna=False) <= 1
    ]

    if constant_columns:
        st.write(constant_columns)
    else:
        st.write("No constant columns detected.")

    st.markdown("#### Possible identifier columns")
    possible_ids = []

    for column in df.columns:
        unique_count = df[column].nunique(dropna=True)
        unique_ratio = unique_count / max(len(df), 1)
        name = str(column).lower()

        id_name = (
            name == "id"
            or name.endswith("_id")
            or name.startswith("id_")
            or name in ["student number", "student_id", "student id", "roll number"]
        )

        # Only flag high-cardinality columns with an ID-like name,
        # or columns that look almost entirely unique.
        if id_name or (
            len(df) >= 20
            and unique_ratio >= 0.98
            and unique_count > 10
        ):
            possible_ids.append(column)

    if possible_ids:
        st.write(possible_ids)
        st.caption(
            "These are suggestions, not confirmed identifiers. "
            "Review them before excluding any column from a model."
        )
    else:
        st.write("No obvious identifier columns detected.")

    st.markdown("#### Potential numeric outliers")

    if numeric_columns:
        outlier_rows = []

        for column in numeric_columns:
            values = df[column].dropna()

            if len(values) < 4:
                continue

            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                count = 0
            else:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                count = int(((values < lower) | (values > upper)).sum())

            outlier_rows.append({
                "Column": column,
                "Potential outliers": count
            })

        if outlier_rows:
            outlier_df = pd.DataFrame(outlier_rows)
            st.dataframe(outlier_df, use_container_width=True)
            st.caption(
                "The IQR rule identifies unusual values, not necessarily errors."
            )
    else:
        st.write("No numeric columns are available for this analysis.")

# --------------------------------------------------
# 5. VISUALIZATIONS
# --------------------------------------------------

with charts_tab:
    st.subheader("Explore Relationships in Your Dataset")

    if numeric_columns:
        selected_numeric = st.selectbox(
            "Choose a numeric column for its distribution",
            numeric_columns,
            key="distribution_column"
        )

        values = df[selected_numeric].dropna()

        if not values.empty:
            fig, ax = plt.subplots(figsize=(9, 4))
            sns.histplot(values, kde=len(values) > 1, ax=ax)
            ax.set_title(f"Distribution of {selected_numeric}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("#### Compare two numeric columns")

        if len(numeric_columns) >= 2:
            x_column = st.selectbox(
                "X-axis",
                numeric_columns,
                index=0,
                key="scatter_x"
            )

            y_options = [
                column for column in numeric_columns
                if column != x_column
            ]

            y_column = st.selectbox(
                "Y-axis",
                y_options,
                key="scatter_y"
            )

            plot_data = df[[x_column, y_column]].dropna()

            if len(plot_data) > 0:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.scatterplot(
                    data=plot_data,
                    x=x_column,
                    y=y_column,
                    ax=ax
                )
                ax.set_title(f"{y_column} vs {x_column}")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        st.markdown("#### Correlation heatmap")

        if len(numeric_columns) >= 2:
            corr = df[numeric_columns].corr()

            fig, ax = plt.subplots(
                figsize=(max(8, len(numeric_columns) * 0.7),
                         max(5, len(numeric_columns) * 0.55))
            )
            sns.heatmap(
                corr,
                cmap="coolwarm",
                center=0,
                annot=len(numeric_columns) <= 10,
                fmt=".2f",
                ax=ax
            )
            ax.set_title("Numeric Feature Correlations")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("At least two numeric columns are needed for correlation.")

    else:
        st.info("This dataset has no numeric columns.")

    st.markdown("#### Categorical column counts")

    if categorical_columns:
        category_column = st.selectbox(
            "Choose a categorical column",
            categorical_columns,
            key="category_column"
        )

        counts = (
            df[category_column]
            .fillna("Missing")
            .astype(str)
            .value_counts()
            .head(20)
        )

        fig, ax = plt.subplots(figsize=(9, 4))
        counts.plot(kind="bar", ax=ax)
        ax.set_title(f"Counts for {category_column}")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    else:
        st.info("No categorical columns were detected.")

    st.markdown("#### Compare group averages")

    if categorical_columns and numeric_columns:
        group_column = st.selectbox(
            "Group by",
            categorical_columns,
            key="group_column"
        )

        value_column = st.selectbox(
            "Calculate average of",
            numeric_columns,
            key="group_numeric"
        )

        grouped = (
            df.groupby(group_column, dropna=False)[value_column]
            .mean()
            .sort_values(ascending=False)
        )

        st.dataframe(
            grouped.rename("Average").reset_index(),
            use_container_width=True
        )

        fig, ax = plt.subplots(figsize=(9, 4))
        grouped.head(20).plot(kind="bar", ax=ax)
        ax.set_title(f"Average {value_column} by {group_column}")
        ax.set_ylabel(f"Average {value_column}")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# --------------------------------------------------
# 6. MACHINE-LEARNING TRAINING
# --------------------------------------------------

with ml_tab:
    st.subheader("Machine-Learning Model Training")

    st.write(
        "Choose a column to predict. The app selects regression for "
        "numeric targets and classification for categorical targets."
    )

    target_column = st.selectbox(
        "Select target column",
        df.columns.tolist(),
        key="target_column"
    )

    target = df[target_column]
    usable_rows = target.notna()
    model_df = df.loc[usable_rows].copy()

    # Remove rows where the selected target is missing.
    model_df = model_df.dropna(subset=[target_column])

    # Remove constant columns and the target itself from features.
    feature_columns = [
        column for column in model_df.columns
        if column != target_column
        and model_df[column].nunique(dropna=True) > 1
    ]

    if not feature_columns:
        st.warning("No usable feature columns remain for this target.")
    else:
        X = model_df[feature_columns].copy()
        y = model_df[target_column].copy()

        # High-cardinality columns often represent identifiers.
        # Allow the user to decide whether to exclude them.
        high_cardinality = [
            column for column in X.columns
            if X[column].nunique(dropna=True) > 50
            and X[column].nunique(dropna=True) / max(len(X), 1) > 0.8
        ]

        if high_cardinality:
            st.warning(
                "Some columns contain many unique values and may be IDs: "
                + ", ".join(high_cardinality)
            )
            excluded_ids = st.multiselect(
                "Optional: exclude identifier-like columns",
                high_cardinality,
                default=high_cardinality
            )
            X = X.drop(columns=excluded_ids, errors="ignore")

        if X.shape[1] == 0:
            st.warning("No features remain after exclusions.")
        else:
            # Determine whether the target should be treated as
            # classification or regression.
            target_is_numeric = pd.api.types.is_numeric_dtype(y)

            if target_is_numeric:
                unique_target_values = y.nunique()
                classification_choice = (
                    unique_target_values <= 10
                    and unique_target_values < max(20, len(y) * 0.05)
                )
            else:
                classification_choice = True

            task = st.radio(
                "Prediction task",
                ["Automatic", "Regression", "Classification"],
                index=0,
                horizontal=True,
                key="task_choice"
            )

            if task == "Regression":
                is_classification = False
            elif task == "Classification":
                is_classification = True
            else:
                is_classification = classification_choice

            st.write(
                f"**Selected task:** "
                f"{'Classification' if is_classification else 'Regression'}"
            )

            if is_classification:
                class_counts = y.astype(str).value_counts()
                st.write("Target class counts:")
                st.dataframe(
                    class_counts.rename("Count").reset_index(),
                    use_container_width=True
                )

                if class_counts.size < 2:
                    st.error(
                        "Classification requires at least two target classes."
                    )
                    st.stop()

                # Encode target labels for sklearn.
                label_encoder = LabelEncoder()
                y_model = label_encoder.fit_transform(y.astype(str))
                class_count = len(label_encoder.classes_)

                st.caption(
                    f"Detected {class_count} target classes."
                )

            else:
                y_model = pd.to_numeric(y, errors="coerce")
                valid_target = y_model.notna()
                X = X.loc[valid_target]
                y_model = y_model.loc[valid_target]

            if len(X) < 10:
                st.warning(
                    "At least 10 usable rows are recommended for model "
                    "comparison. Results from very small datasets may be "
                    "unstable."
                )

            if len(X) >= 5 and X.shape[1] > 0:
                test_size = st.slider(
                    "Test-set proportion",
                    min_value=0.1,
                    max_value=0.4,
                    value=0.2,
                    step=0.05
                )

                # Create preprocessing based on each feature's type.
                numeric_features = X.select_dtypes(
                    include=np.number
                ).columns.tolist()

                categorical_features = [
                    column for column in X.columns
                    if column not in numeric_features
                ]

                numeric_transformer = Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ])

                categorical_transformer = Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(
                        handle_unknown="ignore"
                    ))
                ])

                transformers = []

                if numeric_features:
                    transformers.append((
                        "numeric",
                        numeric_transformer,
                        numeric_features
                    ))

                if categorical_features:
                    transformers.append((
                        "categorical",
                        categorical_transformer,
                        categorical_features
                    ))

                preprocessor = ColumnTransformer(
                    transformers=transformers,
                    remainder="drop"
                )

                # Hold out a test set for final evaluation.
                stratify_values = None

                if is_classification:
                    counts = pd.Series(y_model).value_counts()
                    if counts.min() >= 2:
                        stratify_values = y_model

                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y_model,
                        test_size=test_size,
                        random_state=42,
                        stratify=stratify_values
                    )
                except ValueError as error:
                    st.error(f"Could not split the dataset: {error}")
                    st.stop()

                if is_classification:
                    model_definitions = {
                        "Dummy Baseline": DummyClassifier(
                            strategy="most_frequent"
                        ),
                        "Logistic Regression": LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced"
                        ),
                        "Random Forest": RandomForestClassifier(
                            n_estimators=200,
                            random_state=42,
                            class_weight="balanced"
                        ),
                        "Extra Trees": ExtraTreesClassifier(
                            n_estimators=200,
                            random_state=42,
                            class_weight="balanced"
                        )
                    }
                else:
                    model_definitions = {
                        "Dummy Baseline": DummyRegressor(
                            strategy="mean"
                        ),
                        "Linear Regression": LinearRegression(),
                        "Random Forest": RandomForestRegressor(
                            n_estimators=200,
                            random_state=42
                        ),
                        "Extra Trees": ExtraTreesRegressor(
                            n_estimators=200,
                            random_state=42
                        )
                    }

                if st.button(
                    "🚀 Train and compare models",
                    type="primary"
                ):
                    test_results = []
                    cv_results = []
                    trained_models = {}
                    predictions = {}

                    with st.spinner(
                        "Training models and running cross-validation..."
                    ):
                        for model_name, estimator in model_definitions.items():
                            pipeline = Pipeline(steps=[
                                ("preprocessor", preprocessor),
                                ("model", estimator)
                            ])

                            try:
                                pipeline.fit(X_train, y_train)
                                y_pred = pipeline.predict(X_test)

                                trained_models[model_name] = pipeline
                                predictions[model_name] = y_pred

                                if is_classification:
                                    test_results.append({
                                        "Model": model_name,
                                        "Accuracy": accuracy_score(
                                            y_test, y_pred
                                        ),
                                        "Precision (macro)": precision_score(
                                            y_test,
                                            y_pred,
                                            average="macro",
                                            zero_division=0
                                        ),
                                        "Recall (macro)": recall_score(
                                            y_test,
                                            y_pred,
                                            average="macro",
                                            zero_division=0
                                        ),
                                        "F1 (macro)": f1_score(
                                            y_test,
                                            y_pred,
                                            average="macro",
                                            zero_division=0
                                        )
                                    })
                                else:
                                    test_results.append({
                                        "Model": model_name,
                                        "MAE": mean_absolute_error(
                                            y_test, y_pred
                                        ),
                                        "RMSE": np.sqrt(
                                            mean_squared_error(
                                                y_test, y_pred
                                            )
                                        ),
                                        "R²": r2_score(y_test, y_pred)
                                    })

                                # Cross-validation is performed on the
                                # training split only. The held-out test
                                # set is not used during model selection.
                                if is_classification:
                                    train_counts = pd.Series(
                                        y_train
                                    ).value_counts()

                                    folds = min(
                                        5,
                                        int(train_counts.min())
                                    )

                                    if folds >= 2:
                                        cv = StratifiedKFold(
                                            n_splits=folds,
                                            shuffle=True,
                                            random_state=42
                                        )

                                        scores = cross_validate(
                                            pipeline,
                                            X_train,
                                            y_train,
                                            cv=cv,
                                            scoring={
                                                "accuracy": "accuracy",
                                                "f1_macro": "f1_macro"
                                            },
                                            error_score="raise"
                                        )

                                        cv_results.append({
                                            "Model": model_name,
                                            "CV Accuracy Mean": scores[
                                                "test_accuracy"
                                            ].mean(),
                                            "CV Accuracy Std": scores[
                                                "test_accuracy"
                                            ].std(),
                                            "CV F1 Macro Mean": scores[
                                                "test_f1_macro"
                                            ].mean()
                                        })
                                    else:
                                        cv_results.append({
                                            "Model": model_name,
                                            "CV Accuracy Mean": np.nan,
                                            "CV Accuracy Std": np.nan,
                                            "CV F1 Macro Mean": np.nan
                                        })
                                else:
                                    folds = min(5, len(X_train))

                                    if folds >= 2:
                                        cv = KFold(
                                            n_splits=folds,
                                            shuffle=True,
                                            random_state=42
                                        )

                                        scores = cross_validate(
                                            pipeline,
                                            X_train,
                                            y_train,
                                            cv=cv,
                                            scoring={
                                                "mae": "neg_mean_absolute_error",
                                                "r2": "r2"
                                            },
                                            error_score="raise"
                                        )

                                        cv_results.append({
                                            "Model": model_name,
                                            "CV MAE Mean": -scores[
                                                "test_mae"
                                            ].mean(),
                                            "CV MAE Std": scores[
                                                "test_mae"
                                            ].std(),
                                            "CV R² Mean": scores[
                                                "test_r2"
                                            ].mean()
                                        })
                                    else:
                                        cv_results.append({
                                            "Model": model_name,
                                            "CV MAE Mean": np.nan,
                                            "CV MAE Std": np.nan,
                                            "CV R² Mean": np.nan
                                        })

                            except Exception as error:
                                st.warning(
                                    f"{model_name} could not be evaluated: "
                                    f"{error}"
                                )

                    if test_results:
                        test_results_df = pd.DataFrame(test_results)
                        cv_results_df = pd.DataFrame(cv_results)

                        st.session_state["test_results_df"] = test_results_df
                        st.session_state["cv_results_df"] = cv_results_df
                        st.session_state["trained_models"] = trained_models
                        st.session_state["predictions"] = predictions
                        st.session_state["X_test"] = X_test
                        st.session_state["y_test"] = y_test
                        st.session_state["target_column"] = target_column
                        st.session_state["is_classification"] = is_classification
                        st.session_state["feature_columns"] = list(X.columns)

                    else:
                        st.error(
                            "No model completed successfully. Check the "
                            "selected target and dataset."
                        )

            else:
                st.info(
                    "There are not enough usable rows or features to train "
                    "models with this dataset."
                )

    # Display results saved after training.
    if "test_results_df" in st.session_state:
        st.markdown("---")
        st.subheader("Held-out Test-Set Results")

        test_results_df = st.session_state["test_results_df"]
        cv_results_df = st.session_state["cv_results_df"]

        st.dataframe(
            test_results_df.round(4),
            use_container_width=True
        )

        st.subheader("Cross-Validation Results")
        st.caption(
            "These scores summarize performance across training folds. "
            "They are not the same as the final held-out test-set results."
        )

        st.dataframe(
            cv_results_df.round(4),
            use_container_width=True
        )

        st.markdown("#### Model comparison")

        if st.session_state["is_classification"]:
            metric = st.selectbox(
                "Metric to compare",
                ["Accuracy", "F1 (macro)", "Precision (macro)", "Recall (macro)"],
                key="classification_comparison_metric"
            )

            chart_df = test_results_df.set_index("Model")[[metric]]

            fig, ax = plt.subplots(figsize=(9, 4))
            chart_df.plot(kind="bar", ax=ax)
            ax.set_title(f"Test-set comparison: {metric}")
            ax.set_ylabel(metric)
            ax.tick_params(axis="x", rotation=30)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            best_row = test_results_df.loc[
                test_results_df[metric].idxmax()
            ]
            st.write(
                f"**Best model by {metric}:** {best_row['Model']} "
                f"({best_row[metric]:.4f})"
            )

        else:
            metric = st.selectbox(
                "Metric to compare",
                ["MAE", "RMSE", "R²"],
                key="regression_comparison_metric"
            )

            chart_df = test_results_df.set_index("Model")[[metric]]

            fig, ax = plt.subplots(figsize=(9, 4))
            chart_df.plot(kind="bar", ax=ax)
            ax.set_title(f"Test-set comparison: {metric}")
            ax.set_ylabel(metric)
            ax.tick_params(axis="x", rotation=30)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            if metric in ["MAE", "RMSE"]:
                best_row = test_results_df.loc[
                    test_results_df[metric].idxmin()
                ]
            else:
                best_row = test_results_df.loc[
                    test_results_df[metric].idxmax()
                ]

            st.write(
                f"**Best model by {metric}:** {best_row['Model']} "
                f"({best_row[metric]:.4f})"
            )

        st.markdown("#### Prediction details")

        model_names = list(st.session_state["trained_models"].keys())

        if model_names:
            chosen_model = st.selectbox(
                "Choose a trained model",
                model_names,
                key="prediction_model"
            )

            y_test_saved = st.session_state["y_test"]
            pred_saved = st.session_state["predictions"][chosen_model]

            prediction_df = pd.DataFrame({
                "Actual": np.asarray(y_test_saved),
                "Predicted": np.asarray(pred_saved)
            })

            if not st.session_state["is_classification"]:
                prediction_df["Error"] = (
                    prediction_df["Actual"] - prediction_df["Predicted"]
                )

            st.dataframe(
                prediction_df.head(100).round(4),
                use_container_width=True
            )

            if st.session_state["is_classification"]:
                cm = confusion_matrix(
                    y_test_saved,
                    pred_saved
                )

                fig, ax = plt.subplots(figsize=(7, 5))
                display = ConfusionMatrixDisplay(confusion_matrix=cm)
                display.plot(ax=ax, colorbar=False)
                ax.set_title(f"Confusion Matrix: {chosen_model}")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            else:
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.scatter(y_test_saved, pred_saved, alpha=0.7)

                minimum = min(
                    float(np.min(y_test_saved)),
                    float(np.min(pred_saved))
                )
                maximum = max(
                    float(np.max(y_test_saved)),
                    float(np.max(pred_saved))
                )

                ax.plot(
                    [minimum, maximum],
                    [minimum, maximum],
                    linestyle="--"
                )
                ax.set_xlabel("Actual values")
                ax.set_ylabel("Predicted values")
                ax.set_title(f"Actual vs Predicted: {chosen_model}")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            st.markdown("#### Feature importance")

            if st.button("Calculate permutation feature importance"):
                with st.spinner("Calculating feature importance..."):
                    try:
                        importance_model = st.session_state[
                            "trained_models"
                        ][chosen_model]

                        importance = permutation_importance(
                            importance_model,
                            st.session_state["X_test"],
                            st.session_state["y_test"],
                            n_repeats=5,
                            random_state=42,
                            scoring=(
                                "accuracy"
                                if st.session_state["is_classification"]
                                else "neg_mean_absolute_error"
                            )
                        )

                        importance_df = pd.DataFrame({
                            "Feature": st.session_state["X_test"].columns,
                            "Importance": importance.importances_mean
                        }).sort_values(
                            "Importance",
                            ascending=False
                        )

                        st.dataframe(
                            importance_df.round(4),
                            use_container_width=True
                        )

                        fig, ax = plt.subplots(figsize=(9, 5))
                        plot_df = importance_df.head(15).sort_values(
                            "Importance"
                        )
                        ax.barh(
                            plot_df["Feature"].astype(str),
                            plot_df["Importance"]
                        )
                        ax.set_title(
                            f"Permutation Feature Importance: {chosen_model}"
                        )
                        ax.set_xlabel(
                            "Importance (change in model score)"
                        )
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                    except Exception as error:
                        st.error(
                            f"Could not calculate feature importance: {error}"
                        )

# --------------------------------------------------
# 7. DOWNLOADS
# --------------------------------------------------

with download_tab:
    st.subheader("Download Analysis Results")

    st.download_button(
        "Download uploaded dataset as CSV",
        data=original_df.to_csv(index=False).encode("utf-8"),
        file_name="uploaded_dataset.csv",
        mime="text/csv"
    )

    st.download_button(
        "Download column information",
        data=pd.DataFrame({
            "Column": original_df.columns,
            "Data type": [
                str(original_df[column].dtype)
                for column in original_df.columns
            ],
            "Missing values": [
                int(original_df[column].isna().sum())
                for column in original_df.columns
            ],
            "Unique values": [
                int(original_df[column].nunique(dropna=True))
                for column in original_df.columns
            ]
        }).to_csv(index=False).encode("utf-8"),
        file_name="column_information.csv",
        mime="text/csv"
    )

    if "test_results_df" in st.session_state:
        st.download_button(
            "Download test-set model comparison",
            data=st.session_state["test_results_df"].to_csv(
                index=False
            ).encode("utf-8"),
            file_name="model_comparison.csv",
            mime="text/csv"
        )

        st.download_button(
            "Download cross-validation results",
            data=st.session_state["cv_results_df"].to_csv(
                index=False
            ).encode("utf-8"),
            file_name="cross_validation_results.csv",
            mime="text/csv"
        )

        if "predictions" in st.session_state:
            chosen_names = list(st.session_state["predictions"].keys())

            for model_name in chosen_names:
                safe_name = (
                    model_name.lower()
                    .replace(" ", "_")
                    .replace("/", "_")
                )

                pred = st.session_state["predictions"][model_name]
                y_actual = st.session_state["y_test"]

                prediction_export = pd.DataFrame({
                    "Actual": np.asarray(y_actual),
                    "Predicted": np.asarray(pred)
                })

                st.download_button(
                    f"Download predictions: {model_name}",
                    data=prediction_export.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name=f"{safe_name}_predictions.csv",
                    mime="text/csv",
                    key=f"download_predictions_{safe_name}"
                )

    st.caption(
        "Model scores are estimates from this dataset and split. "
        "A high score does not guarantee performance on new datasets. "
        "Review data leakage, sample size, target choice, and class balance."
    )

st.markdown("---")
st.caption(
    "Student Attendance and Performance Analyzer | "
    "Exploratory data analysis and machine-learning experiments"
)
