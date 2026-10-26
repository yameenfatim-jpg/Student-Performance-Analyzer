import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

# Load dataset
df = pd.read_csv("student-mat.csv", sep=";")

# ============================================================
# DATASET UNDERSTANDING
# ============================================================

print("\n========== DATASET OVERVIEW ==========")

# Number of rows and columns
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

# Column names
print("\nColumn names:")
print(df.columns.tolist())


# ============================================================
# COLUMN CHARACTERISTICS
# ============================================================

print("\n========== COLUMN CHARACTERISTICS ==========")

for column in df.columns:

    unique_count = df[column].nunique()
    missing_count = df[column].isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100

    # Detect numerical or categorical
    if pd.api.types.is_numeric_dtype(df[column]):
        column_type = "Numerical"
    else:
        column_type = "Categorical"

    # Detect binary columns
    if unique_count == 2:
        category_type = "Binary"
    else:
        category_type = "Non-binary"

    print(f"\nColumn: {column}")
    print(f"  Type: {column_type}")
    print(f"  Category type: {category_type}")
    print(f"  Unique values: {unique_count}")
    print(f"  Missing values: {missing_count}")
    print(f"  Missing percentage: {missing_percentage:.2f}%")


# ============================================================
# NUMERICAL COLUMNS
# ============================================================

print("\n========== NUMERICAL COLUMNS ==========")

numerical_columns = df.select_dtypes(include="number").columns.tolist()

print(numerical_columns)


# ============================================================
# CATEGORICAL COLUMNS
# ============================================================

print("\n========== CATEGORICAL COLUMNS ==========")

categorical_columns = df.select_dtypes(exclude="number").columns.tolist()

print(categorical_columns)


# ============================================================
# UNIQUE VALUES
# ============================================================

print("\n========== UNIQUE VALUES ==========")

for column in df.columns:

    print(f"\n{column}:")
    print(df[column].unique())


# ============================================================
# DATA QUALITY
# ============================================================

print("\n========== DATA QUALITY ==========")

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate rows:", df.duplicated().sum())


# ============================================================
# NUMERICAL SUMMARY
# ============================================================

print("\n========== NUMERICAL SUMMARY ==========")

print(df.describe())


# ============================================================
# POSSIBLE IDENTIFIER COLUMNS
# ============================================================

print("\n========== POSSIBLE IDENTIFIER COLUMNS ==========")

id_keywords = [
    "id",
    "student_id",
    "studentid",
    "roll",
    "roll_no",
    "roll_number",
    "registration",
    "registration_no",
    "registration_number",
    "student_number",
    "serial",
    "serial_number"
]

possible_id_columns = []

for column in df.columns:

    column_name = column.lower().replace(" ", "_")

    # Check column name
    name_match = any(keyword in column_name for keyword in id_keywords)

    # Check whether values are mostly unique
    unique_ratio = df[column].nunique() / len(df)

    if name_match and unique_ratio > 0.5:
        possible_id_columns.append(column)

print(possible_id_columns)


# ============================================================
# CONSTANT COLUMNS
# ============================================================

print("\n========== CONSTANT COLUMNS ==========")

constant_columns = []

for column in df.columns:

    if df[column].nunique() <= 1:
        constant_columns.append(column)

print(constant_columns)
# ============================================================
# DATA QUALITY & CLEANING CHECKS
# ============================================================

print("\n\n========== DATA QUALITY CHECK ==========")


# ------------------------------------------------------------
# 1. MISSING VALUES
# ------------------------------------------------------------

print("\n--- MISSING VALUES ---")

missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

for column in df.columns:

    if missing_values[column] > 0:
        print(
            f"{column}: "
            f"{missing_values[column]} missing "
            f"({missing_percentage[column]:.2f}%)"
        )

if missing_values.sum() == 0:
    print("No missing values found.")


# ------------------------------------------------------------
# 2. DUPLICATE ROWS
# ------------------------------------------------------------

print("\n--- DUPLICATE ROWS ---")

duplicate_rows = df.duplicated().sum()

print("Number of duplicate rows:", duplicate_rows)

if duplicate_rows == 0:
    print("No duplicate rows found.")


# ------------------------------------------------------------
# 3. DUPLICATE VALUES IN POSSIBLE ID COLUMNS
# ------------------------------------------------------------

print("\n--- DUPLICATE VALUES IN POSSIBLE ID COLUMNS ---")

if len(possible_id_columns) == 0:

    print("No possible ID column detected.")

else:

    for column in possible_id_columns:

        duplicate_ids = df[column][df[column].duplicated(keep=False)]

        if len(duplicate_ids) == 0:

            print(f"{column}: No duplicate IDs found.")

        else:

            print(
                f"{column}: "
                f"{duplicate_ids.nunique()} duplicated ID values found."
            )


# ------------------------------------------------------------
# 4. EMPTY COLUMNS
# ------------------------------------------------------------

print("\n--- EMPTY COLUMNS ---")

empty_columns = []

for column in df.columns:

    if df[column].isnull().all():
        empty_columns.append(column)

if len(empty_columns) == 0:

    print("No completely empty columns found.")

else:

    print("Empty columns:", empty_columns)


# ------------------------------------------------------------
# 5. CONSTANT COLUMNS
# ------------------------------------------------------------

print("\n--- CONSTANT COLUMNS ---")

if len(constant_columns) == 0:

    print("No constant columns found.")

else:

    print("Constant columns:", constant_columns)


# ------------------------------------------------------------
# 6. NUMERICAL COLUMNS STORED AS TEXT
# ------------------------------------------------------------

print("\n--- POSSIBLE NUMERICAL COLUMNS STORED AS TEXT ---")

possible_numeric_text_columns = []

for column in categorical_columns:

    converted_values = pd.to_numeric(df[column], errors="coerce")

    valid_numeric_values = converted_values.notna().sum()

    if valid_numeric_values > 0:

        numeric_ratio = valid_numeric_values / len(df)

        if numeric_ratio >= 0.80:
            possible_numeric_text_columns.append(column)

if len(possible_numeric_text_columns) == 0:

    print("No obvious numerical columns stored as text found.")

else:

    print(
        "Possible numerical columns stored as text:",
        possible_numeric_text_columns
    )


# ------------------------------------------------------------
# 7. NEGATIVE VALUES
# ------------------------------------------------------------

print("\n--- NEGATIVE VALUES ---")

negative_values_found = False

for column in numerical_columns:

    negative_count = (df[column] < 0).sum()

    if negative_count > 0:

        negative_values_found = True

        print(
            f"{column}: "
            f"{negative_count} negative values"
        )

if not negative_values_found:

    print("No negative numerical values found.")


# ------------------------------------------------------------
# 8. POTENTIAL OUTLIERS USING IQR
# ------------------------------------------------------------

print("\n--- POTENTIAL OUTLIERS ---")

outlier_summary = {}

for column in numerical_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_count = (
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ).sum()

    outlier_summary[column] = outlier_count

    if outlier_count > 0:

        print(
            f"{column}: "
            f"{outlier_count} potential outliers"
        )

if sum(outlier_summary.values()) == 0:

    print("No potential outliers detected.")


# ------------------------------------------------------------
# 9. FINAL DATA QUALITY SUMMARY
# ------------------------------------------------------------

print("\n========== DATA QUALITY SUMMARY ==========")

total_missing = missing_values.sum()
total_duplicates = duplicate_rows
total_empty_columns = len(empty_columns)
total_constant_columns = len(constant_columns)

print("Total missing values:", total_missing)
print("Duplicate rows:", total_duplicates)
print("Empty columns:", total_empty_columns)
print("Constant columns:", total_constant_columns)
print("Possible ID columns:", possible_id_columns)
print(
    "Columns with potential numerical-text issues:",
    possible_numeric_text_columns
)

# ============================================================
# AUTOMATIC EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

print("\n\n========== AUTOMATIC EDA ==========")


# ------------------------------------------------------------
# 1. NUMERICAL COLUMN ANALYSIS
# ------------------------------------------------------------

print("\n========== NUMERICAL ANALYSIS ==========")

for column in numerical_columns:

    series = df[column].dropna()

    print(f"\n--- {column} ---")

    print("Mean:", round(series.mean(), 2))
    print("Median:", round(series.median(), 2))
    print("Minimum:", series.min())
    print("Maximum:", series.max())
    print("Standard deviation:", round(series.std(), 2))
    print("Skewness:", round(series.skew(), 2))


# ------------------------------------------------------------
# 2. CATEGORICAL COLUMN ANALYSIS
# ------------------------------------------------------------

print("\n========== CATEGORICAL ANALYSIS ==========")

for column in categorical_columns:

    print(f"\n--- {column} ---")

    value_counts = df[column].value_counts()

    print("Number of categories:", df[column].nunique())

    print("Most common values:")

    print(value_counts.head(5))


# ------------------------------------------------------------
# 3. CORRELATION ANALYSIS
# ------------------------------------------------------------

print("\n========== CORRELATION ANALYSIS ==========")

if len(numerical_columns) >= 2:

    correlation_matrix = df[numerical_columns].corr()

    print("\nCorrelation matrix:")
    print(correlation_matrix.round(2))

else:

    correlation_matrix = None

    print("Not enough numerical columns for correlation analysis.")


# ------------------------------------------------------------
# 4. STRONG NUMERICAL RELATIONSHIPS
# ------------------------------------------------------------

print("\n========== STRONG NUMERICAL RELATIONSHIPS ==========")

if correlation_matrix is not None:

    strong_relationships = []

    for i in range(len(numerical_columns)):

        for j in range(i + 1, len(numerical_columns)):

            column_1 = numerical_columns[i]
            column_2 = numerical_columns[j]

            correlation = correlation_matrix.loc[
                column_1,
                column_2
            ]

            if abs(correlation) >= 0.50:

                strong_relationships.append(
                    (
                        column_1,
                        column_2,
                        correlation
                    )
                )

    if len(strong_relationships) == 0:

        print("No strong numerical relationships found.")

    else:

        for column_1, column_2, correlation in strong_relationships:

            relationship_type = (
                "Positive"
                if correlation > 0
                else "Negative"
            )

            print(
                f"{column_1} ↔ {column_2}: "
                f"{correlation:.2f} "
                f"({relationship_type})"
            )


# ------------------------------------------------------------
# 5. LOW-CORRELATION RELATIONSHIPS
# ------------------------------------------------------------

print("\n========== WEAK NUMERICAL RELATIONSHIPS ==========")

if correlation_matrix is not None:

    weak_relationships = []

    for i in range(len(numerical_columns)):

        for j in range(i + 1, len(numerical_columns)):

            column_1 = numerical_columns[i]
            column_2 = numerical_columns[j]

            correlation = correlation_matrix.loc[
                column_1,
                column_2
            ]

            if abs(correlation) < 0.20:

                weak_relationships.append(
                    (
                        column_1,
                        column_2,
                        correlation
                    )
                )

    print(
        "Number of weak relationships:",
        len(weak_relationships)
    )


# ------------------------------------------------------------
# 6. GROUP ANALYSIS
# ------------------------------------------------------------

print("\n========== AUTOMATIC GROUP ANALYSIS ==========")

if len(categorical_columns) > 0 and len(numerical_columns) > 0:

    for categorical_column in categorical_columns:

        # Avoid extremely high-cardinality columns
        if df[categorical_column].nunique() <= 10:

            print(
                f"\n--- Numerical averages by "
                f"{categorical_column} ---"
            )

            grouped_data = df.groupby(
                categorical_column,
                observed=True
            )[numerical_columns].mean()

            print(grouped_data.round(2))

else:

    print("Not enough categorical/numerical columns for group analysis.")


# ------------------------------------------------------------
# 7. DISTRIBUTION SUMMARY
# ------------------------------------------------------------

print("\n========== DISTRIBUTION SUMMARY ==========")

for column in numerical_columns:

    series = df[column].dropna()

    skewness = series.skew()

    if skewness > 1:

        distribution = "Highly right-skewed"

    elif skewness > 0.5:

        distribution = "Moderately right-skewed"

    elif skewness < -1:

        distribution = "Highly left-skewed"

    elif skewness < -0.5:

        distribution = "Moderately left-skewed"

    else:

        distribution = "Approximately symmetric"

    print(
        f"{column}: {distribution}"
    )


# ------------------------------------------------------------
# 8. MOST VARIABLE NUMERICAL COLUMNS
# ------------------------------------------------------------

print("\n========== NUMERICAL VARIABILITY ==========")

if len(numerical_columns) > 0:

    variability = df[numerical_columns].std().sort_values(
        ascending=False
    )

    print(
        "Columns with highest standard deviation:"
    )

    print(variability)


# ------------------------------------------------------------
# 9. EDA SUMMARY
# ------------------------------------------------------------

print("\n========== EDA SUMMARY ==========")

print(
    "Numerical columns analyzed:",
    len(numerical_columns)
)

print(
    "Categorical columns analyzed:",
    len(categorical_columns)
)

if correlation_matrix is not None:

    print(
        "Strong relationships found:",
        len(strong_relationships)
    )

print(
    "Group analyses performed:",
    len(categorical_columns)
)

print("\nAutomatic EDA completed.")


# ============================================================
# AUTOMATIC VISUALIZATIONS
# ============================================================

import math

print("\n\n========== AUTOMATIC VISUALIZATIONS ==========")


# ------------------------------------------------------------
# 1. HISTOGRAMS FOR NUMERICAL COLUMNS
# ------------------------------------------------------------

print("\nCreating histograms...")

for column in numerical_columns:

    values = df[column].dropna()

    # Skip columns with no usable values
    if len(values) == 0:
        continue

    # A histogram is not useful for a constant column
    if values.nunique() <= 1:
        continue

    plt.figure(figsize=(8, 5))

    plt.hist(values, bins="auto", edgecolor="black")

    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.close()


# ------------------------------------------------------------
# 2. BAR CHARTS FOR CATEGORICAL COLUMNS
# ------------------------------------------------------------

print("\nCreating categorical bar charts...")

for column in categorical_columns:

    values = df[column].dropna()
    category_counts = values.value_counts()

    # Skip empty columns
    if len(category_counts) == 0:
        continue

    # Skip columns with only one category
    if len(category_counts) <= 1:
        continue

    # Limit chart size for high-cardinality columns
    if len(category_counts) > 15:
        category_counts = category_counts.head(15)

    plt.figure(figsize=(8, 5))

    category_counts.plot(kind="bar")

    plt.title(f"Category Counts: {column}")
    plt.xlabel(column)
    plt.ylabel("Number of records")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.close()


# ------------------------------------------------------------
# 3. SCATTER PLOTS FOR NUMERICAL PAIRS
# ------------------------------------------------------------

print("\nCreating scatter plots...")

scatter_plot_limit = 10
scatter_plots_created = 0

if len(numerical_columns) >= 2:

    for i in range(len(numerical_columns)):

        for j in range(i + 1, len(numerical_columns)):

            if scatter_plots_created >= scatter_plot_limit:
                break

            column_x = numerical_columns[i]
            column_y = numerical_columns[j]

            # Skip columns with no variation
            if df[column_x].nunique() <= 1:
                continue

            if df[column_y].nunique() <= 1:
                continue

            plt.figure(figsize=(7, 5))

            plt.scatter(
                df[column_x],
                df[column_y],
                alpha=0.6
            )

            plt.title(f"{column_x} vs {column_y}")
            plt.xlabel(column_x)
            plt.ylabel(column_y)

            plt.tight_layout()
            plt.close()

            scatter_plots_created += 1

        if scatter_plots_created >= scatter_plot_limit:
            break

print("Scatter plots created:", scatter_plots_created)


# ------------------------------------------------------------
# 4. CORRELATION HEATMAP
# ------------------------------------------------------------

print("\nCreating correlation heatmap...")

if len(numerical_columns) >= 2:

    heatmap_data = df[numerical_columns].corr()

    figure_size = max(
        8,
        min(16, len(numerical_columns) * 0.7)
    )

    plt.figure(
        figsize=(figure_size, figure_size)
    )

    plt.imshow(
        heatmap_data,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="auto"
    )

    plt.colorbar(label="Correlation")

    plt.xticks(
        range(len(numerical_columns)),
        numerical_columns,
        rotation=90
    )

    plt.yticks(
        range(len(numerical_columns)),
        numerical_columns
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()
    plt.close()

else:

    print(
        "Heatmap skipped: at least two numerical columns are required."
    )


# ------------------------------------------------------------
# 5. NUMERICAL COMPARISON BY CATEGORICAL GROUP
# ------------------------------------------------------------

print("\nCreating group comparison charts...")

group_plot_limit = 10
group_plots_created = 0

for categorical_column in categorical_columns:

    # Only use categorical columns with a manageable number of groups
    if df[categorical_column].nunique() < 2:
        continue

    if df[categorical_column].nunique() > 10:
        continue

    for numerical_column in numerical_columns:

        if group_plots_created >= group_plot_limit:
            break

        grouped_means = df.groupby(
            categorical_column,
            observed=True
        )[numerical_column].mean().dropna()

        if len(grouped_means) < 2:
            continue

        plt.figure(figsize=(8, 5))

        grouped_means.plot(kind="bar")

        plt.title(
            f"Average {numerical_column} by {categorical_column}"
        )

        plt.xlabel(categorical_column)
        plt.ylabel(f"Average {numerical_column}")

        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        plt.close()

        group_plots_created += 1

    if group_plots_created >= group_plot_limit:
        break

print("Group comparison charts created:", group_plots_created)


# ------------------------------------------------------------
# 6. VISUALIZATION SUMMARY
# ------------------------------------------------------------

print("\n========== VISUALIZATION SUMMARY ==========")

print(
    "Numerical columns available:",
    len(numerical_columns)
)

print(
    "Categorical columns available:",
    len(categorical_columns)
)

print(
    "Scatter plots created:",
    scatter_plots_created
)

print(
    "Group comparison charts created:",
    group_plots_created
)

print("\nAutomatic visualizations completed.")



# ============================================================
# AUTOMATIC ML TASK DETECTION
# ============================================================

print("\n\n========== AUTOMATIC ML TASK DETECTION ==========")


# ------------------------------------------------------------
# 1. CHECK WHETHER THE DATASET IS SUITABLE FOR ML
# ------------------------------------------------------------

print("\n--- DATASET SUITABILITY ---")

if len(df) < 20:

    print(
        "Warning: The dataset has fewer than 20 rows. "
        "Machine-learning results may be unreliable."
    )

else:

    print("Dataset has enough rows for initial ML exploration.")


# ------------------------------------------------------------
# 2. PREPARE POSSIBLE TARGET COLUMNS
# ------------------------------------------------------------

print("\n--- POSSIBLE TARGET COLUMNS ---")

# Exclude columns previously identified as possible IDs
excluded_columns = set(possible_id_columns)

# Exclude constant columns
excluded_columns.update(constant_columns)

# Exclude completely empty columns
excluded_columns.update(empty_columns)

# Keep columns that have enough distinct values to be candidates
candidate_columns = []

for column in df.columns:

    if column in excluded_columns:
        continue

    if df[column].nunique() < 2:
        continue

    candidate_columns.append(column)

print("Potential target columns:")

if candidate_columns:

    for column in candidate_columns:
        print(
            f"{column}: "
            f"{df[column].nunique()} unique values"
        )

else:

    print("No suitable target candidates found.")


# ------------------------------------------------------------
# 3. SUGGEST REGRESSION OR CLASSIFICATION
# ------------------------------------------------------------

print("\n--- POSSIBLE ML TASKS ---")

ml_task_candidates = []

for column in candidate_columns:

    unique_count = df[column].nunique()
    unique_ratio = unique_count / len(df)

    if pd.api.types.is_numeric_dtype(df[column]):

        # Numeric columns with many distinct values may suit regression
        if unique_count > 10:

            task = "Possible regression"

        # Numeric columns with few distinct values may suit classification
        else:

            task = "Possible classification"

    else:

        # Avoid targets with nearly one unique category per row
        if unique_ratio > 0.50:

            print(
                f"{column}: Skipped as a target candidate "
                "because it has too many distinct categories."
            )

            continue

        task = "Possible classification"

    ml_task_candidates.append(
        {
            "target": column,
            "task": task,
            "unique_values": unique_count
        }
    )

    print(
        f"Target: {column} | "
        f"Suggested task: {task}"
    )


# ------------------------------------------------------------
# 4. ML DETECTION SUMMARY
# ------------------------------------------------------------

print("\n========== ML DETECTION SUMMARY ==========")

print(
    "Candidate columns examined:",
    len(candidate_columns)
)

print(
    "Potential ML tasks identified:",
    len(ml_task_candidates)
)

print(
    "Note: These are suggestions, not validated ML tasks."
)

print("\nAutomatic ML task detection completed.")



# ============================================================
# PREPARING DATA FOR MACHINE LEARNING
# ============================================================

from sklearn.model_selection import train_test_split

print("\n\n========== ML DATA PREPARATION ==========")

# 1. Select a target for this learning example
target_column = "G3"

# 2. Check that the target exists
if target_column not in df.columns:

    print(f"Target column '{target_column}' was not found.")

else:

    # 3. Separate features (X) and target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    print("\nTarget column:", target_column)
    print("Feature columns:", X.columns.tolist())
    print("Number of features:", X.shape[1])
    print("Number of records:", X.shape[0])

    # 4. Remove rows where the target is missing
    valid_target_rows = y.notna()

    X = X.loc[valid_target_rows].copy()
    y = y.loc[valid_target_rows].copy()

    # 5. Split into training and testing sets
    if len(X) < 5:

        print("Not enough records for a train-test split.")

    else:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )

        print("\n--- TRAIN-TEST SPLIT ---")
        print("Training feature records:", len(X_train))
        print("Testing feature records:", len(X_test))
        print("Training target records:", len(y_train))
        print("Testing target records:", len(y_test))

        print("\nData preparation completed.")



# ============================================================
# PREPROCESSING: MISSING VALUES AND CATEGORICAL FEATURES
# ============================================================

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

print("\n\n========== ML PREPROCESSING ==========")

# 1. Identify numerical and categorical features
numerical_features = X_train.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude="number"
).columns.tolist()

print("Numerical features:", numerical_features)
print("Categorical features:", categorical_features)

# 2. Prepare numerical preprocessing
numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# 3. Prepare categorical preprocessing
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# 4. Combine both preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ],
    remainder="drop"
)

# 5. Learn preprocessing from training data
X_train_processed = preprocessor.fit_transform(X_train)

# 6. Apply the learned preprocessing to testing data
X_test_processed = preprocessor.transform(X_test)

print("\n--- PREPROCESSING SUMMARY ---")
print("Original training shape:", X_train.shape)
print("Processed training shape:", X_train_processed.shape)

print("Original testing shape:", X_test.shape)
print("Processed testing shape:", X_test_processed.shape)

print("\nPreprocessing completed successfully.")


# ============================================================
# PART 8: TRAIN AND EVALUATE THE FIRST ML MODEL
# ============================================================

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("\n\n========== MODEL TRAINING ==========")

# 1. Create a machine-learning pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

# 2. Train the model using training data
model.fit(X_train, y_train)

print("Model training completed successfully!")

# 3. Predict final grades for the test records
y_pred = model.predict(X_test)

print("\nPredictions generated successfully!")

# 4. Compare actual grades with predicted grades
results = pd.DataFrame({
    "Actual Grade": y_test.to_numpy(),
    "Predicted Grade": y_pred
})

results["Predicted Grade"] = results["Predicted Grade"].round(2)

print("\n--- ACTUAL VS PREDICTED GRADES ---")
print(results.head(10).to_string(index=False))

# 5. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n--- MODEL EVALUATION ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.3f}")

print("\nModel training and evaluation completed.")


# ============================================================
# PART 9: BASELINE MODEL COMPARISON
# ============================================================

from sklearn.dummy import DummyRegressor

print("\n\n========== BASELINE MODEL COMPARISON ==========")

# 1. Create a simple baseline model
baseline_model = DummyRegressor(strategy="mean")

# 2. Train it using the same training target
baseline_model.fit(X_train_processed, y_train)

# 3. Generate predictions on the same test records
baseline_predictions = baseline_model.predict(X_test_processed)

# 4. Evaluate the baseline
baseline_mae = mean_absolute_error(y_test, baseline_predictions)
baseline_rmse = mean_squared_error(
    y_test, baseline_predictions
) ** 0.5
baseline_r2 = r2_score(y_test, baseline_predictions)

# 5. Compare the baseline with Linear Regression
comparison = pd.DataFrame({
    "Model": ["Dummy Regressor", "Linear Regression"],
    "MAE": [
        baseline_mae,
        mean_absolute_error(y_test, y_pred)
    ],
    "RMSE": [
        baseline_rmse,
        rmse
    ],
    "R2": [
        baseline_r2,
        r2
    ]
})

print("\n--- MODEL COMPARISON ---")
print(comparison.round(3).to_string(index=False))

print("\nLower MAE and RMSE are better.")
print("Higher R2 is generally better.")
print("Compare both models on the same test records.")


# STEP 10: ERROR ANALYSIS

# Create a table comparing actual and predicted grades
error_analysis = pd.DataFrame({
    "Actual Grade": y_test.to_numpy(),
    "Predicted Grade": y_pred
})

# Calculate the signed prediction error
error_analysis["Residual"] = (
    error_analysis["Actual Grade"]
    - error_analysis["Predicted Grade"]
)

# Calculate the absolute prediction error
error_analysis["Absolute Error"] = (
    error_analysis["Residual"].abs()
)

# Display predictions with the largest errors
largest_errors = error_analysis.sort_values(
    by="Absolute Error",
    ascending=False
)

print("\n--- ERROR ANALYSIS ---")
print(largest_errors.head(10).round(2).to_string(index=False))

# Summarize the errors
print("\nAverage absolute error:",
      round(error_analysis["Absolute Error"].mean(), 2))

print("Largest absolute error:",
      round(error_analysis["Absolute Error"].max(), 2))

print("Predictions within 2 grade points:",
      (error_analysis["Absolute Error"] <= 2).sum(),
      "out of", len(error_analysis))


# STEP 11: UNDERSTANDING FEATURE INFLUENCE

# Get the trained Linear Regression model
linear_model = model.named_steps["regressor"]

# Get the feature names after preprocessing
feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

# Get the learned coefficients
coefficients = linear_model.coef_

# Create a table of features and their coefficients
feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

# Calculate coefficient magnitudes
feature_importance["Absolute Coefficient"] = (
    feature_importance["Coefficient"].abs()
)

# Sort by absolute coefficient magnitude
feature_importance = feature_importance.sort_values(
    by="Absolute Coefficient",
    ascending=False
)

print("\n--- TOP 15 FEATURES BY COEFFICIENT MAGNITUDE ---")
print(
    feature_importance.head(15)
    .round(3)
    .to_string(index=False)
)



# ============================================================
# STEP 12: CROSS-VALIDATION FOR ALL THREE MODELS
# ============================================================

from sklearn.model_selection import KFold, cross_val_score
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

print("\n--- STEP 12: CROSS-VALIDATION MODEL COMPARISON ---")

# Use the training data only.
# The test set remains separate for final evaluation.

cv_folds = min(5, len(X_train))

if cv_folds >= 2:

    cv = KFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=42
    )

    cv_models = {
        "Dummy Baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    }

    cv_results = []

    for model_name, estimator in cv_models.items():

        cv_pipeline = Pipeline(steps=[
            ("preprocessor", clone(preprocessor)),
            ("regressor", estimator)
        ])

        scores = cross_val_score(
            cv_pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1
        )

        mae_scores = -scores

        cv_results.append({
            "Model": model_name,
            "Average MAE": mae_scores.mean(),
            "MAE Variation": mae_scores.std()
        })

        print(f"\n{model_name}")
        print("MAE in each fold:", np.round(mae_scores, 2))
        print("Average MAE:", round(mae_scores.mean(), 3))
        print("MAE variation:", round(mae_scores.std(), 3))

    cv_comparison = pd.DataFrame(cv_results)
    cv_comparison = cv_comparison.sort_values(
        by="Average MAE"
    ).reset_index(drop=True)

    print("\n--- CROSS-VALIDATION SUMMARY ---")
    print(cv_comparison.round(3).to_string(index=False))

    cv_comparison.to_csv(
        "cross_validation_results.csv",
        index=False
    )

    print("\nSaved cross_validation_results.csv")

else:
    print("Not enough training rows for cross-validation.")


# ============================================================
# STEP 13: COMPARE LINEAR REGRESSION WITH RANDOM FOREST
# ============================================================

print("\n--- STEP 13: MODEL COMPARISON ---")

random_forest_model = Pipeline(steps=[
    ("preprocessor", clone(preprocessor)),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

random_forest_model.fit(X_train, y_train)
rf_predictions = random_forest_model.predict(X_test)

comparison_results = pd.DataFrame({
    "Model": ["Dummy Regressor", "Linear Regression", "Random Forest"],
    "MAE": [
        mean_absolute_error(y_test, baseline_predictions),
        mean_absolute_error(y_test, y_pred),
        mean_absolute_error(y_test, rf_predictions)
    ],
    "RMSE": [
        mean_squared_error(y_test, baseline_predictions) ** 0.5,
        mean_squared_error(y_test, y_pred) ** 0.5,
        mean_squared_error(y_test, rf_predictions) ** 0.5
    ],
    "R2": [
        r2_score(y_test, baseline_predictions),
        r2_score(y_test, y_pred),
        r2_score(y_test, rf_predictions)
    ]
})

print(comparison_results.round(3).to_string(index=False))

# Select the model with the lowest MAE on the test set.
best_model_name = comparison_results.loc[
    comparison_results["MAE"].idxmin(), "Model"
]

print("\nLowest test-set MAE:", best_model_name)

if best_model_name == "Random Forest":
    best_model = random_forest_model
    best_predictions = rf_predictions
elif best_model_name == "Linear Regression":
    best_model = model
    best_predictions = y_pred
else:
    best_model = baseline_model
    best_predictions = baseline_predictions

print(
    "Important: this is a preliminary comparison on one test set. "
    "Use cross-validation and a final untouched test set for a more reliable "
    "model-selection process."
)


# ============================================================
# STEP 14: RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

print("\n--- STEP 14: RANDOM FOREST FEATURE IMPORTANCE ---")

# Feature importance is available for the fitted Random Forest.
rf_regressor = random_forest_model.named_steps["regressor"]
rf_preprocessor = random_forest_model.named_steps["preprocessor"]

rf_feature_names = rf_preprocessor.get_feature_names_out()
rf_importances = rf_regressor.feature_importances_

rf_importance_table = pd.DataFrame({
    "Feature": rf_feature_names,
    "Importance": rf_importances
}).sort_values(
    by="Importance",
    ascending=False
)

print(
    rf_importance_table.head(15)
    .round(4)
    .to_string(index=False)
)

# Save a bar chart of the top 15 features.
top_features = rf_importance_table.head(15).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10, 7))
plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)
plt.title("Top 15 Random Forest Features")
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("random_forest_feature_importance.png", dpi=150)
plt.close()

print("Feature importance chart saved as "
      "random_forest_feature_importance.png")


# ============================================================
# STEP 15: PREDICTION ERROR VISUALIZATION
# ============================================================

print("\n--- STEP 15: PREDICTION ERROR VISUALIZATION ---")

error_analysis = pd.DataFrame({
    "Actual Grade": y_test.to_numpy(),
    "Predicted Grade": best_predictions
})

error_analysis["Residual"] = (
    error_analysis["Actual Grade"]
    - error_analysis["Predicted Grade"]
)

error_analysis["Absolute Error"] = (
    error_analysis["Residual"].abs()
)

# Actual versus predicted plot.
plt.figure(figsize=(8, 6))
plt.scatter(
    error_analysis["Actual Grade"],
    error_analysis["Predicted Grade"],
    alpha=0.7
)

minimum = min(
    error_analysis["Actual Grade"].min(),
    error_analysis["Predicted Grade"].min()
)
maximum = max(
    error_analysis["Actual Grade"].max(),
    error_analysis["Predicted Grade"].max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Grade")
plt.ylabel("Predicted Grade")
plt.title("Actual vs Predicted Grades")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.close()

# Residual distribution plot.
plt.figure(figsize=(8, 6))
plt.hist(error_analysis["Residual"], bins="auto", edgecolor="black")
plt.axvline(0, linestyle="--")
plt.xlabel("Residual (actual - predicted)")
plt.ylabel("Frequency")
plt.title("Distribution of Prediction Errors")
plt.tight_layout()
plt.savefig("prediction_residuals.png", dpi=150)
plt.close()

print("Saved: actual_vs_predicted.png")
print("Saved: prediction_residuals.png")

print("\n--- FINAL ERROR SUMMARY ---")
print("Average absolute error:",
      round(error_analysis["Absolute Error"].mean(), 2))
print("Largest absolute error:",
      round(error_analysis["Absolute Error"].max(), 2))
print("Predictions within 2 grade points:",
      int((error_analysis["Absolute Error"] <= 2).sum()),
      "out of", len(error_analysis))


# ============================================================
# STEP 16: SAVE MODEL EVALUATION RESULTS
# ============================================================

comparison_results.to_csv("model_comparison_results.csv", index=False)
error_analysis.to_csv("prediction_error_analysis.csv", index=False)
rf_importance_table.to_csv("random_forest_feature_importance.csv", index=False)

print("\n--- STEP 16: FILES SAVED ---")
print("model_comparison_results.csv")
print("prediction_error_analysis.csv")
print("random_forest_feature_importance.csv")


# ============================================================
# PROJECT MILESTONE
# ============================================================

print("\n==============================================")
print("STUDENT ATTENDANCE AND PERFORMANCE ANALYZER")
print("ML EXPERIMENT COMPLETED")
print("==============================================")
print("Completed in this script:")
print("1. Data exploration and quality checks")
print("2. Automatic exploratory analysis")
print("3. Visualization generation")
print("4. Initial target preparation")
print("5. Data preprocessing")
print("6. Linear Regression training")
print("7. Baseline comparison")
print("8. Prediction error analysis")
print("9. Cross-validation")
print("10. Random Forest comparison and feature importance")
print("11. Evaluation files and charts saved")
print("\nNote: This experiment uses G3 as its target.")
print("The adaptive multi-dataset application is a separate")
print("development stage still to be built and tested.")
print("==============================================")


# ============================================================
# STEP 17: REUSABLE AUTOMATIC DATASET ANALYZER
# ============================================================

def analyze_uploaded_dataset(uploaded_df):
    """
    Analyze a dataset with unknown columns.
    Returns a dictionary containing useful analysis results.
    """

    if not isinstance(uploaded_df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if uploaded_df.empty:
        raise ValueError("The uploaded dataset is empty.")

    data = uploaded_df.copy()

    # Clean column names
    data.columns = [str(column).strip() for column in data.columns]

    # Remove completely empty columns
    empty_columns = [
        column for column in data.columns
        if data[column].isna().all()
    ]

    data = data.drop(columns=empty_columns)

    # Identify column types
    numerical_columns = data.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = data.select_dtypes(
        exclude="number"
    ).columns.tolist()

    # Dataset structure
    dataset_shape = data.shape

    # Missing values
    missing_values = data.isna().sum()

    missing_percentage = (
        data.isna().mean() * 100
    ).round(2)

    # Duplicate rows
    duplicate_rows = int(data.duplicated().sum())

    # Constant columns
    constant_columns = [
        column for column in data.columns
        if data[column].nunique(dropna=False) <= 1
    ]

    # Numerical statistics
    numerical_summary = data[numerical_columns].describe().T

    # Categorical statistics
    categorical_summary = {}

    for column in categorical_columns:
        categorical_summary[column] = (
            data[column]
            .value_counts(dropna=False)
            .head(20)
        )

    # Correlations between numerical columns
    if len(numerical_columns) >= 2:
        correlation_matrix = data[numerical_columns].corr()
    else:
        correlation_matrix = pd.DataFrame()

    # Strong numerical correlations
    strong_correlations = []

    if not correlation_matrix.empty:
        for i, first_column in enumerate(correlation_matrix.columns):
            for second_column in correlation_matrix.columns[i + 1:]:
                correlation_value = correlation_matrix.loc[
                    first_column, second_column
                ]

                if (
                    pd.notna(correlation_value)
                    and abs(correlation_value) >= 0.50
                ):
                    strong_correlations.append({
                        "Feature 1": first_column,
                        "Feature 2": second_column,
                        "Correlation": round(
                            float(correlation_value), 3
                        )
                    })

    strong_correlations_df = pd.DataFrame(
        strong_correlations,
        columns=["Feature 1", "Feature 2", "Correlation"]
    )

    # Candidate prediction targets
    candidate_targets = []

    for column in data.columns:
        non_missing_count = int(data[column].notna().sum())
        unique_count = int(data[column].nunique(dropna=True))

        if non_missing_count >= 10 and unique_count > 1:
            if pd.api.types.is_numeric_dtype(data[column]):
                task_type = (
                    "Regression"
                    if unique_count > 15
                    else "Possible classification or regression"
                )
            else:
                task_type = "Classification"

            candidate_targets.append({
                "Column": column,
                "Data type": str(data[column].dtype),
                "Unique values": unique_count,
                "Possible task": task_type
            })

    candidate_targets_df = pd.DataFrame(
        candidate_targets,
        columns=[
            "Column",
            "Data type",
            "Unique values",
            "Possible task"
        ]
    )

    # Collect all results
    analysis_results = {
        "data": data,
        "dataset_shape": dataset_shape,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "empty_columns": empty_columns,
        "missing_values": missing_values,
        "missing_percentage": missing_percentage,
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "numerical_summary": numerical_summary,
        "categorical_summary": categorical_summary,
        "correlation_matrix": correlation_matrix,
        "strong_correlations": strong_correlations_df,
        "candidate_targets": candidate_targets_df
    }

    return analysis_results


print("\n--- STEP 17: REUSABLE DATASET ANALYZER READY ---")
print("The function analyze_uploaded_dataset() can now")
print("analyze different pandas DataFrames.")
print("It does not require a fixed set of column names.")


# ============================================================
# STEP 18: TEST THE ANALYZER ON THE CURRENT DATASET
# ============================================================

print("\n--- STEP 18: TESTING AUTOMATIC ANALYSIS ---")

automatic_results = analyze_uploaded_dataset(df)

print("\nDataset shape:")
print(automatic_results["dataset_shape"])

print("\nNumerical columns:")
print(automatic_results["numerical_columns"])

print("\nCategorical columns:")
print(automatic_results["categorical_columns"])

print("\nMissing values:")
print(
    automatic_results["missing_values"][
        automatic_results["missing_values"] > 0
    ]
)

print("\nDuplicate rows:")
print(automatic_results["duplicate_rows"])

print("\nConstant columns:")
print(automatic_results["constant_columns"])

print("\nNumerical summary:")
print(
    automatic_results["numerical_summary"]
    .round(2)
    .to_string()
)

print("\nStrong numerical correlations:")
print(
    automatic_results["strong_correlations"]
    .to_string(index=False)
)

print("\nPossible prediction targets:")
print(
    automatic_results["candidate_targets"]
    .to_string(index=False)
)


# ============================================================
# STEP 19: EXPORT AUTOMATIC ANALYSIS REPORT
# ============================================================

print("\n--- STEP 19: EXPORTING ANALYSIS REPORT ---")

automatic_results["data"].head(100).to_csv(
    "dataset_preview.csv",
    index=False
)

automatic_results["missing_percentage"].rename(
    "Missing Percentage"
).to_csv("missing_values_report.csv")

automatic_results["numerical_summary"].to_csv(
    "numerical_summary_report.csv"
)

automatic_results["strong_correlations"].to_csv(
    "strong_correlations_report.csv",
    index=False
)

automatic_results["candidate_targets"].to_csv(
    "candidate_prediction_targets.csv",
    index=False
)

print("Saved dataset_preview.csv")
print("Saved missing_values_report.csv")
print("Saved numerical_summary_report.csv")
print("Saved strong_correlations_report.csv")
print("Saved candidate_prediction_targets.csv")


# ============================================================
# STEP 20: FINAL PROJECT STATUS
# ============================================================

print("\n==============================================")
print("STUDENT ATTENDANCE AND PERFORMANCE ANALYZER")
print("AUTOMATIC ANALYSIS MODULE READY")
print("==============================================")
print("The project now includes:")
print("1. Exploratory data analysis")
print("2. Data-quality checks")
print("3. Numerical and categorical summaries")
print("4. Correlation analysis")
print("5. Initial ML experiments")
print("6. Model comparison and error analysis")
print("7. Reusable analysis for different DataFrames")
print("8. Exportable analysis reports")
print("\nNext development stage:")
print("Connect file upload, analysis, visualizations, and ML")
print("into one interactive application.")
print("==============================================")



