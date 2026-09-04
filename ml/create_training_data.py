import pandas as pd
import numpy as np

# ==========================================
# REVENUE RESCUE AI
# CREATE ML TRAINING DATA
# ==========================================

print("\n==========================================")
print("     CREATING ML TRAINING DATA")
print("==========================================")

# Load transaction data
df = pd.read_csv("data/transactions.csv")

# We only train recovery prediction
# on failed transactions
df = df[df["status"] == "FAILED"].copy()

print(f"\nFailed transactions found: {len(df)}")


# ==========================================
# Create realistic recovery probability
# ==========================================

np.random.seed(42)


def recovery_probability(row):

    probability = 0.30

    # Timeout is usually recoverable
    if row["failure_reason"] == "Timeout":
        probability += 0.20

    # Network problems may recover after retry
    elif row["failure_reason"] == "Network Error":
        probability += 0.25

    # Technical errors have moderate recovery chance
    elif row["failure_reason"] == "Technical Error":
        probability += 0.15

    # Insufficient funds are harder to recover immediately
    elif row["failure_reason"] == "Insufficient Funds":
        probability -= 0.20

    # First attempt is more promising
    if row["attempt_number"] == 1:
        probability += 0.15

    elif row["attempt_number"] >= 3:
        probability -= 0.15

    # Customers with successful history
    # are more likely to recover
    if row["previous_successes"] >= 10:
        probability += 0.15

    elif row["previous_successes"] >= 5:
        probability += 0.10

    # Previous failures reduce probability
    if row["previous_failures"] >= 4:
        probability -= 0.10

    # Higher-value transactions get
    # slightly higher priority
    if row["amount"] >= 25000:
        probability += 0.05

    # Keep probability between 5% and 95%
    probability = max(0.05, min(0.95, probability))

    return probability


df["recovery_probability"] = df.apply(
    recovery_probability,
    axis=1
)


# ==========================================
# Generate historical recovery outcome
# ==========================================

df["recovered"] = (
    np.random.random(len(df))
    < df["recovery_probability"]
).astype(int)


# ==========================================
# Create useful ML features
# ==========================================

df["is_high_value"] = (
    df["amount"] >= 25000
).astype(int)

df["is_first_attempt"] = (
    df["attempt_number"] == 1
).astype(int)

df["customer_success_ratio"] = (
    df["previous_successes"]
    /
    (
        df["previous_successes"]
        + df["previous_failures"]
        + 1
    )
)


# ==========================================
# Select ML dataset
# ==========================================

training_columns = [
    "amount",
    "payment_method",
    "failure_reason",
    "attempt_number",
    "previous_successes",
    "previous_failures",
    "is_high_value",
    "is_first_attempt",
    "customer_success_ratio",
    "recovered"
]

training_df = df[training_columns].copy()


# ==========================================
# Save training data
# ==========================================

output_file = "data/recovery_training.csv"

training_df.to_csv(
    output_file,
    index=False
)


# ==========================================
# Display summary
# ==========================================

print("\n==========================================")
print("           TRAINING DATA READY")
print("==========================================")

print(
    f"\nTraining samples: {len(training_df)}"
)

print(
    f"Recovered: "
    f"{training_df['recovered'].sum()}"
)

print(
    f"Not recovered: "
    f"{(training_df['recovered'] == 0).sum()}"
)

print(
    f"\nRecovery rate: "
    f"{training_df['recovered'].mean() * 100:.2f}%"
)

print("\nFeatures:")
print(training_columns)

print("\nSaved to:")
print(output_file)

print("\n==========================================")