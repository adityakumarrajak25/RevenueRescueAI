import pandas as pd
import joblib
import os


# ==========================================
# REVENUE RESCUE AI
# RECOVERY DECISION ENGINE
# ==========================================

print("\n" + "=" * 60)
print("              REVENUE RESCUE AI")
print("             RECOVERY DECISION ENGINE")
print("=" * 60)


# ==========================================
# 1. LOAD DATA
# ==========================================

data_file = "data/transactions.csv"
model_file = "ml/recovery_model.joblib"

if not os.path.exists(data_file):
    print(f"\nERROR: Data file not found: {data_file}")
    exit()

if not os.path.exists(model_file):
    print(f"\nERROR: ML model not found: {model_file}")
    print("\nPlease train the model first:")
    print("python ml/train_recovery_model.py")
    exit()


df = pd.read_csv(data_file)
# ==========================================
# CREATE FEATURES REQUIRED BY ML MODEL
# ==========================================

# Attempt number
if "attempt_number" not in df.columns:
    df["attempt_number"] = 1

# Previous successes
if "previous_successes" not in df.columns:
    df["previous_successes"] = 0

# Previous failures
if "previous_failures" not in df.columns:
    df["previous_failures"] = 0

# High-value transaction
df["is_high_value"] = (df["amount"] >= 10000).astype(int)

# First attempt
df["is_first_attempt"] = (df["attempt_number"] == 1).astype(int)

# Customer success ratio
total_attempts = (
    df["previous_successes"]
    + df["previous_failures"]
)

df["customer_success_ratio"] = (
    df["previous_successes"]
    / total_attempts.replace(0, 1)
)
model = joblib.load(model_file)

print("\nData loaded successfully.")
print("Transactions:", len(df))


# ==========================================
# 2. SELECT FAILED TRANSACTIONS
# ==========================================

failed = df[df["status"] == "FAILED"].copy()

if len(failed) == 0:
    print("\nNo failed transactions found.")
    exit()


# ==========================================
# 3. CALCULATE ML RECOVERY PROBABILITY
# ==========================================

model_features = [
    "payment_method",
    "failure_reason",
    "amount",
    "attempt_number",
    "previous_successes",
    "previous_failures",
    "is_high_value",
    "is_first_attempt",
    "customer_success_ratio"
]

missing_features = [
    column
    for column in model_features
    if column not in failed.columns
]

if missing_features:
    print("\nERROR: Missing model features:")
    print(missing_features)
    exit()


failed["recovery_probability"] = (
    model.predict_proba(
        failed[model_features]
    )[:, 1]
)


# ==========================================
# 4. RECOVERY SCORE
# ==========================================

def amount_score(amount):

    if amount >= 40000:
        return 40
    elif amount >= 25000:
        return 30
    elif amount >= 10000:
        return 20
    elif amount >= 5000:
        return 10
    else:
        return 5


def failure_reason_score(reason):

    if reason == "Timeout":
        return 25
    elif reason == "Network Error":
        return 25
    elif reason == "Technical Error":
        return 20
    elif reason == "Insufficient Funds":
        return 5
    else:
        return 10


def retry_score(attempt):

    if attempt == 1:
        return 20
    elif attempt == 2:
        return 10
    else:
        return 0


def customer_history_score(successes, failures):

    if successes >= 10 and failures <= 2:
        return 15
    elif successes >= 5:
        return 10
    elif successes >= 2:
        return 5
    else:
        return 0


failed["amount_score"] = failed["amount"].apply(
    amount_score
)

failed["failure_reason_score"] = (
    failed["failure_reason"]
    .apply(failure_reason_score)
)

failed["retry_score"] = (
    failed["attempt_number"]
    .apply(retry_score)
)

failed["customer_history_score"] = failed.apply(
    lambda row: customer_history_score(
        row["previous_successes"],
        row["previous_failures"]
    ),
    axis=1
)


failed["recovery_score"] = (
    failed["amount_score"]
    + failed["failure_reason_score"]
    + failed["retry_score"]
    + failed["customer_history_score"]
)


# ==========================================
# 5. RECOVERY PRIORITY
# ==========================================

def priority(score):

    if score >= 80:
        return "VERY HIGH"
    elif score >= 60:
        return "HIGH"
    elif score >= 40:
        return "MEDIUM"
    else:
        return "LOW"


failed["recovery_priority"] = (
    failed["recovery_score"]
    .apply(priority)
)


# ==========================================
# 6. DECISION ENGINE
# ==========================================

def make_decision(row):

    probability = row["recovery_probability"]
    reason = row["failure_reason"]
    attempt = row["attempt_number"]
    score = row["recovery_score"]

    # --------------------------------------
    # Insufficient Funds
    # --------------------------------------

    if reason == "Insufficient Funds":

        if probability >= 0.70:
            return "REQUEST_CUSTOMER_ACTION"

        return "DO_NOT_RETRY"


    # --------------------------------------
    # Too many attempts
    # --------------------------------------

    if attempt >= 3:
        return "STOP_AUTOMATIC_RETRY"


    # --------------------------------------
    # High probability recovery
    # --------------------------------------

    if probability >= 0.75 and score >= 60:

        if reason == "Timeout":
            return "SMART_RETRY"

        elif reason == "Network Error":
            return "ALTERNATE_ROUTE"

        elif reason == "Technical Error":
            return "FALLBACK_PROCESSOR"


    # --------------------------------------
    # Medium probability
    # --------------------------------------

    if probability >= 0.50:

        return "RETRY_WITH_DELAY"


    # --------------------------------------
    # Low probability
    # --------------------------------------

    return "MANUAL_REVIEW"


failed["ai_decision"] = failed.apply(
    make_decision,
    axis=1
)


# ==========================================
# 7. EXPECTED REVENUE
# ==========================================

failed["expected_revenue"] = (
    failed["amount"]
    * failed["recovery_probability"]
)


# ==========================================
# 8. SORT BEST OPPORTUNITIES
# ==========================================

failed = failed.sort_values(
    "expected_revenue",
    ascending=False
)


# ==========================================
# 9. DISPLAY RESULTS
# ==========================================

print("\n" + "=" * 60)
print("             TOP RECOVERY OPPORTUNITIES")
print("=" * 60)


columns = [
    "transaction_id",
    "amount",
    "payment_method",
    "failure_reason",
    "attempt_number",
    "recovery_score",
    "recovery_priority",
    "recovery_probability",
    "ai_decision",
    "expected_revenue"
]


display_df = failed[columns].head(20).copy()

display_df["recovery_probability"] = (
    display_df["recovery_probability"] * 100
).round(2)

display_df["expected_revenue"] = (
    display_df["expected_revenue"]
).round(2)


print(
    display_df.to_string(index=False)
)


# ==========================================
# 10. BUSINESS SUMMARY
# ==========================================

total_failed = len(failed)

high_priority = failed[
    failed["recovery_priority"].isin(
        ["VERY HIGH", "HIGH"]
    )
]

potential_revenue = high_priority["amount"].sum()

expected_revenue = high_priority[
    "expected_revenue"
].sum()

average_probability = (
    failed["recovery_probability"].mean() * 100
)


print("\n" + "=" * 60)
print("                  AI SUMMARY")
print("=" * 60)

print(
    f"\nFailed transactions       : {total_failed}"
)

print(
    f"High priority cases      : {len(high_priority)}"
)

print(
    f"Potential revenue        : ₹{potential_revenue:,.2f}"
)

print(
    f"Expected recoverable     : ₹{expected_revenue:,.2f}"
)

print(
    f"Average recovery chance  : {average_probability:.2f}%"
)


# ==========================================
# 11. DECISION DISTRIBUTION
# ==========================================

print("\nAI Decision Distribution:")
print(
    failed["ai_decision"].value_counts()
)


# ==========================================
# 12. SAVE DECISIONS
# ==========================================

output_file = "data/recovery_decisions.csv"

failed.to_csv(
    output_file,
    index=False
)


print("\n" + "=" * 60)
print("DECISION ENGINE COMPLETE")
print("=" * 60)

print(
    f"\nResults saved to:"
    f"\n{output_file}"
)

print("=" * 60)