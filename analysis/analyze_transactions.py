import pandas as pd

# ==========================================
# REVENUE RESCUE AI - TRANSACTION ANALYSIS
# ==========================================

# Load dataset
df = pd.read_csv("data/transactions.csv")

print("\n======================================")
print("      REVENUE RESCUE AI ANALYSIS")
print("======================================")

# ------------------------------------------
# 1. Basic Dataset Information
# ------------------------------------------

print("\n[1] DATASET INFORMATION")
print("--------------------------------------")

print("Total transactions:", len(df))
print("Total columns:", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())


# ------------------------------------------
# 2. Success vs Failure
# ------------------------------------------

print("\n[2] PAYMENT STATUS")
print("--------------------------------------")

status_counts = df["status"].value_counts()

print(status_counts)

failure_rate = (df["status"] == "FAILED").mean() * 100

print(f"\nFailure rate: {failure_rate:.2f}%")


# ------------------------------------------
# 3. Revenue Analysis
# ------------------------------------------

print("\n[3] REVENUE ANALYSIS")
print("--------------------------------------")

total_revenue = df.loc[
    df["status"] == "SUCCESS", "amount"
].sum()

revenue_at_risk = df.loc[
    df["status"] == "FAILED", "amount"
].sum()

print(f"Successful revenue : ₹{total_revenue:,.2f}")
print(f"Revenue at risk    : ₹{revenue_at_risk:,.2f}")


# ------------------------------------------
# 4. Failure Reasons
# ------------------------------------------

print("\n[4] FAILURE REASONS")
print("--------------------------------------")

failure_reasons = df.loc[
    df["status"] == "FAILED",
    "failure_reason"
].value_counts()

print(failure_reasons)


# ------------------------------------------
# 5. Payment Method Performance
# ------------------------------------------

print("\n[5] PAYMENT METHOD PERFORMANCE")
print("--------------------------------------")

method_analysis = (
    df.groupby("payment_method")
    .agg(
        transactions=("transaction_id", "count"),
        failed=("status", lambda x: (x == "FAILED").sum()),
        total_amount=("amount", "sum")
    )
)

method_analysis["failure_rate"] = (
    method_analysis["failed"]
    / method_analysis["transactions"]
    * 100
)

print(method_analysis)


# ------------------------------------------
# 6. Average Transaction Amount
# ------------------------------------------

print("\n[6] TRANSACTION VALUE")
print("--------------------------------------")

print(
    f"Average transaction: "
    f"₹{df['amount'].mean():,.2f}"
)

print(
    f"Average successful transaction: "
    f"₹{df.loc[df['status'] == 'SUCCESS', 'amount'].mean():,.2f}"
)

print(
    f"Average failed transaction: "
    f"₹{df.loc[df['status'] == 'FAILED', 'amount'].mean():,.2f}"
)


# ------------------------------------------
# 7. Failed Transactions by Reason
# ------------------------------------------

print("\n[7] REVENUE AT RISK BY FAILURE REASON")
print("--------------------------------------")

risk_by_reason = (
    df[df["status"] == "FAILED"]
    .groupby("failure_reason")["amount"]
    .agg(["count", "sum"])
    .sort_values("sum", ascending=False)
)

risk_by_reason.columns = [
    "failed_transactions",
    "revenue_at_risk"
]

print(risk_by_reason)


# ------------------------------------------
# 8. Recovery Eligibility
# ------------------------------------------

print("\n[8] RECOVERY ELIGIBILITY")
print("--------------------------------------")

eligible = df["recovery_eligible"].sum()

print("Recovery eligible:", eligible)

print(
    "Recovery eligibility rate:",
    f"{eligible / len(df) * 100:.2f}%"
)


# ------------------------------------------
# 9. High Value Failed Transactions
# ------------------------------------------

print("\n[9] HIGH VALUE FAILED TRANSACTIONS")
print("--------------------------------------")

high_value = df[
    (df["status"] == "FAILED") &
    (df["amount"] >= 25000)
]

print(
    "High-value failed transactions:",
    len(high_value)
)

print(
    f"High-value revenue at risk: "
    f"₹{high_value['amount'].sum():,.2f}"
)


# ------------------------------------------
# 10. Final Business Insight
# ------------------------------------------

print("\n======================================")
print("          BUSINESS INSIGHT")
print("======================================")

print(
    f"\nThere are {len(df):,} transactions "
    f"in the dataset."
)

print(
    f"The payment failure rate is "
    f"{failure_rate:.2f}%."
)

print(
    f"₹{revenue_at_risk:,.2f} is currently "
    f"at risk because of failed payments."
)

print(
    "\nThe next stage is to predict which "
    "failed transactions are most likely "
    "to be successfully recovered."
)

print("\n======================================")