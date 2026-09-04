import pandas as pd

# ==========================================
# REVENUE RESCUE AI
# RECOVERY OPPORTUNITY SCORING
# ==========================================

df = pd.read_csv("data/transactions.csv")

# Only failed transactions need recovery
failed = df[df["status"] == "FAILED"].copy()


# ==========================================
# 1. Amount Score
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


failed["amount_score"] = failed["amount"].apply(amount_score)


# ==========================================
# 2. Failure Reason Score
# ==========================================

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


failed["failure_reason_score"] = (
    failed["failure_reason"]
    .apply(failure_reason_score)
)


# ==========================================
# 3. Retry Score
# ==========================================

def retry_score(attempt):

    if attempt == 1:
        return 20

    elif attempt == 2:
        return 10

    else:
        return 0


failed["retry_score"] = (
    failed["attempt_number"]
    .apply(retry_score)
)


# ==========================================
# 4. Customer History Score
# ==========================================

def customer_history_score(successes, failures):

    if successes >= 10 and failures <= 2:
        return 15

    elif successes >= 5:
        return 10

    elif successes >= 2:
        return 5

    else:
        return 0


failed["customer_history_score"] = failed.apply(
    lambda row: customer_history_score(
        row["previous_successes"],
        row["previous_failures"]
    ),
    axis=1
)


# ==========================================
# 5. Final Recovery Score
# ==========================================

failed["recovery_score"] = (
    failed["amount_score"]
    + failed["failure_reason_score"]
    + failed["retry_score"]
    + failed["customer_history_score"]
)


# ==========================================
# 6. Recovery Priority
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
# 7. Recommended Action
# ==========================================

def recommended_action(row):

    reason = row["failure_reason"]
    attempt = row["attempt_number"]

    if reason == "Insufficient Funds":
        return "Do not retry immediately"

    if reason == "Timeout":
        return "Retry after short delay"

    if reason == "Network Error":
        return "Retry with alternate route"

    if reason == "Technical Error":
        return "Retry with fallback"

    if attempt >= 3:
        return "Stop automatic retries"

    return "Review"


failed["recommended_action"] = failed.apply(
    recommended_action,
    axis=1
)


# ==========================================
# 8. Sort by Recovery Opportunity
# ==========================================

failed = failed.sort_values(
    "recovery_score",
    ascending=False
)


# ==========================================
# 9. Display Results
# ==========================================

print("\n==========================================")
print("       REVENUE RESCUE AI")
print("       RECOVERY OPPORTUNITIES")
print("==========================================")

print("\nTotal failed transactions:", len(failed))

print(
    "\nPriority distribution:"
)

print(
    failed["recovery_priority"]
    .value_counts()
)


print("\nTOP 20 RECOVERY OPPORTUNITIES")
print("------------------------------------------")

columns = [
    "transaction_id",
    "amount",
    "failure_reason",
    "attempt_number",
    "recovery_score",
    "recovery_priority",
    "recommended_action"
]

print(
    failed[columns]
    .head(20)
    .to_string(index=False)
)


# ==========================================
# 10. Potential Revenue
# ==========================================

high_priority = failed[
    failed["recovery_priority"].isin(
        ["VERY HIGH", "HIGH"]
    )
]

potential_revenue = high_priority["amount"].sum()

print("\n==========================================")

print(
    f"High-priority recovery opportunities: "
    f"{len(high_priority)}"
)

print(
    f"Potential revenue to rescue: "
    f"₹{potential_revenue:,.2f}"
)

print("==========================================")