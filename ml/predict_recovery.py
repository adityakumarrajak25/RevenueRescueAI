import pandas as pd
import joblib


# ==========================================
# REVENUE RESCUE AI
# RECOVERY PREDICTION ENGINE
# ==========================================

print("\n==========================================")
print("       REVENUE RESCUE AI")
print("       RECOVERY PREDICTION ENGINE")
print("==========================================")


# ==========================================
# 1. Load trained ML model
# ==========================================

model = joblib.load(
    "ml/recovery_model.joblib"
)

print("\nML model loaded successfully.")


# ==========================================
# 2. Create a sample failed transaction
# ==========================================

transaction = {
    "amount": 18500.00,
    "payment_method": "UPI",
    "failure_reason": "Network Error",
    "attempt_number": 1,
    "previous_successes": 14,
    "previous_failures": 1,
    "is_high_value": 1,
    "is_first_attempt": 1,
    "customer_success_ratio": 14 / 15
}


# Convert transaction into DataFrame

transaction_df = pd.DataFrame(
    [transaction]
)


# ==========================================
# 3. Predict recovery probability
# ==========================================

recovery_probability = model.predict_proba(
    transaction_df
)[0][1]


recovery_percentage = (
    recovery_probability * 100
)


# ==========================================
# 4. Determine priority
# ==========================================

if recovery_probability >= 0.75:

    priority = "HIGH"

elif recovery_probability >= 0.50:

    priority = "MEDIUM"

else:

    priority = "LOW"


# ==========================================
# 5. Determine recommended action
# ==========================================

failure_reason = transaction[
    "failure_reason"
]

payment_method = transaction[
    "payment_method"
]


if failure_reason == "Network Error":

    action = (
        "Retry payment after a short delay"
    )

elif failure_reason == "Timeout":

    action = (
        "Retry payment with exponential backoff"
    )

elif failure_reason == "Insufficient Funds":

    action = (
        "Send payment reminder and suggest "
        "another payment method"
    )

elif failure_reason == "Technical Error":

    action = (
        "Retry using an alternate payment route"
    )

else:

    action = (
        "Review transaction manually"
    )


# ==========================================
# 6. Calculate revenue at risk
# ==========================================

revenue_at_risk = (
    transaction["amount"]
    * recovery_probability
)


# ==========================================
# 7. Display result
# ==========================================

print("\n==========================================")
print("           TRANSACTION ANALYSIS")
print("==========================================")


print(
    f"\nAmount              : "
    f"₹{transaction['amount']:,.2f}"
)

print(
    f"Payment Method      : "
    f"{payment_method}"
)

print(
    f"Failure Reason      : "
    f"{failure_reason}"
)

print(
    f"Attempt Number      : "
    f"{transaction['attempt_number']}"
)

print(
    f"Previous Successes  : "
    f"{transaction['previous_successes']}"
)

print(
    f"Previous Failures   : "
    f"{transaction['previous_failures']}"
)


print("\n------------------------------------------")

print(
    f"Recovery Probability: "
    f"{recovery_percentage:.2f}%"
)

print(
    f"Priority             : "
    f"{priority}"
)

print(
    f"Recommended Action   : "
    f"{action}"
)

print(
    f"Potential Revenue    : "
    f"₹{revenue_at_risk:,.2f}"
)


print("\n==========================================")
print("             DECISION COMPLETE")
print("==========================================")
