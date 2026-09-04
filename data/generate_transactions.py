import random
import pandas as pd
from datetime import datetime, timedelta


# -----------------------------
# Configuration
# -----------------------------

NUM_TRANSACTIONS = 10000

PAYMENT_METHODS = [
    "UPI",
    "CARD",
    "NETBANKING",
    "WALLET"
]

FAILURE_REASONS = [
    "Bank Declined",
    "Network Error",
    "Insufficient Funds",
    "Timeout",
    "Technical Error"
]


# -----------------------------
# Generate Transactions
# -----------------------------

transactions = []

start_time = datetime.now() - timedelta(days=7)


for i in range(NUM_TRANSACTIONS):

    transaction_id = f"TXN{i + 1:06d}"

    customer_id = f"CUST{random.randint(1, 2000):04d}"

    merchant_id = f"MERCHANT{random.randint(1, 10):03d}"

    amount = round(random.uniform(100, 50000), 2)

    payment_method = random.choice(PAYMENT_METHODS)

    timestamp = start_time + timedelta(
        seconds=random.randint(0, 7 * 24 * 60 * 60)
    )

    # Normal payment failure probability
    failure_probability = 0.05

    if random.random() < failure_probability:

        status = "FAILED"

        failure_reason = random.choice(FAILURE_REASONS)

        attempt_number = random.randint(1, 3)

    else:

        status = "SUCCESS"

        failure_reason = None

        attempt_number = 1

    previous_successes = random.randint(0, 20)

    previous_failures = random.randint(0, 5)

    # Basic recovery eligibility rule
    recovery_eligible = (
        status == "FAILED"
        and attempt_number < 3
    )

    transactions.append({
        "transaction_id": transaction_id,
        "customer_id": customer_id,
        "merchant_id": merchant_id,
        "amount": amount,
        "payment_method": payment_method,
        "timestamp": timestamp,
        "status": status,
        "failure_reason": failure_reason,
        "attempt_number": attempt_number,
        "previous_successes": previous_successes,
        "previous_failures": previous_failures,
        "recovery_eligible": recovery_eligible
    })


# -----------------------------
# Convert to DataFrame
# -----------------------------

df = pd.DataFrame(transactions)


# -----------------------------
# Save CSV
# -----------------------------

output_file = "data/transactions.csv"

df.to_csv(output_file, index=False)


# -----------------------------
# Basic Summary
# -----------------------------

print("\n===================================")
print("REVENUE RESCUE - DATA GENERATOR")
print("===================================")

print(f"Total transactions : {len(df)}")

print(
    f"Successful payments: "
    f"{(df['status'] == 'SUCCESS').sum()}"
)

print(
    f"Failed payments    : "
    f"{(df['status'] == 'FAILED').sum()}"
)

print(
    f"Failure rate       : "
    f"{(df['status'] == 'FAILED').mean() * 100:.2f}%"
)

print(
    f"Revenue at risk    : "
    f"₹{df.loc[df['status'] == 'FAILED', 'amount'].sum():,.2f}"
)

print("\nPayment methods:")
print(df["payment_method"].value_counts())

print("\nData saved to:")
print(output_file)