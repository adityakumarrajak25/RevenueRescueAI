import joblib
import pandas as pd


# ==========================================
# REVENUE RESCUE AI
# RECOVERY DECISION ENGINE
# ==========================================


class RecoveryDecisionEngine:

    def __init__(self):

        self.model = joblib.load(
            "ml/recovery_model.joblib"
        )


    # ======================================
    # ML Prediction
    # ======================================

    def predict_recovery(self, transaction):

        df = pd.DataFrame(
            [transaction]
        )

        probability = self.model.predict_proba(
            df
        )[0][1]

        return probability


    # ======================================
    # Business Decision
    # ======================================

    def make_decision(self, transaction):

        probability = self.predict_recovery(
            transaction
        )

        amount = transaction["amount"]

        failure_reason = transaction[
            "failure_reason"
        ]

        attempt_number = transaction[
            "attempt_number"
        ]

        previous_successes = transaction[
            "previous_successes"
        ]

        previous_failures = transaction[
            "previous_failures"
        ]


        # ----------------------------------
        # Customer reliability
        # ----------------------------------

        total_attempts = (
            previous_successes
            + previous_failures
        )

        if total_attempts > 0:

            customer_success_rate = (
                previous_successes
                / total_attempts
            )

        else:

            customer_success_rate = 0


        # ==================================
        # DECISION RULES
        # ==================================


        # Rule 1:
        # Too many retries → stop

        if attempt_number >= 3:

            action = "STOP_RETRY"

            reason = (
                "Maximum retry attempts reached"
            )

            priority = "LOW"


        # Rule 2:
        # Insufficient funds

        elif failure_reason == (
            "Insufficient Funds"
        ):

            action = "PAYMENT_REMINDER"

            reason = (
                "Customer may need to "
                "add funds or change payment method"
            )

            priority = "MEDIUM"


        # Rule 3:
        # High recovery probability

        elif probability >= 0.75:

            action = "SMART_RETRY"

            reason = (
                "High probability of successful "
                "recovery"
            )

            priority = "HIGH"


        # Rule 4:
        # Medium probability

        elif probability >= 0.50:

            action = "ALTERNATE_METHOD"

            reason = (
                "Moderate recovery probability; "
                "alternate payment route recommended"
            )

            priority = "MEDIUM"


        # Rule 5:
        # Low probability

        else:

            action = "DO_NOT_RETRY"

            reason = (
                "Low probability of successful recovery"
            )

            priority = "LOW"


        # ==================================
        # Revenue calculation
        # ==================================

        expected_revenue = (
            amount * probability
        )


        # ==================================
        # Return decision
        # ==================================

        return {

            "recovery_probability":
                round(
                    probability * 100,
                    2
                ),

            "priority":
                priority,

            "action":
                action,

            "reason":
                reason,

            "transaction_amount":
                amount,

            "expected_revenue":
                round(
                    expected_revenue,
                    2
                ),

            "customer_success_rate":
                round(
                    customer_success_rate * 100,
                    2
                )
        }


# ==========================================
# TEST THE DECISION ENGINE
# ==========================================


if __name__ == "__main__":

    engine = RecoveryDecisionEngine()


    transaction = {

        "amount": 18500,

        "payment_method":
            "UPI",

        "failure_reason":
            "Network Error",

        "attempt_number":
            1,

        "previous_successes":
            14,

        "previous_failures":
            1,

        "is_high_value":
            1,

        "is_first_attempt":
            1,

        "customer_success_ratio":
            14 / 15
    }


    result = engine.make_decision(
        transaction
    )


    print("\n")
    print("=" * 55)

    print(
        "        REVENUE RESCUE AI"
    )

    print(
        "        DECISION ENGINE"
    )

    print("=" * 55)


    print(
        f"\nTransaction Amount : "
        f"₹{result['transaction_amount']:,.2f}"
    )

    print(
        f"Recovery Probability: "
        f"{result['recovery_probability']}%"
    )

    print(
        f"Customer Success Rate: "
        f"{result['customer_success_rate']}%"
    )

    print(
        f"Priority            : "
        f"{result['priority']}"
    )

    print(
        f"Decision            : "
        f"{result['action']}"
    )

    print(
        f"Reason              : "
        f"{result['reason']}"
    )

    print(
        f"Expected Revenue    : "
        f"₹{result['expected_revenue']:,.2f}"
    )


    print("\n" + "=" * 55)
    