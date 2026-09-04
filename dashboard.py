import streamlit as st
import pandas as pd
import joblib

# ==========================================
# REVENUE RESCUE AI - DASHBOARD
# ==========================================

st.set_page_config(
    page_title="Revenue Rescue AI",
    page_icon="💰",
    layout="wide"
)

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():
    df = pd.read_csv("data/transactions.csv")
    return df


@st.cache_resource
def load_model():
    return joblib.load("ml/recovery_model.joblib")


df = load_data()
model = load_model()


# ==========================================
# PREPARE FEATURES
# ==========================================

if "attempt_number" not in df.columns:
    df["attempt_number"] = 1

if "previous_successes" not in df.columns:
    df["previous_successes"] = 0

if "previous_failures" not in df.columns:
    df["previous_failures"] = 0

df["is_high_value"] = (
    df["amount"] >= 10000
).astype(int)

df["is_first_attempt"] = (
    df["attempt_number"] == 1
).astype(int)

total_attempts = (
    df["previous_successes"]
    + df["previous_failures"]
)

df["customer_success_ratio"] = (
    df["previous_successes"]
    / total_attempts.replace(0, 1)
)


# ==========================================
# FAILED TRANSACTIONS
# ==========================================

failed = df[
    df["status"] == "FAILED"
].copy()


# ==========================================
# ML PREDICTION
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

X = failed[model_features]

failed["recovery_probability"] = (
    model.predict_proba(X)[:, 1] * 100
)


# ==========================================
# AI DECISION
# ==========================================

def ai_decision(row):

    reason = row["failure_reason"]
    probability = row["recovery_probability"]
    attempt = row["attempt_number"]

    if reason == "Insufficient Funds":
        return "DO_NOT_RETRY"

    if attempt >= 3:
        return "STOP_AUTOMATIC_RETRY"

    if reason == "Timeout":
        return "RETRY_WITH_DELAY"

    if reason == "Network Error":
        return "ALTERNATE_ROUTE"

    if reason == "Technical Error":
        return "FALLBACK_PROCESSOR"

    if probability >= 70:
        return "SHORT_RETRY"

    if probability >= 40:
        return "MANUAL_REVIEW"

    return "REQUEST_CUSTOMER_ACTION"


failed["ai_decision"] = failed.apply(
    ai_decision,
    axis=1
)
# ==========================================
# AI DECISION ENGINE
# ==========================================

def ai_recovery_decision(row):

    reason = row["failure_reason"]
    probability = row["recovery_probability"]
    attempt = row["attempt_number"]

    # Insufficient funds
    if reason == "Insufficient Funds":
        if probability >= 50:
            return "REQUEST_CUSTOMER_ACTION"
        else:
            return "DO_NOT_RETRY"

    # Timeout
    if reason == "Timeout":
        if probability >= 50:
            return "RETRY_WITH_DELAY"
        else:
            return "STOP_AUTOMATIC_RETRY"

    # Network error
    if reason == "Network Error":
        if probability >= 50:
            return "ALTERNATE_ROUTE"
        else:
            return "STOP_AUTOMATIC_RETRY"

    # Technical error
    if reason == "Technical Error":
        if probability >= 50:
            return "RETRY_WITH_FALLBACK"
        else:
            return "MANUAL_REVIEW"

    # Too many attempts
    if attempt >= 3:
        return "STOP_AUTOMATIC_RETRY"

    return "MANUAL_REVIEW"


failed["ai_decision"] = failed.apply(
    ai_recovery_decision,
    axis=1
)

# ==========================================
# PRIORITY
# ==========================================

def priority(probability):

    if probability >= 80:
        return "VERY HIGH"

    elif probability >= 60:
        return "HIGH"

    elif probability >= 40:
        return "MEDIUM"

    return "LOW"


failed["priority"] = (
    failed["recovery_probability"]
    .apply(priority)
)


# ==========================================
# REVENUE METRICS
# ==========================================

total_failed = len(failed)

potential_revenue = failed["amount"].sum()

expected_revenue = (
    failed["amount"]
    * failed["recovery_probability"]
    / 100
).sum()

average_recovery = (
    failed["recovery_probability"].mean()
)

high_priority = failed[
    failed["priority"].isin(
        ["VERY HIGH", "HIGH"]
    )
]

high_priority_count = len(high_priority)


# ==========================================
# DASHBOARD HEADER
# ==========================================

st.title("💰 Revenue Rescue AI")

st.subheader(
    "AI-Powered Payment Recovery Intelligence"
)

st.write(
    "Identify failed payments, predict recovery probability "
    "and recommend the best recovery action."
)


# ==========================================
# KPI CARDS
# ==========================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Failed Transactions",
        f"{total_failed:,}"
    )

with col2:
    st.metric(
        "Potential Revenue",
        f"₹{potential_revenue:,.0f}"
    )

with col3:
    st.metric(
        "Expected Recoverable",
        f"₹{expected_revenue:,.0f}"
    )

with col4:
    st.metric(
        "Avg Recovery Chance",
        f"{average_recovery:.2f}%"
    )


st.divider()


# ==========================================
# HIGH PRIORITY SECTION
# ==========================================

st.header("🚨 Recovery Opportunities")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "High Priority Cases",
        f"{high_priority_count:,}"
    )

with col2:

    st.metric(
        "High Priority Revenue",
        f"₹{high_priority['amount'].sum():,.0f}"
    )


# ==========================================
# SIDEBAR FILTERS
# ==========================================

st.sidebar.header("🔎 Filters")

priority_filter = st.sidebar.multiselect(
    "Recovery Priority",
    options=[
        "VERY HIGH",
        "HIGH",
        "MEDIUM",
        "LOW"
    ],
    default=[
        "VERY HIGH",
        "HIGH"
    ]
)

if priority_filter:
    filtered = failed[
        failed["priority"].isin(priority_filter)
    ]
else:
    filtered = failed


# ==========================================
# TOP OPPORTUNITIES
# ==========================================

st.header("🎯 Top Recovery Opportunities")

display_columns = [
    "transaction_id",
    "amount",
    "payment_method",
    "failure_reason",
    "attempt_number",
    "recovery_probability",
    "priority",
    "ai_decision"
]

display_df = (
    filtered[
        [
            c for c in display_columns
            if c in filtered.columns
        ]
    ]
    .sort_values(
        "recovery_probability",
        ascending=False
    )
    .head(20)
)

st.dataframe(
    display_df,
   width="stretch",
    hide_index=True
)


# ==========================================
# ANALYTICS
# ==========================================

st.divider()

st.header("📊 Failure Analytics")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Failure Reasons")

    reason_counts = (
        failed["failure_reason"]
        .value_counts()
    )

    st.bar_chart(reason_counts)


with col2:

    st.subheader("AI Recovery Decisions")

    decision_counts = (
        failed["ai_decision"]
        .value_counts()
    )

    st.bar_chart(decision_counts)


# ==========================================
# PAYMENT METHOD ANALYSIS
# ==========================================

st.subheader("💳 Payment Method Failures")

payment_counts = (
    failed["payment_method"]
    .value_counts()
)

st.bar_chart(payment_counts)


# ==========================================
# REVENUE BY FAILURE REASON
# ==========================================

st.subheader("💰 Revenue at Risk by Failure Reason")

revenue_by_reason = (
    failed
    .groupby("failure_reason")["amount"]
    .sum()
    .sort_values(ascending=False)
)

st.bar_chart(revenue_by_reason)



# ==========================================
# AI RECOVERY ACTION CENTER
# ==========================================

st.divider()

st.header("🤖 AI Recovery Action Center")

st.write(
    "Select a failed transaction to see the AI recovery recommendation."
)

# Failed transactions
failed_df = df[df["status"] == "FAILED"].copy()

# Transaction selector
transaction_options = failed_df.apply(
    lambda row: (
        f"{row['transaction_id']} — "
        f"{row['failure_reason']} — "
        f"₹{row['amount']:,.0f}"
    ),
    axis=1
).tolist()

selected_transaction = st.selectbox(
    "Select Failed Transaction",
    transaction_options
)

# Selected transaction

selected_id = selected_transaction.split(" — ")[0]

transaction = failed_df[
    failed_df["transaction_id"].astype(str) == selected_id
].iloc[0]

# ==========================================
# TRANSACTION DETAILS
# ==========================================

st.subheader("Transaction Details")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Transaction Amount",
        f"₹{transaction['amount']:,.2f}"
    )

with col2:
    st.metric(
        "Failure Reason",
        transaction["failure_reason"]
    )

with col3:
    st.metric(
        "Attempt Number",
        transaction["attempt_number"]
    )

with col4:
    st.metric(
        "Payment Method",
        transaction["payment_method"]
    )


# ==========================================
# AI PREDICTION
# ==========================================

st.subheader("🧠 AI Recovery Prediction")

prediction_data = pd.DataFrame([{
    "payment_method": transaction["payment_method"],
    "failure_reason": transaction["failure_reason"],
    "amount": transaction["amount"],
    "attempt_number": transaction["attempt_number"],
    "previous_successes": transaction["previous_successes"],
    "previous_failures": transaction["previous_failures"],
    "is_high_value": transaction["is_high_value"],
    "is_first_attempt": transaction["is_first_attempt"],
    "customer_success_ratio": transaction["customer_success_ratio"]
}])


recovery_probability = model.predict_proba(
    prediction_data
)[0][1]


recovery_percentage = recovery_probability * 100


st.metric(
    "Predicted Recovery Probability",
    f"{recovery_percentage:.1f}%"
)
# ==========================================
# AI CONFIDENCE LEVEL
# ==========================================

if recovery_percentage >= 75:

    confidence_level = "HIGH"
    confidence_message = (
        "Strong recovery candidate. Automated recovery "
        "can be considered."
    )

elif recovery_percentage >= 50:

    confidence_level = "MEDIUM"
    confidence_message = (
        "Moderate recovery opportunity. Use a controlled "
        "recovery strategy."
    )

else:

    confidence_level = "LOW"
    confidence_message = (
        "Low recovery probability. Avoid aggressive "
        "automatic retries."
    )


st.metric(
    "AI Recovery Confidence",
    confidence_level
)

st.info(
    f"💡 {confidence_message}"
)

# ==========================================
# AI RECOMMENDATION
# ==========================================

st.subheader("🎯 Recommended Recovery Action")


if transaction["failure_reason"] == "Insufficient Funds":

    action = "Do not retry immediately"
    explanation = (
        "The payment failed because of insufficient funds. "
        "An immediate retry is unlikely to succeed."
    )

elif transaction["failure_reason"] == "Timeout":

    action = "Retry after short delay"
    explanation = (
        "The failure appears temporary. "
        "A delayed retry can potentially recover the payment."
    )

elif transaction["failure_reason"] == "Network Error":

    action = "Retry with alternate route"
    explanation = (
        "A network-related failure may be recoverable "
        "through an alternate payment route."
    )

elif transaction["failure_reason"] == "Technical Error":

    action = "Retry with fallback"
    explanation = (
        "The transaction encountered a technical failure. "
        "A fallback processing route is recommended."
    )

else:

    action = "Review manually"
    explanation = (
        "The AI recommends reviewing this transaction "
        "before attempting recovery."
    )


st.success(
    f"🤖 AI Recommendation: **{action}**"
)

st.info(
    f"Why: {explanation}"
)
# ==========================================
# REVENUE RECOVERY SIMULATOR
# ==========================================

st.divider()

st.header("💰 Revenue Recovery Simulator")

st.write(
    "Estimate how much failed payment revenue can potentially "
    "be recovered using AI-powered recovery decisions."
)

# ------------------------------------------
# Recovery probability for all failed payments
# ------------------------------------------

failed_simulation = failed_df.copy()

# Prepare ML input
simulation_features = failed_simulation[
    [
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
]

# Predict recovery probabilities
simulation_probabilities = model.predict_proba(
    simulation_features
)[:, 1]

failed_simulation["recovery_probability"] = (
    simulation_probabilities
)

# ------------------------------------------
# Calculate expected recoverable revenue
# ------------------------------------------

failed_simulation["expected_revenue"] = (
    failed_simulation["amount"]
    * failed_simulation["recovery_probability"]
)

total_failed_revenue = (
    failed_simulation["amount"].sum()
)

expected_recovery = (
    failed_simulation["expected_revenue"].sum()
)

recovery_rate = (
    expected_recovery / total_failed_revenue * 100
    if total_failed_revenue > 0
    else 0
)

# ------------------------------------------
# Display metrics
# ------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "💸 Revenue at Risk",
        f"₹{total_failed_revenue:,.0f}"
    )

with col2:
    st.metric(
        "🤖 Expected Recoverable Revenue",
        f"₹{expected_recovery:,.0f}"
    )

with col3:
    st.metric(
        "📈 Expected Recovery Rate",
        f"{recovery_rate:.1f}%"
    )

# ------------------------------------------
# Recovery scenario
# ------------------------------------------

st.subheader("📊 Recovery Scenario")

recovery_target = st.slider(
    "Select recovery effectiveness (%)",
    min_value=10,
    max_value=100,
    value=50,
    step=5
)

simulated_revenue = (
    total_failed_revenue
    * recovery_target
    / 100
)

st.success(
    f"At {recovery_target}% recovery effectiveness, "
    f"Revenue Rescue AI could potentially rescue "
    f"₹{simulated_revenue:,.0f}."
)

# ------------------------------------------
# Comparison
# ------------------------------------------

comparison_df = pd.DataFrame({
    "Category": [
        "Revenue at Risk",
        "Expected AI Recovery"
    ],
    "Amount": [
        total_failed_revenue,
        expected_recovery
    ]
})

st.bar_chart(
    comparison_df.set_index("Category")
)
# ==========================================
# AI DECISION EXPLANATION
# ==========================================

st.divider()

st.header("🔍 Why Did AI Make This Decision?")

st.write(
    "Understand the key factors influencing the AI recovery decision."
)

# Use the currently selected transaction
amount = transaction["amount"]
reason = transaction["failure_reason"]
attempt = transaction["attempt_number"]
successes = transaction["previous_successes"]
failures = transaction["previous_failures"]

# ------------------------------------------
# Identify important factors
# ------------------------------------------

factors = []

if amount >= 25000:
    factors.append(
        f"💰 High transaction value: ₹{amount:,.2f}"
    )

if attempt == 1:
    factors.append(
        "🔄 First retry opportunity — higher recovery potential"
    )
elif attempt == 2:
    factors.append(
        "⚠️ Second attempt — recovery probability reduced"
    )
else:
    factors.append(
        "🛑 Multiple previous attempts — automatic retry risk is high"
    )

if reason in ["Timeout", "Network Error"]:
    factors.append(
        f"🌐 {reason} is generally a temporary failure type"
    )

elif reason == "Technical Error":
    factors.append(
        "⚙️ Technical failure may be recoverable through fallback processing"
    )

elif reason == "Insufficient Funds":
    factors.append(
        "💳 Insufficient funds usually requires customer action"
    )

if successes >= 10 and failures <= 2:
    factors.append(
        "⭐ Strong customer payment history"
    )

elif successes >= 5:
    factors.append(
        "👍 Positive customer payment history"
    )

elif failures > successes:
    factors.append(
        "⚠️ Customer has a relatively high failure history"
    )

# ------------------------------------------
# Display explanation
# ------------------------------------------

st.subheader("Key Decision Factors")

for factor in factors:
    st.write(factor)

# ------------------------------------------
# Final explanation
# ------------------------------------------

st.subheader("🧠 AI Reasoning")

if recovery_percentage >= 75:

    explanation_text = (
        f"The AI predicts a high recovery probability of "
        f"{recovery_percentage:.1f}%. "
        f"The transaction is therefore a strong candidate "
        f"for automated recovery."
    )

elif recovery_percentage >= 50:

    explanation_text = (
        f"The AI predicts a moderate recovery probability of "
        f"{recovery_percentage:.1f}%. "
        f"The transaction may be recoverable, but the system "
        f"should use a controlled recovery strategy."
    )

else:

    explanation_text = (
        f"The AI predicts a relatively low recovery probability "
        f"of {recovery_percentage:.1f}%. "
        f"Automatic recovery may not be economically justified."
    )

st.info(explanation_text)
# ==========================================
# AI RECOVERY STRATEGY OPTIMIZER
# ==========================================

st.divider()

st.header("⚡ AI Recovery Strategy Optimizer")

st.write(
    "The system evaluates the transaction context and "
    "selects the recovery strategy with the highest expected outcome."
)

# ------------------------------------------
# Base recovery probability from ML model
# ------------------------------------------

base_probability = recovery_probability

# ------------------------------------------
# Determine suitable strategies
# ------------------------------------------

strategy_scores = {}

# Default strategies
strategies = [
    "Retry Immediately",
    "Retry After Delay",
    "Alternate Payment Route",
    "Fallback Processor",
    "Request Customer Action"
]

# ------------------------------------------
# Strategy scoring logic
# ------------------------------------------

for strategy in strategies:

    score = base_probability

    # Failure reason based intelligence
    if reason == "Timeout":

        if strategy == "Retry After Delay":
            score *= 1.10

        elif strategy == "Retry Immediately":
            score *= 0.85

        elif strategy == "Fallback Processor":
            score *= 0.95

    elif reason == "Network Error":

        if strategy == "Alternate Payment Route":
            score *= 1.10

        elif strategy == "Fallback Processor":
            score *= 1.05

        elif strategy == "Retry Immediately":
            score *= 0.90

    elif reason == "Technical Error":

        if strategy == "Fallback Processor":
            score *= 1.15

        elif strategy == "Alternate Payment Route":
            score *= 1.05

        elif strategy == "Retry Immediately":
            score *= 0.85

    elif reason == "Insufficient Funds":

        if strategy == "Request Customer Action":
            score *= 1.20

        elif strategy == "Retry Immediately":
            score *= 0.60

        elif strategy == "Retry After Delay":
            score *= 0.80

    elif reason == "Bank Declined":

        if strategy == "Alternate Payment Route":
            score *= 1.10

        elif strategy == "Request Customer Action":
            score *= 1.05

        elif strategy == "Retry Immediately":
            score *= 0.75

    # --------------------------------------
    # Attempt number intelligence
    # --------------------------------------

    if attempt >= 3:

        if strategy == "Retry Immediately":
            score *= 0.70

        elif strategy == "Request Customer Action":
            score *= 1.10

        elif strategy == "Alternate Payment Route":
            score *= 1.05

    # --------------------------------------
    # High-value transaction protection
    # --------------------------------------

    if amount >= 25000:

        if strategy == "Fallback Processor":
            score *= 1.05

        elif strategy == "Alternate Payment Route":
            score *= 1.05

    # --------------------------------------
    # Customer history intelligence
    # --------------------------------------

    if successes >= 5 and failures <= 2:

        if strategy in [
            "Retry After Delay",
            "Fallback Processor"
        ]:
            score *= 1.05

    # Keep probability within 0-100%
    score = min(score, 1.0)

    strategy_scores[strategy] = score


# ------------------------------------------
# Calculate expected revenue
# ------------------------------------------

strategy_results = []

for strategy, probability in strategy_scores.items():

    expected_revenue = amount * probability
# ------------------------------------------
# Recovery Economics
# ------------------------------------------

recovery_cost = 2.0

net_expected_value = (
    expected_revenue - recovery_cost
)
strategy_results.append({
        "Strategy": strategy,
        "Success Probability": probability,
        "Expected Revenue": expected_revenue,
        "net_expected_value":net_expected_value,
    })


strategy_df = pd.DataFrame(strategy_results)


# ------------------------------------------
# Find best strategy
# ------------------------------------------

best_strategy = strategy_df.loc[
    strategy_df["Expected Revenue"].idxmax()
]


# ------------------------------------------
# Display recommendation
# ------------------------------------------

st.subheader("🏆 AI Recommended Strategy")

st.success(
    f"AI recommends: **{best_strategy['Strategy']}**"
)

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Expected Success Probability",
        f"{best_strategy['Success Probability'] * 100:.1f}%"
    )

with col2:

    st.metric(
        "Expected Revenue",
        f"₹{best_strategy['Expected Revenue']:,.2f}"
    )


# ------------------------------------------
# Why this strategy?
# ------------------------------------------

st.subheader("🧠 Why this strategy?")

if reason == "Insufficient Funds":

    st.write(
        "The payment failed because of insufficient funds. "
        "The system prioritizes customer action instead of "
        "repeated automatic retries."
    )

elif reason == "Timeout":

    st.write(
        "The failure appears temporary. "
        "The system prioritizes a delayed retry to avoid "
        "immediately repeating a potentially temporary failure."
    )

elif reason == "Network Error":

    st.write(
        "The failure is network-related. "
        "The system prioritizes an alternate payment route "
        "or fallback processing."
    )

elif reason == "Technical Error":

    st.write(
        "The failure appears technical. "
        "The system prioritizes fallback processing "
        "to improve the chance of successful recovery."
    )

elif reason == "Bank Declined":

    st.write(
        "The payment was declined by the bank. "
        "The system prioritizes an alternate payment route "
        "or customer action rather than repeated immediate retries."
    )

else:

    st.write(
        "The strategy is selected using the predicted recovery "
        "probability, transaction characteristics and previous attempts."
    )


# ------------------------------------------
# Strategy comparison
# ------------------------------------------

st.subheader("📊 Strategy Comparison")

comparison_df = strategy_df.copy()

comparison_df["Success Probability"] = (
    comparison_df["Success Probability"] * 100
).round(1).astype(str) + "%"

comparison_df["Expected Revenue"] = (
    comparison_df["Expected Revenue"]
    .apply(lambda x: f"₹{x:,.2f}")
)

st.dataframe(
    comparison_df,
  width="stretch",
    hide_index=True
)# ==========================================
# EXECUTIVE BUSINESS IMPACT
# ==========================================

st.divider()

st.header("📈 Executive Business Impact")

st.write(
    "Business-level view of the revenue recovery opportunity "
    "identified by Revenue Rescue AI."
)

# ------------------------------------------
# Business calculations
# ------------------------------------------

total_revenue_at_risk = failed["amount"].sum()

total_expected_recovery = (
    failed["amount"]
    * failed["recovery_probability"]
    / 100
).sum()

overall_recovery_rate = (
    total_expected_recovery
    / total_revenue_at_risk
    * 100
    if total_revenue_at_risk > 0
    else 0
)

very_high_cases = len(
    failed[failed["priority"] == "VERY HIGH"]
)

high_cases = len(
    failed[failed["priority"] == "HIGH"]
)

# ------------------------------------------
# KPI cards
# ------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "💸 Revenue at Risk",
        f"₹{total_revenue_at_risk:,.0f}"
    )

with col2:
    st.metric(
        "🤖 Expected Recovery",
        f"₹{total_expected_recovery:,.0f}"
    )

with col3:
    st.metric(
        "📊 Expected Recovery Rate",
        f"{overall_recovery_rate:.1f}%"
    )

with col4:
    st.metric(
        "🚨 Priority Cases",
        f"{very_high_cases + high_cases:,}"
    )

# ------------------------------------------
# Business interpretation
# ------------------------------------------

st.subheader("💡 AI Business Insight")

if overall_recovery_rate >= 60:

    st.success(
        f"Revenue Rescue AI identifies a strong recovery opportunity. "
        f"Approximately {overall_recovery_rate:.1f}% of failed-payment "
        f"value is estimated to be recoverable based on the model's "
        f"predicted probabilities."
    )

elif overall_recovery_rate >= 40:

    st.info(
        f"The system identifies a moderate recovery opportunity. "
        f"The estimated recovery potential is "
        f"{overall_recovery_rate:.1f}% of failed-payment value."
    )

else:

    st.warning(
        f"The estimated recovery opportunity is relatively limited "
        f"at {overall_recovery_rate:.1f}%. "
        f"Recovery actions should therefore be prioritized carefully."
    )

# ------------------------------------------
# Priority distribution
# ------------------------------------------

st.subheader("🎯 Recovery Priority Distribution")

priority_distribution = (
    failed["priority"]
    .value_counts()
    .reindex(
        ["VERY HIGH", "HIGH", "MEDIUM", "LOW"],
        fill_value=0
    )
)

st.bar_chart(priority_distribution)

# ------------------------------------------
# AI action distribution
# ------------------------------------------

st.subheader("🤖 Recommended AI Actions")

action_distribution = (
    failed["ai_decision"]
    .value_counts()
)

st.bar_chart(action_distribution)