
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from pricing_model import (
    generate_flex_value_samples,
    summarize,
    price_ppt_method,
    price_cost_of_capital,
    price_certainty_equivalent_cara,
    price_certainty_equivalent_crra,
    price_revenue_share,
)

st.set_page_config(page_title="Flex Value Pricing (EEM → EDG)", layout="wide")

st.title("Flex Value Pricing Model (EEM → EDG)")
st.caption(
    "Playground to turn a flex value distribution into a transfer price using multiple risk-adjusted pricing methods."
)

# ------------------------
# Sidebar: Data source
# ------------------------
st.sidebar.header("1) Flex Value Distribution Source")

source = st.sidebar.radio(
    "Choose input source",
    ["Synthetic generator", "Upload CSV"],
    index=0,
)

samples = None

if source == "Synthetic generator":
    st.sidebar.subheader("Synthetic generator settings")
    n = st.sidebar.slider("Number of simulations", 5_000, 200_000, 30_000, step=5_000)
    mean = st.sidebar.number_input("Mean flex value (€)", value=50.0, step=1.0)
    vol = st.sidebar.number_input("Volatility / std (€)", value=30.0, step=1.0, min_value=0.1)
    dist = st.sidebar.selectbox("Distribution", ["mixture", "student_t", "normal", "lognormal"], index=0)
    df = st.sidebar.slider("t degrees of freedom (if student_t/mixture)", 3, 30, 6)
    jump_p = st.sidebar.slider("Downside jump probability (mixture)", 0.0, 0.2, 0.03, step=0.005)
    jump_size = st.sidebar.number_input("Downside jump scale (€)", value=80.0, step=5.0, min_value=0.0)
    seed = st.sidebar.number_input("Random seed", value=42, step=1)

    samples = generate_flex_value_samples(
        n=n, mean=mean, vol=vol, dist=dist, df=df,
        downside_jump_prob=jump_p, downside_jump_size=jump_size, seed=int(seed)
    )

else:
    st.sidebar.subheader("Upload CSV")
    st.sidebar.write("CSV must contain a column with flex values in EUR (e.g., `flex_value`).")
    file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    colname = st.sidebar.text_input("Column name", value="flex_value")
    if file is not None:
        dfu = pd.read_csv(file)
        if colname not in dfu.columns:
            st.error(f"Column '{colname}' not found. Available: {list(dfu.columns)}")
        else:
            samples = dfu[colname].astype(float).to_numpy()

if samples is None:
    st.stop()

samples = samples[np.isfinite(samples)]
if samples.size < 1000:
    st.error("Need at least 1,000 valid samples.")
    st.stop()

# ------------------------
# Main: Summary + charts
# ------------------------
colA, colB = st.columns([1.2, 1.0], gap="large")

with colA:
    st.subheader("Distribution summary")
    summ = summarize(samples)
    summ_df = pd.DataFrame([summ]).T.rename(columns={0: "value"})
    st.dataframe(summ_df, use_container_width=True)

with colB:
    st.subheader("Flex value histogram")
    fig = plt.figure()
    plt.hist(samples, bins=80)
    plt.axvline(np.mean(samples))
    plt.xlabel("Flex value (€)")
    plt.ylabel("Count")
    st.pyplot(fig, clear_figure=True)

st.divider()

# ------------------------
# Pricing methods
# ------------------------
st.header("2) Pricing methods")

method = st.selectbox(
    "Choose method",
    [
        "PPT method (Mean - scaled 95% gap × div × hurdle)",
        "Cost-of-Capital on Economic Capital (VaR/ES)",
        "Certainty Equivalent (CARA utility)",
        "Certainty Equivalent (CRRA utility)",
        "Revenue share with floor/cap (expected payout)",
    ],
    index=0,
)

result = None

if method.startswith("PPT"):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        qlvl = st.slider("Quantile level (e.g., 0.95 means use P5 downside)", 0.80, 0.995, 0.95, step=0.005)
    with c2:
        scale = st.number_input("Scaling factor", value=3.0, step=0.1, min_value=0.0)
    with c3:
        div = st.number_input("Diversification factor", value=0.8, step=0.05, min_value=0.0)
    with c4:
        hurdle = st.number_input("Hurdle rate", value=0.1888, step=0.01, min_value=0.0, format="%.4f")

    result = price_ppt_method(
        samples,
        quantile_level=float(qlvl),
        scaling_factor=float(scale),
        diversification_factor=float(div),
        hurdle_rate=float(hurdle),
    )

elif method.startswith("Cost-of-Capital"):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        alpha = st.slider("Downside alpha (e.g., 0.05 = 5% tail)", 0.001, 0.20, 0.05, step=0.001)
    with c2:
        measure = st.selectbox("Risk measure", ["es", "var"], index=0)
    with c3:
        div = st.number_input("Diversification factor", value=0.8, step=0.05, min_value=0.0)
    with c4:
        hurdle = st.number_input("Hurdle rate", value=0.1888, step=0.01, min_value=0.0, format="%.4f")

    result = price_cost_of_capital(
        samples,
        alpha=float(alpha),
        measure=measure,
        hurdle_rate=float(hurdle),
        diversification_factor=float(div),
        side="downside",
    )

elif method.startswith("Certainty Equivalent (CARA"):
    c1, c2 = st.columns(2)
    with c1:
        a = st.slider("Risk aversion a (higher = more conservative)", 0.0001, 0.50, 0.05, step=0.0001)
    with c2:
        st.write("Rule of thumb: choose a so that exp(-a·€) reacts at the scale you care about.")
    result = price_certainty_equivalent_cara(samples, risk_aversion=float(a))

elif method.startswith("Certainty Equivalent (CRRA"):
    c1, c2, c3 = st.columns(3)
    with c1:
        gamma = st.slider("Risk aversion γ", 0.1, 10.0, 2.0, step=0.1)
    with c2:
        shift = st.number_input("Shift (€) to ensure positivity", value=200.0, step=10.0, min_value=0.0)
    with c3:
        st.write("CRRA requires x + shift > 0. Increase shift if you get an error.")
    try:
        result = price_certainty_equivalent_crra(samples, gamma=float(gamma), shift=float(shift))
    except Exception as e:
        st.error(str(e))
        st.stop()

else:  # revenue share
    c1, c2, c3 = st.columns(3)
    with c1:
        share = st.slider("Share to EDG", 0.0, 1.0, 0.7, step=0.01)
    with c2:
        floor = st.number_input("Floor payout (€)", value=0.0, step=1.0)
    with c3:
        cap_on = st.checkbox("Apply cap?", value=False)
        cap = None
        if cap_on:
            cap = st.number_input("Cap payout (€)", value=80.0, step=1.0)

    result = price_revenue_share(samples, share_to_edg=float(share), floor=float(floor), cap=cap)

# ------------------------
# Results display
# ------------------------
st.subheader("Result")
res_df = pd.DataFrame([result]).T.rename(columns={0: "value"})
st.dataframe(res_df, use_container_width=True)

price = float(result.get("price", np.nan))
st.metric("Transfer price to EDG (€)", f"{price:,.2f}")

# ------------------------
# Sensitivity
# ------------------------
st.divider()
st.header("3) Sensitivity (one parameter)")

if method.startswith("PPT"):
    param = st.selectbox("Vary parameter", ["scaling_factor", "diversification_factor", "hurdle_rate"], index=0)
    grid = np.linspace(0.0, 6.0, 41) if param == "scaling_factor" else np.linspace(0.0, 1.2, 41) if param == "diversification_factor" else np.linspace(0.0, 0.35, 41)
    prices = []
    for v in grid:
        kw = dict(quantile_level=float(qlvl), scaling_factor=float(scale), diversification_factor=float(div), hurdle_rate=float(hurdle))
        kw[param] = float(v)
        r = price_ppt_method(samples, **kw)
        prices.append(r["price"])
    fig = plt.figure()
    plt.plot(grid, prices)
    plt.xlabel(param)
    plt.ylabel("Price (€)")
    st.pyplot(fig, clear_figure=True)

elif method.startswith("Cost-of-Capital"):
    param = st.selectbox("Vary parameter", ["alpha", "diversification_factor", "hurdle_rate"], index=0)
    grid = np.linspace(0.001, 0.20, 40) if param == "alpha" else np.linspace(0.0, 1.2, 41) if param == "diversification_factor" else np.linspace(0.0, 0.35, 41)
    prices = []
    for v in grid:
        kw = dict(alpha=float(alpha), measure=measure, diversification_factor=float(div), hurdle_rate=float(hurdle))
        kw[param] = float(v)
        r = price_cost_of_capital(samples, **kw)
        prices.append(r["price"])
    fig = plt.figure()
    plt.plot(grid, prices)
    plt.xlabel(param)
    plt.ylabel("Price (€)")
    st.pyplot(fig, clear_figure=True)

else:
    st.info("Sensitivity is most meaningful for PPT / CoC methods. Switch to those to see parameter sensitivity.")

# ------------------------
# Export
# ------------------------
st.divider()
st.header("4) Export")

export = pd.DataFrame({"flex_value": samples})
buf = io.BytesIO()
export.to_csv(buf, index=False)
st.download_button("Download simulated flex values CSV", data=buf.getvalue(), file_name="flex_values.csv", mime="text/csv")
