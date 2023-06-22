import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


@st.cache
def monte_carlo_simulation_vc(n_runs, fund, n_investments, failure_rate, min_return, max_return, power_law_exponent):
    data = []
    for _ in range(n_runs):
        vc_investments = 0
        for _ in range(n_investments):
            p = np.random.rand()
            if p >= failure_rate:
                multiplier = max(np.random.power(a=power_law_exponent), 1.0)
                vc_investments += np.random.uniform(min_return, max_return) * multiplier
        total_roi = vc_investments
        data.append(total_roi)

    return pd.Series(data) * fund / n_investments


@st.cache
def monte_carlo_simulation_growth(n_runs, fund, n_investments, failure_rate, lognorm_mean, lognorm_std):
    data = []
    for _ in range(n_runs):
        growth_investments = 0
        for _ in range(n_investments):
            p = np.random.rand()
            if p >= failure_rate:
                growth_investments += np.random.lognormal(mean=lognorm_mean, sigma=lognorm_std)
        total_roi = growth_investments
        data.append(total_roi)

    return pd.Series(data) * fund / n_investments


def main():
    st.title("Monte Carlo Simulation for Portfolio Returns")

    # Sidebar controls
    n_runs = st.sidebar.number_input("Number of simulations:", min_value=100, value=1000, step=100)
    fund = st.sidebar.number_input("Initial Fund:", min_value=100000, value=100000000, step=100000)
    
    st.sidebar.subheader("VC Investments")
    n_investments_vc = st.sidebar.number_input("Number of VC Investments:", min_value=1, value=10, step=1)
    vc_failure_rate = st.sidebar.slider("VC Failure Rate:", min_value=0.0, max_value=1.0, value=0.65, step=0.01)
    vc_min_return = st.sidebar.number_input("VC Min Return Multiplier:", min_value=1.0, value=1.0, step=0.1)
    vc_max_return = st.sidebar.number_input("VC Max Return Multiplier:", min_value=1.0, value=25.0, step=0.1)
    vc_power_law_exponent = st.sidebar.slider("VC Power Law Exponent:", min_value=0.0, max_value=5.0, value=2.0, step=0.1)

    st.sidebar.subheader("Growth Investments")
    n_investments_growth = st.sidebar.number_input("Number of Growth Investments:", min_value=1, value=10, step=1)
    growth_failure_rate = st.sidebar.slider("Growth Failure Rate:", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    growth_lognorm_mean = st.sidebar.slider("Growth Log-Normal Mean (μ of log):", min_value=0.0, max_value=5.0, value=2.0, step=0.1)
    growth_lognorm_std = st.sidebar.slider("Growth Log-Normal Std Dev (σ of log):", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
    
    vc_data = monte_carlo_simulation_vc(n_runs, fund, n_investments_vc, vc_failure_rate, vc_min_return, vc_max_return, vc_power_law_exponent)
    growth_data = monte_carlo_simulation_growth(n_runs, fund, n_investments_growth, growth_failure_rate, growth_lognorm_mean, growth_lognorm_std)
    
    # Histogram
    fig, ax = plt.subplots()
    sns.histplot(vc_data, bins=50, color='blue', label='VC Deals', ax=ax, stat='density', kde=True)
    sns.histplot(growth_data, bins=50, color='green', label='Growth Deals', ax=ax, stat='density', kde=True)
    ax.set_xlabel('TVPI')
    ax.set_ylabel('Density')
    ax.legend()
    st.pyplot(fig)

    # Summary statistics
    st.subheader("Summary Statistics")
    st.subheader("VC Deals")
    st.table(vc_data.describe())

    st.subheader("Growth Deals")
    st.table(growth_data.describe())


if __name__ == "__main__":
    main()
