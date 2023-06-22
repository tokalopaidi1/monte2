import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import lognorm, powerlaw


def vc_monte_carlo_simulation(n_runs, n_vc, vc_failure_rate, vc_min_return, vc_max_return, vc_power_law_exponent):
    vc_investments = np.zeros(n_runs)
    for i in range(n_vc):
        p = np.random.rand(n_runs)
        multipliers = powerlaw.rvs(a=vc_power_law_exponent, scale=(vc_max_return - vc_min_return), size=n_runs) + vc_min_return
        vc_investments += np.where(p >= vc_failure_rate, multipliers, 0)
    return vc_investments


def growth_monte_carlo_simulation(n_runs, n_growth, growth_failure_rate, growth_distribution_mean, growth_distribution_std):
    growth_investments = np.zeros(n_runs)
    for i in range(n_growth):
        p = np.random.rand(n_runs)
        multipliers = lognorm.rvs(s=growth_distribution_std, scale=np.exp(growth_distribution_mean), size=n_runs)
        growth_investments += np.where(p >= growth_failure_rate, multipliers, 0)
    return growth_investments


def main():
    st.title("Monte Carlo Simulation for Portfolio Returns")

    # Sidebar controls
    n_runs = st.sidebar.number_input("Number of simulations:", min_value=100, value=1000, step=10)
    fund = st.sidebar.number_input("Initial Fund:", min_value=100000, value=10000000, step=100000)
    n_investments = st.sidebar.number_input("Number of Investments:", min_value=1, value=20, step=1)

    st.sidebar.subheader("VC Investments")
    vc_failure_rate = st.sidebar.slider("VC Failure Rate:", min_value=0.0, max_value=1.0, value=0.65, step=0.01)
    vc_min_return = st.sidebar.number_input("VC Min Return Multiplier:", min_value=1.0, value=1.0, step=0.1)
    vc_max_return = st.sidebar.number_input("VC Max Return Multiplier:", min_value=1.0, value=200.0, step=0.1)
    vc_power_law_exponent = st.sidebar.slider("VC Power Law Exponent:", min_value=0.0, max_value=10.0, value=2.0, step=0.01)

    st.sidebar.subheader("Growth Investments")
    growth_failure_rate = st.sidebar.slider("Growth Failure Rate:", min_value=0.0, max_value=1.0, value=0.75, step=0.01)
    growth_distribution_mean = st.sidebar.slider("Growth Distribution Mean:", min_value=0.0, max_value=50.0, value=5, step=0.1)
    growth_distribution_std = st.sidebar.slider("Growth Distribution Std Dev:", min_value=0.1, max_value=20.0, value=4.0, step=0.01)

    data = []
    for n_growth in range(n_investments + 1):
        n_vc = n_investments - n_growth
        vc_returns = vc_monte_carlo_simulation(n_runs, n_vc, vc_failure_rate, vc_min_return, vc_max_return, vc_power_law_exponent)
        growth_returns = growth_monte_carlo_simulation(n_runs, n_growth, growth_failure_rate, growth_distribution_mean, growth_distribution_std)

        combined_returns = vc_returns + growth_returns
        combined_returns *= fund / n_investments
        mean_return = np.mean(combined_returns)
        data.append([n_vc, n_growth, mean_return])

    # Creating a DataFrame
    df = pd.DataFrame(data, columns=["VC Investments", "Growth Investments", "Mean Return"])

    # Creating a Histogram
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.histplot(combined_returns, kde=True, ax=ax1, stat='density')
    ax1.set_xlabel("Portfolio Return")
    ax1.set_ylabel("Probability Density")
    ax1.set_title("Portfolio Return Distribution")

    # Creating a Possibility vs LTVI Plot
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(df['VC Investments'], df['Mean Return'], label='VC Investments')
    ax2.plot(df['Growth Investments'], df['Mean Return'], label='Growth Investments')
    ax2.set_xlabel("Number of Investments")
    ax2.set_ylabel("Mean Return")
    ax2.legend()
    ax2.set_title("Possibility vs LTVI")

    st.table(df)
    st.pyplot(fig1)
    st.pyplot(fig2)


if __name__ == "__main__":
    main()
