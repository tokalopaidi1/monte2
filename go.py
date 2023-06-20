import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skewnorm


@st.cache
def monte_carlo_simulation(n_runs, fund, n_investments, vc_failure_rate, vc_min_return, vc_max_return, vc_zipf_s,
                           growth_failure_rate, growth_min_return, growth_max_return, growth_distribution_mean, growth_distribution_std, growth_distribution_skew):

    data = []
    for n_growth in range(n_investments + 1):
        n_vc = n_investments - n_growth
        for _ in range(n_runs):
            vc_investments = []
            for _ in range(n_vc):
                p = np.random.rand()
                if p < vc_failure_rate:
                    vc_investments.append(0)
                else:
                    multiplier = (np.random.zipf(vc_zipf_s) - 1) / (np.exp(1) - 1) * (vc_max_return - vc_min_return) + vc_min_return
                    vc_investments.append(multiplier)

            growth_investments = []
            for _ in range(n_growth):
                p = np.random.rand()
                if p < growth_failure_rate:
                    growth_investments.append(0)
                else:
                    a = growth_distribution_skew
                    rv = skewnorm.rvs(a, loc=growth_distribution_mean, scale=growth_distribution_std)
                    rv = np.clip(rv, growth_min_return, growth_max_return)
                    growth_investments.append(rv)

            total_roi = sum(vc_investments) + sum(growth_investments)
            data.append([n_growth, total_roi])

    df = pd.DataFrame(data, columns=['growth_deals', 'roi'])
    df['roi'] = df['roi'] * fund / n_investments

    summary = df.groupby('growth_deals').roi.agg(['mean', 'std', 'count', 'median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75),
                                                  lambda x: (x >= 2).mean(), lambda x: (x >= 3).mean(), lambda x: (x >= 5).mean(), lambda x: (x >= 8).mean()]).reset_index()
    summary.columns = ['growth_deals', 'mean_return', 'std_dev', 'count', 'median', 'percentile_25', 'percentile_75', 'chance_x2', 'chance_x3', 'chance_x5', 'chance_x8']

    return df, summary


def main():
    st.title("Monte Carlo Simulation for Portfolio Returns")

    # Sidebar controls
    n_runs = st.sidebar.number_input("Number of simulations:", min_value=100, value=1000, step=100)
    fund = st.sidebar.number_input("Initial Fund:", min_value=100000, value=100000000, step=100000)
    n_investments = st.sidebar.number_input("Number of Investments:", min_value=1, value=20, step=1)

    st.sidebar.subheader("VC Investments")
    vc_failure_rate = st.sidebar.slider("VC Failure Rate:", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    vc_min_return = st.sidebar.number_input("VC Min Return Multiplier:", min_value=1.0, value=1.0, step=0.1)
    vc_max_return = st.sidebar.number_input("VC Max Return Multiplier:", min_value=1.0, value=30.0, step=0.1)
    vc_zipf_s = st.sidebar.slider("VC Zipf's distribution parameter (s):", min_value=1.01, max_value=3.0, value=1.5, step=0.01)

    st.sidebar.subheader("Growth Investments")
    growth_failure_rate = st.sidebar.slider("Growth Failure Rate:", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    growth_min_return = st.sidebar.number_input("Growth Min Return Multiplier:", min_value=1.0, value=1.0, step=0.1)
    growth_max_return = st.sidebar.number_input("Growth Max Return Multiplier:", min_value=1.0, value=30.0, step=0.1)
    growth_distribution_mean = st.sidebar.slider("Growth Distribution Mean:", min_value=0.0, max_value=30.0, value=15.0, step=0.1)
    growth_distribution_std = st.sidebar.slider("Growth Distribution Std Dev:", min_value=0.1, max_value=30.0, value=14.0, step=0.1)
    growth_distribution_skew = st.sidebar.slider("Growth Distribution Skewness:", min_value=-10.0, max_value=10.0, value=7.20, step=0.1)

    # Data Generation
    df, summary = monte_carlo_simulation(n_runs, fund, n_investments, vc_failure_rate, vc_min_return, vc_max_return, vc_zipf_s,
                                         growth_failure_rate, growth_min_return, growth_max_return, growth_distribution_mean, growth_distribution_std, growth_distribution_skew)

    # Histogram
    st.subheader("Histogram of Portfolio Returns")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.histplot(df, x='roi', hue='growth_deals', element='step', stat='probability', common_norm=False, kde=True, ax=ax1)
    ax1.set_xlabel("Return on Investment")
    ax1.set_ylabel("Probability")
    ax1.set_title("Growth vs VC Deals")
    st.pyplot(fig1)

    # Additional Histogram for only VC vs only Growth deals
    st.subheader("Histogram of Portfolio Returns (only VC vs only Growth deals)")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.histplot(data=df[df.growth_deals.isin([0, n_investments])], x='roi', hue='growth_deals', element='step', stat='probability', common_norm=False, kde=True, ax=ax2)
    ax2.set_xlabel("Return on Investment")
    ax2.set_ylabel("Probability")
    ax2.set_title("Only VC vs Only Growth Deals")
    st.pyplot(fig2)

    # Summary Table
    st.subheader("Summary Statistics")
    st.table(summary)


if __name__ == "__main__":
    main()
