import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import powerlaw, skewnorm


@st.cache
def monte_carlo_simulation(n_runs, fund, n_investments, vc_failure_rate, vc_min_return, vc_max_return, vc_power_law_exponent,
                           growth_failure_rate, growth_min_return, growth_max_return, growth_distribution_mean, growth_distribution_std, growth_distribution_skewness):

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
                    multiplier = max(powerlaw.rvs(a=vc_power_law_exponent, scale=1.0), 1.0)
                    vc_investments.append(np.random.uniform(vc_min_return, vc_max_return) * multiplier)

            growth_investments = []
            for _ in range(n_growth):
                p = np.random.rand()
                if p < growth_failure_rate:
                    growth_investments.append(0)
                else:
                    rv = skewnorm.rvs(growth_distribution_skewness, loc=growth_distribution_mean, scale=growth_distribution_std)
                    rv = max(min(rv, growth_max_return), growth_min_return)
                    growth_investments.append(rv)

            total_roi = sum(vc_investments) + sum(growth_investments)
            data.append([n_growth, total_roi])

    df = pd.DataFrame(data, columns=['growth_deals', 'roi'])
    df['roi'] = df['roi'] * fund / n_investments

    summary = df.groupby('growth_deals').roi.agg(['mean', 'std', 'count', 'median', lambda x: x.quantile(0.25),
                                                  lambda x: x.quantile(0.75)]).reset_index()
    summary.columns = ['growth_deals', 'mean_return', 'std_dev', 'count', 'median', 'percentile_25', 'percentile_75']

    return df, summary


def main():
    st.title("Monte Carlo Simulation for Portfolio Returns")

    # Sidebar controls
    n_runs = st.sidebar.number_input("Number of simulations:", min_value=100, value=1000, step=100)
    fund = st.sidebar.number_input("Initial Fund:", min_value=100000, value=100000000, step=100000)
    n_investments = st.sidebar.number_input("Number of Investments:", min_value=1, value=20, step=1)

    st.sidebar.subheader("VC Investments")
    vc_failure_rate = st.sidebar.slider("VC Failure Rate:", min_value=0.0, max_value=1.0, value=0.65, step=0.01)
    vc_min_return = st.sidebar.number_input("VC Min Return Multiplier:", min_value=1.0, value=1.0, step=0.1)
    vc_max_return = st.sidebar.number_input("VC Max Return Multiplier:", min_value=1.0, value=25.0, step=0.1)
    vc_power_law_exponent = st.sidebar.slider("VC Power Law Exponent:", min_value=0.0, max_value=5.0, value=2.0, step=0.1)

    st.sidebar.subheader("Growth Investments")
    growth_failure_rate = st.sidebar.slider("Growth Failure Rate:", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    growth_min_return = st.sidebar.number_input("Growth Min Return Multiplier:", min_value=1.0, value=1.0, step=0.1)
    growth_max_return = st.sidebar.number_input("Growth Max Return Multiplier:", min_value=1.0, value=30.0, step=0.1)
    growth_distribution_mean = st.sidebar.number_input("Growth Distribution Mean:", value=15.0, step=0.1)
    growth_distribution_std = st.sidebar.number_input("Growth Distribution Std Dev:", value=14.0, step=0.1)
    growth_distribution_skewness = st.sidebar.slider("Growth Distribution Skewness:", min_value=0.0, max_value=10.0, value=7.2, step=0.1)

    # Monte Carlo Simulation
    data, summary = monte_carlo_simulation(n_runs, fund, n_investments, vc_failure_rate, vc_min_return, vc_max_return, vc_power_law_exponent,
                                           growth_failure_rate, growth_min_return, growth_max_return, growth_distribution_mean, growth_distribution_std, growth_distribution_skewness)

    # Histogram Plot
    fig, ax = plt.subplots()
    sns.histplot(data[data['growth_deals'] == 0]['roi'], bins=50, kde=True, color='blue', label='VC Only', ax=ax)
    sns.histplot(data[data['growth_deals'] == n_investments]['roi'], bins=50, kde=True, color='green', label='Growth Only', ax=ax)
    ax.set_xlabel('ROI')
    ax.set_ylabel('Frequency')
    ax.legend()
    st.pyplot(fig)

    # Bar Plot
    fig, ax = plt.subplots()
    sns.barplot(x='growth_deals', y='mean_return', data=summary, ax=ax)
    ax.set_xlabel('Number of Growth Deals')
    ax.set_ylabel('Mean Return')
    st.pyplot(fig)

    # Display Summary Table
    st.subheader("Summary Statistics")
    st.table(summary)

    # Display Raw Data
    st.subheader("Raw Data")
    st.write(data)


if __name__ == "__main__":
    main()
