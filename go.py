import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skewnorm


def generate_vc_deals(n, failure_rate, min_return, max_return, exponent):
    probabilities = np.random.uniform(size=n)
    return np.where(probabilities < failure_rate, 0, np.random.power(a=exponent, size=n) * (max_return - min_return) + min_return)


def generate_growth_deals(n, failure_rate, min_return, max_return, mean, std, skew):
    probabilities = np.random.uniform(size=n)
    returns = skewnorm.rvs(a=skew, loc=mean, scale=std, size=n)
    normalized_returns = (returns - returns.min()) / (returns.max() - returns.min())
    scaled_returns = normalized_returns * (max_return - min_return) + min_return
    return np.where(probabilities < failure_rate, 0, scaled_returns)


def monte_carlo_simulation(n_runs, fund, n_investments, vc_failure_rate, vc_min_return, vc_max_return, vc_power_law_exponent,
                           growth_failure_rate, growth_min_return, growth_max_return, growth_distribution_mean, growth_distribution_std, growth_distribution_skew):
    data = []
    for i in range(n_runs):
        for growth_deals in range(n_investments + 1):
            vc_deals = n_investments - growth_deals
            vc_investment = fund / n_investments * vc_deals
            growth_investment = fund / n_investments * growth_deals
            vc_returns = generate_vc_deals(vc_deals, vc_failure_rate, vc_min_return, vc_max_return, vc_power_law_exponent)
            growth_returns = generate_growth_deals(growth_deals, growth_failure_rate, growth_min_return, growth_max_return,
                                                   growth_distribution_mean, growth_distribution_std, growth_distribution_skew)
            total_return = vc_investment * vc_returns.sum() + growth_investment * growth_returns.sum()
            roi = total_return / fund
            data.append([roi, growth_deals])

    df = pd.DataFrame(data, columns=['roi', 'growth_deals'])
    summary = df.groupby('growth_deals').agg(['mean', 'std', 'min', 'max', 'count'])
    summary.columns = summary.columns.map('_'.join)
    return df, summary


def main():
    # Sidebar controls
    st.sidebar.subheader("General Parameters")
    n_runs = st.sidebar.number_input("Number of Simulations:", min_value=100, max_value=10000, value=1000)
    fund = st.sidebar.number_input("Fund Size ($):", min_value=100000, value=1000000, step=100000)
    n_investments = st.sidebar.number_input("Number of Investments:", min_value=1, max_value=100, value=20)

    st.sidebar.subheader("VC Investments")
    vc_failure_rate = st.sidebar.slider("VC Failure Rate:", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    vc_min_return = st.sidebar.slider("VC Min Return Multiplier:", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    vc_max_return = st.sidebar.slider("VC Max Return Multiplier:", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
    vc_power_law_exponent = st.sidebar.slider("VC Power Law Exponent:", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

    st.sidebar.subheader("Growth Investments")
    growth_failure_rate = st.sidebar.slider("Growth Failure Rate:", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    growth_min_return = st.sidebar.slider("Growth Min Return Multiplier:", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    growth_max_return = st.sidebar.slider("Growth Max Return Multiplier:", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
    growth_distribution_mean = st.sidebar.slider("Growth Distribution Mean:", min_value=0.0, max_value=50.0, value=15.0, step=0.1)
    growth_distribution_std = st.sidebar.slider("Growth Distribution Std Dev:", min_value=0.0, max_value=50.0, value=14.0, step=0.1)
    growth_distribution_skew = st.sidebar.slider("Growth Distribution Skewness:", min_value=-10.0, max_value=10.0, value=7.20, step=0.1)

    # Monte Carlo simulation
    data, summary = monte_carlo_simulation(n_runs, fund, n_investments,
                                           vc_failure_rate, vc_min_return, vc_max_return, vc_power_law_exponent,
                                           growth_failure_rate, growth_min_return, growth_max_return,
                                           growth_distribution_mean, growth_distribution_std, growth_distribution_skew)

    # Summary statistics
    st.subheader("Summary Statistics")
    st.table(summary)

    # ROI histogram
    st.subheader("Return on Investment Histogram")
    fig, ax = plt.subplots()
    bins = np.linspace(0, data['roi'].max(), 50)
    vc_hist = data[data['growth_deals'] == 0]['roi']
    growth_hist = data[data['growth_deals'] == n_investments]['roi']
    
    ax.hist(vc_hist, bins=bins, alpha=0.5, label='VC Only', density=True)
    ax.hist(growth_hist, bins=bins, alpha=0.5, label='Growth Only', density=True)
    ax.set_xlabel('LTVI')
    ax.set_ylabel('Probability')
    ax.legend(loc='upper right')
    
    st.pyplot(fig)

if __name__ == "__main__":
    main()
