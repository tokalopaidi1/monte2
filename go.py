import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skewnorm, powerlaw


def generate_vc_deals(n, failure_rate, min_return, max_return, exponent):
    probabilities = np.random.uniform(size=n)
    return np.where(probabilities < failure_rate, 0, np.random.power(a=exponent, size=n) * (max_return - min_return) + min_return)


def generate_growth_deals(n, failure_rate, min_return, max_return, mean, std, skew):
    probabilities = np.random.uniform(size=n)
    returns = skewnorm.rvs(a=skew, loc=mean, scale=std, size=n)
    normalized_returns = (returns - returns.min()) / (returns.max() - returns.min()) if returns.size > 0 else np.array([])
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


st.title('Monte Carlo Simulation for VC and Growth Investments')

n_runs = st.sidebar.number_input("Number of Runs:", min_value=1, value=10000)
fund = st.sidebar.number_input("Total Fund Size:", min_value=1.0, value=1000000.0)
n_investments = st.sidebar.number_input("Number of Investments:", min_value=1, value=20)

st.sidebar.markdown("VC Deals")
vc_failure_rate = st.sidebar.slider("VC Failure Rate:", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
vc_min_return = st.sidebar.slider("VC Min Return Multiplier:", min_value=1.0, max_value=10.0, value=1.0, step=0.1)
vc_max_return = st.sidebar.slider("VC Max Return Multiplier:", min_value=1.0, max_value=100.0, value=30.0, step=0.1)
vc_power_law_exponent = st.sidebar.slider("VC Power Law Exponent:", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

st.sidebar.markdown("Growth Deals")
growth_failure_rate = st.sidebar.slider("Growth Failure Rate:", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
growth_min_return = st.sidebar.slider("Growth Min Return Multiplier:", min_value=1.0, max_value=10.0, value=1.0, step=0.1)
growth_max_return = st.sidebar.slider("Growth Max Return Multiplier:", min_value=1.0, max_value=20.0, value=5.0, step=0.1)
growth_distribution_mean = st.sidebar.slider("Growth Distribution Mean:", min_value=1.0, max_value=10.0, value=3.0, step=0.1)
growth_distribution_std = st.sidebar.slider("Growth Distribution Std Dev:", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
growth_distribution_skew = st.sidebar.slider("Growth Distribution Skew:", min_value=-10.0, max_value=10.0, value=5.0, step=0.1)

data, summary = monte_carlo_simulation(n_runs, fund, n_investments, vc_failure_rate, vc_min_return, vc_max_return, vc_power_law_exponent,
                                       growth_failure_rate, growth_min_return, growth_max_return, growth_distribution_mean, growth_distribution_std, growth_distribution_skew)

fig, ax = plt.subplots()
binwidth = (data['roi'].max() - data['roi'].min()) / 100
bins = np.arange(data['roi'].min(), data['roi'].max() + binwidth, binwidth)
sns.histplot(data, x='roi', hue='growth_deals', element='step', stat='probability', common_norm=False, bins=bins, ax=ax)
ax.set_xlabel('LTVI')
ax.set_ylabel('Probability')
st.pyplot(fig)

st.table(summary)
