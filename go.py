import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skewnorm


@st.cache(suppress_st_warning=True)
def monte_carlo_simulation(n_runs, fund, n_investments, vc_failure_rate, vc_min_return, vc_max_return,
                           vc_power_law_exponent, growth_failure_rate, growth_min_return, growth_max_return,
                           growth_distribution_mean, growth_distribution_std, growth_skewness):

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
                    multiplier = max(np.random.pareto(vc_power_law_exponent) + 1, 1.0)
                    vc_investments.append(np.random.uniform(vc_min_return, vc_max_return) * multiplier)

            growth_investments = []
            for _ in range(n_growth):
                p = np.random.rand()
                if p < growth_failure_rate:
                    growth_investments.append(0)
                else:
                    # Skewed normal distribution
                    skewed_normal_value = skewnorm.rvs(a=growth_skewness, loc=growth_distribution_mean,
                                                       scale=growth_distribution_std)
                    scaled_value = max(min(skewed_normal_value, growth_max_return), growth_min_return)
                    growth_investments.append(scaled_value)

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
    growth_distribution_mean = st.sidebar.slider("Growth Distribution Mean:", min_value=1.0, max_value=30.0, value=15.0, step=1.0)
    growth_distribution_std = st.sidebar.slider("Growth Distribution Std Dev:", min_value=1.0, max_value=20.0, value=7.0, step=1.0)
    growth_skewness = st.sidebar.slider("Growth Distribution Skewness:", min_value=0.0, max_value=10.0, value=4.0, step=0.1)

    data, summary = monte_carlo_simulation(n_runs, fund, n_investments, vc_failure_rate, vc_min_return, vc_max_return,
                                           vc_power_law_exponent, growth_failure_rate, growth_min_return,
                                           growth_max_return, growth_distribution_mean, growth_distribution_std, growth_skewness)

    # Histogram
    fig, ax = plt.subplots()
    vc_only_data = data[data['growth_deals'] == 0]['roi']
    growth_only_data = data[data['growth_deals'] == n_investments]['roi']
    sns.histplot(vc_only_data, ax=ax, color='blue', label='VC Deals Only', bins=50, kde=True)
    sns.histplot(growth_only_data, ax=ax, color='red', label='Growth Deals Only', bins=50, kde=True)
    ax.set_title('Histogram of Returns for VC Deals Only vs Growth Deals Only')
    ax.set_xlabel('Return on Investment')
    ax.set_ylabel('Frequency')
    ax.legend()
    st.pyplot(fig)

    # Bar plot
    bar_fig, bar_ax = plt.subplots()
    sns.barplot(x='growth_deals', y='mean_return', data=summary, ax=bar_ax)
    bar_ax.set_title('Mean Return vs Number of Growth Deals')
    bar_ax.set_xlabel('Number of Growth Deals')
    bar_ax.set_ylabel('Mean Return')
    st.pyplot(bar_fig)

    # Summary statistics
    st.subheader("Summary Statistics")
    st.table(summary)

    # Raw data
    st.subheader("Raw Data")
    st.dataframe(data)


if __name__ == "__main__":
    main()
