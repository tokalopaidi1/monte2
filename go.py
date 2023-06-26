import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mode


@st.cache
def monte_carlo_simulation(n_runs, fund, n_investments, vc_failure_rate, vc_min_return, vc_max_return, vc_power_law_exponent,
                           growth_failure_rate, growth_lognorm_mean, growth_lognorm_std):

    data = []
    for n_growth in range(n_investments + 1):
        n_vc = n_investments - n_growth
        for _ in range(n_runs):
            vc_investments = 0
            growth_investments = 0

            for _ in range(n_vc):
                p = np.random.rand()
                if p >= vc_failure_rate:
                    multiplier = max(np.random.power(a=vc_power_law_exponent), 1.0)
                    vc_investments += np.random.uniform(vc_min_return, vc_max_return) * multiplier

            for _ in range(n_growth):
                p = np.random.rand()
                if p >= growth_failure_rate:
                    growth_investments += np.random.lognormal(mean=growth_lognorm_mean, sigma=growth_lognorm_std)

            total_roi = vc_investments + growth_investments
            data.append([n_growth, total_roi])

    df = pd.DataFrame(data, columns=['growth_deals', 'roi'])
    df['roi'] = df['roi'] * fund / n_investments

    summary = df.groupby('growth_deals').roi.agg(['mean', 'std', 'count', 'median', lambda x: x.quantile(0.25),
                                                  lambda x: x.quantile(0.75)]).reset_index()
    summary.columns = ['growth_deals', 'mean_return', 'std_dev', 'count', 'median', 'percentile_25', 'percentile_75']

    # Calculate mode separately and add to summary
    mode_values = df.groupby('growth_deals')['roi'].apply(lambda x: mode(x)[0][0]).reset_index(name='mode')
    summary = pd.merge(summary, mode_values, on='growth_deals')

    # Calculate Sharpe Ratio and add to summary
    summary['sharpe_ratio'] = summary['mean_return'] / summary['std_dev']

    return df, summary


def main():
    st.title("Monte Carlo Simulation for Portfolio Returns")

    # Sidebar controls
    n_runs = st.sidebar.number_input("Number of simulations:", min_value=100, value=1000, step=100)
    fund = st.sidebar.number_input("Initial Fund:", min_value=100000, value=8000000, step=100000)
    n_investments = st.sidebar.number_input("Number of Investments:", min_value=1, value=20, step=1)

    st.sidebar.subheader("VC Investments")
    vc_failure_rate = st.sidebar.slider("VC Failure Rate:", min_value=0.0, max_value=1.0, value=0.85, step=0.01)
    vc_min_return = st.sidebar.number_input("VC Min Return Multiplier:", min_value=1.0, value=1.0, step=0.1)
    vc_max_return = st.sidebar.number_input("VC Max Return Multiplier:", min_value=1.0, value=300.0, step=0.1)
    vc_power_law_exponent = st.sidebar.slider("VC Power Law Exponent:", min_value=0.1, max_value=10.0, value=1.88, step=0.01)

    st.sidebar.subheader("Growth Investments")
    growth_failure_rate = st.sidebar.slider("Growth Failure Rate:", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
    growth_lognorm_mean = st.sidebar.slider("Growth Log-Normal Mean (μ of log):", min_value=0.0, max_value=10.0, value=2.83, step=0.01)
    growth_lognorm_std = st.sidebar.slider("Growth Log-Normal Std Dev (σ of log):", min_value=0.1, max_value=10.0, value=1.11, step=0.01)

    data, summary = monte_carlo_simulation(n_runs, fund, n_investments,
                                           vc_failure_rate, vc_min_return, vc_max_return, vc_power_law_exponent,
                                           growth_failure_rate, growth_lognorm_mean, growth_lognorm_std)

    # Mean ROI plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(summary.growth_deals, summary.mean_return, label='Mean ROI', color='blue')
    ax.plot(summary.growth_deals, summary.percentile_25, label='25th Percentile', color='red')
    ax.plot(summary.growth_deals, summary.percentile_75, label='75th Percentile', color='green')
    ax.axhline(y=fund * 2, color='gray', linestyle='dashed')
    ax.axhline(y=fund * 3, color='gray', linestyle='dashed')
    ax.axhline(y=fund * 5, color='green', linestyle='dashed')
    ax.set_title('Monte Carlo Simulation of Portfolio Returns')
    ax.set_xlabel('Number of Growth Investments')
    ax.set_ylabel('Mean Return on Investment')
    ax.legend(['Mean ROI', '25th Percentile', '75th Percentile', '2x Fund', '3x Fund', '5x Fund'])
    st.pyplot(fig)

    # Distribution of ROI for a fixed number of growth deals
    fixed_growth_deals = st.slider("Number of Growth Deals for Distribution Plot:", min_value=0, max_value=n_investments, value=int(n_investments / 2), step=1)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.histplot(data[data['growth_deals'] == fixed_growth_deals]['roi'], kde=True, ax=ax2)
    ax2.set_title(f'Distribution of ROI for {fixed_growth_deals} Growth Deals')
    ax2.set_xlabel('Return on Investment')
    ax2.set_ylabel('Frequency')
    st.pyplot(fig2)

    # Histogram with KDE
    fig3, ax3 = plt.subplots()
    vc_only_data = data[data['growth_deals'] == 0]['roi']
    growth_only_data = data[data['growth_deals'] == n_investments]['roi']
    sns.histplot(vc_only_data, bins=50, color='blue', label='VC Deals', ax=ax3, stat='density', kde=True)
    sns.histplot(growth_only_data, bins=50, color='green', label='Growth Deals', ax=ax3, stat='density', kde=True)
    ax3.set_xlabel('TVPI')
    ax3.set_ylabel('Density')
    ax3.legend()
    st.pyplot(fig3)

    # Sharpe Ratio vs. % Growth Deals
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    pct_growth_deals = (summary.growth_deals / n_investments) * 100
    ax4.plot(pct_growth_deals, summary.sharpe_ratio, label='Sharpe Ratio', color='purple')
    ax4.set_title('Sharpe Ratio vs. Percentage of Growth Deals')
    ax4.set_xlabel('Percentage of Growth Deals in Portfolio (%)')
    ax4.set_ylabel('Sharpe Ratio')
    ax4.legend(['Sharpe Ratio'])
    st.pyplot(fig4)

    # CDF Plot
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    sns.ecdfplot(data=data['roi'], ax=ax5)
    ax5.set_title('Cumulative Distribution Function of ROI')
    ax5.set_xlabel('Return on Investment')
    ax5.set_ylabel('Cumulative Probability')
    st.pyplot(fig5)

    # Scatter Plot of Top Quartile Returns vs. Number of Growth Deals
    fig6, ax6 = plt.subplots(figsize=(10, 5))
    top_quartile = data.groupby('growth_deals')['roi'].apply(lambda x: x.quantile(0.75)).reset_index(name='top_quartile')
    ax6.scatter(top_quartile['growth_deals'], top_quartile['top_quartile'], color='orange')
    ax6.set_title('Top Quartile Returns vs. Number of Growth Deals')
    ax6.set_xlabel('Number of Growth Deals')
    ax6.set_ylabel('Top Quartile Returns')
    st.pyplot(fig6)

    # Scatter Plot of Sharpe Ratio (Point-wise) vs. Number of Growth Deals
    fig7, ax7 = plt.subplots(figsize=(10, 5))
    std_dev = data.groupby('growth_deals')['roi'].std().reset_index(name='std_dev')
    data = pd.merge(data, std_dev, on='growth_deals')
    data['point_wise_sharpe_ratio'] = data['roi'] / data['std_dev']
    ax7.scatter(data['growth_deals'], data['point_wise_sharpe_ratio'], color='magenta')
    ax7.set_title('Sharpe Ratio (Point-wise) vs. Number of Growth Deals')
    ax7.set_xlabel('Number of Growth Deals')
    ax7.set_ylabel('Sharpe Ratio (Point-wise)')
    st.pyplot(fig7) 

    # Summary statistics
    st.subheader("Summary Statistics")
    st.table(summary)

    # Raw data
    st.subheader("Raw Data")
    st.write(data)


if __name__ == "__main__":
    main()
