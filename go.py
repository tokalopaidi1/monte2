import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import powerlaw


def monte_carlo_simulation(n_runs, fund, n_investments, vc_failure_rate, vc_range1_rate, vc_range2_rate,
                           growth_failure_rate, growth_range1_rate, growth_range2_rate,
                           growth_distribution_mean, growth_distribution_std, vc_power_law_exponent):

    results = []

    for growth_deals in range(n_investments + 1):
        vc_deals = n_investments - growth_deals

        for _ in range(n_runs):
            portfolio_return = 0

            vc_returns = np.zeros(vc_deals)
            vc_outcomes = np.random.random(vc_deals)
            vc_failures = vc_outcomes < vc_failure_rate
            vc_range1 = np.logical_and(~vc_failures, vc_outcomes < (vc_failure_rate + vc_range1_rate))

            for i in range(vc_deals):
                if vc_failures[i]:
                    multiplier = 0
                elif vc_range1[i]:
                    multiplier = np.random.uniform(2, 15)
                else:
                    power_law_dist = powerlaw(a=vc_power_law_exponent, scale=185.0)
                    multiplier = 15 + power_law_dist.rvs()

                investment = (fund / n_investments)
                portfolio_return += investment * multiplier
                vc_returns[i] = multiplier

            growth_returns = np.zeros(growth_deals)
            growth_outcomes = np.random.normal(growth_distribution_mean, growth_distribution_std, growth_deals)
            growth_failures = growth_outcomes < growth_failure_rate
            growth_range1 = np.logical_and(~growth_failures, growth_outcomes < (growth_failure_rate + growth_range1_rate))

            for i in range(growth_deals):
                if growth_failures[i]:
                    multiplier = 0
                elif growth_range1[i]:
                    multiplier = np.random.uniform(2, 7)
                else:
                    multiplier = np.random.uniform(8, 20)

                investment = (fund / n_investments)
                portfolio_return += investment * multiplier
                growth_returns[i] = multiplier

            results.append({'growth_deals': growth_deals, 'portfolio_return': portfolio_return,
                            'vc_returns': vc_returns, 'growth_returns': growth_returns})

    df = pd.DataFrame(results)

    summary = df.groupby('growth_deals').agg(
        mean_return=pd.NamedAgg(column='portfolio_return', aggfunc='mean'),
        max_return=pd.NamedAgg(column='portfolio_return', aggfunc='max'),
        min_return=pd.NamedAgg(column='portfolio_return', aggfunc='min'),
        std_dev=pd.NamedAgg(column='portfolio_return', aggfunc='std'),
        percentile_25=pd.NamedAgg(column='portfolio_return', aggfunc=lambda x: np.percentile(x, 25)),
        median=pd.NamedAgg(column='portfolio_return', aggfunc='median'),
        percentile_75=pd.NamedAgg(column='portfolio_return', aggfunc=lambda x: np.percentile(x, 75)),
        prob_2x=pd.NamedAgg(column='portfolio_return', aggfunc=lambda x: np.mean(x >= 2 * fund)),
        prob_3x=pd.NamedAgg(column='portfolio_return', aggfunc=lambda x: np.mean(x >= 3 * fund)),
        prob_5x=pd.NamedAgg(column='portfolio_return', aggfunc=lambda x: np.mean(x >= 5 * fund)),
        return_distribution=pd.NamedAgg(column='portfolio_return', aggfunc=lambda x: list(x))
    ).reset_index()

    return df, summary


def main():

    st.title('Monte Carlo Simulation App')

    st.sidebar.title('Simulation Parameters')
    n_runs = st.sidebar.number_input('Number of Runs', value=1000, min_value=1)
    fund = st.sidebar.number_input('Initial Fund Amount', value=8000000, min_value=1)
    n_investments = st.sidebar.number_input('Number of Investments', value=25, min_value=1)

    st.sidebar.title('VC Deals')
    vc_failure_rate = st.sidebar.slider('VC Percentage of Failure', 0.0, 1.0, 0.2, step=0.01)
    vc_range1_rate = st.sidebar.slider('VC Percentage for 2x-15x', 0.0, 1.0, 0.5, step=0.01)
    vc_range2_rate = st.sidebar.slider('VC Percentage for 15x-200x', 0.0, 1.0, 0.3, step=0.01)
    vc_power_law_exponent = st.sidebar.slider('Power Law Exponent for VC Deals', 1.0, 5.0, 2.5, step=0.1)

    st.sidebar.title('Growth Deals')
    growth_failure_rate = st.sidebar.slider('Growth Percentage of Failure', 0.0, 1.0, 0.1, step=0.01)
    growth_range1_rate = st.sidebar.slider('Growth Percentage for 2x-7x', 0.0, 1.0, 0.7, step=0.01)
    growth_range2_rate = st.sidebar.slider('Growth Percentage for 8x-20x', 0.0, 1.0, 0.2, step=0.01)

    st.sidebar.title('Growth Deals Distribution')
    growth_distribution_mean = st.sidebar.number_input('Growth Mean', value=1.5)
    growth_distribution_std = st.sidebar.number_input('Growth Standard Deviation', value=0.5)

    df, summary = monte_carlo_simulation(n_runs, fund, n_investments,
                                         vc_failure_rate, vc_range1_rate, vc_range2_rate,
                                         growth_failure_rate, growth_range1_rate, growth_range2_rate,
                                         growth_distribution_mean, growth_distribution_std, vc_power_law_exponent)

    st.header('Simulation Results')
    st.subheader('Raw Data')
    st.dataframe(df)
    st.subheader('Summary Statistics')
    st.dataframe(summary)

    st.header('Portfolio Return Distribution')

    st.subheader('VC Deals')
    vc_chart_data = np.concatenate(df['vc_returns'].values)
    fig_vc, ax_vc = plt.subplots()
    sns.histplot(vc_chart_data, kde=True, ax=ax_vc, bins=np.linspace(0, 200, 50))
    ax_vc.set_xlabel('Return')
    ax_vc.set_ylabel('Frequency')
    st.pyplot(fig_vc)

    st.subheader('Growth Deals')
    growth_chart_data = np.concatenate(df['growth_returns'].values)
    fig_growth, ax_growth = plt.subplots()
    sns.histplot(growth_chart_data, kde=True, ax=ax_growth, bins=np.linspace(0, 20, 50))
    ax_growth.set_xlabel('Return')
    ax_growth.set_ylabel('Frequency')
    st.pyplot(fig_growth)

    st.subheader('Combined')
    fig_combined, ax_combined = plt.subplots()
    sns.histplot(vc_chart_data, kde=True, color='blue', label='VC Deals', ax=ax_combined, bins=np.linspace(0, 200, 50))
    sns.histplot(growth_chart_data, kde=True, color='green', label='Growth Deals', ax=ax_combined, bins=np.linspace(0, 20, 50))
    ax_combined.set_xlabel('Return')
    ax_combined.set_ylabel('Frequency')
    ax_combined.legend()
    st.pyplot(fig_combined)

    st.header('Portfolio Return Distribution per Scenario')
    n_scenarios = summary.shape[0]
    fig, axs = plt.subplots(n_scenarios, figsize=(10, 5*n_scenarios))
    for i, row in summary.iterrows():
        axs[i].hist(row['return_distribution'], bins=np.linspace(0, np.max(row['return_distribution']), 50), alpha=0.7)
        axs[i].set_title(f"Scenario: {row['growth_deals']} Growth Deals")
        axs[i].set_xlabel("Return")
        axs[i].set_ylabel("Frequency")
    plt.tight_layout()
    st.pyplot(fig)


if __name__ == '__main__':
    main()
