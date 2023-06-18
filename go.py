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

            # VC Deals
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
                    power_law_dist = powerlaw(a=vc_power_law_exponent, scale=15.0)
                    multiplier = max(power_law_dist.rvs(), 15.0)

                investment = (fund / n_investments)
                portfolio_return += investment * multiplier
                vc_returns[i] = multiplier

            # Growth Deals
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

            # Convert portfolio return to multiple of invested fund
            portfolio_return_multiple = portfolio_return / fund
            results.append({'growth_deals': growth_deals, 'portfolio_return_multiple': portfolio_return_multiple,
                            'vc_returns': vc_returns, 'growth_returns': growth_returns})

    df = pd.DataFrame(results)

    summary = df.groupby('growth_deals').agg(
        mean_return=pd.NamedAgg(column='portfolio_return_multiple', aggfunc='mean'),
        return_distribution=pd.NamedAgg(column='portfolio_return_multiple', aggfunc=lambda x: list(x))
    ).reset_index()

    return df, summary


def main():

    st.title('Monte Carlo Simulation App')

    # Sidebar controls
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
    growth_distribution_mean = st.sidebar.number_input('Growth Mean', value=1.5)
    growth_distribution_std = st.sidebar.number_input('Growth Standard Deviation', value=0.5)

    df, summary = monte_carlo_simulation(n_runs, fund, n_investments,
                                         vc_failure_rate, vc_range1_rate, vc_range2_rate,
                                         growth_failure_rate, growth_range1_rate, growth_range2_rate,
                                         growth_distribution_mean, growth_distribution_std, vc_power_law_exponent)

    st.header('Simulation Results')
    st.subheader('Summary Statistics')
    st.dataframe(summary[['growth_deals', 'mean_return']])

    # Combined probability distribution per scenario
    st.header('Portfolio Return Distribution per Scenario (Combined)')
    fig, ax = plt.subplots()
    for i, row in summary.iterrows():
        sns.histplot(row['return_distribution'], kde=True, ax=ax, bins=np.linspace(0, 20, 50), label=f"{row['growth_deals']} Growth Deals", stat="probability")
    ax.set_xlabel('Return (Multiples of Fund)')
    ax.set_ylabel('Probability')
    ax.legend()
    st.pyplot(fig)

    # Probability distribution of expected return from just growth deals vs VC deals
    st.header('Probability Distribution of Expected Return (Growth vs VC Deals)')
    vc_returns = np.concatenate(df['vc_returns'].values) / n_investments
    growth_returns = np.concatenate(df['growth_returns'].values) / n_investments
    fig, ax = plt.subplots()
    sns.histplot(vc_returns, kde=True, color='blue', label='VC Deals', ax=ax, bins=np.linspace(0, 200, 50), stat="probability")
    sns.histplot(growth_returns, kde=True, color='green', label='Growth Deals', ax=ax, bins=np.linspace(0, 20, 50), stat="probability")
    ax.set_xlabel('Return (Multiples of Investment)')
    ax.set_ylabel('Probability')
    ax.legend()
    st.pyplot(fig)


if __name__ == '__main__':
    main()
