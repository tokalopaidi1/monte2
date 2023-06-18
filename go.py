import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import powerlaw


def monte_carlo_simulation(n_runs, fund, n_investments, vc_failure_rate, vc_range1_min, vc_range1_max,
                           vc_range2_min, vc_range2_max, growth_failure_rate, growth_range1_min,
                           growth_range1_max, growth_range2_min, growth_range2_max,
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
            vc_range2 = np.logical_and(~vc_range1, vc_outcomes < (vc_failure_rate + vc_range1_rate + vc_range2_rate))

            for i in range(vc_deals):
                if vc_failures[i]:
                    multiplier = 0
                elif vc_range1[i]:
                    multiplier = np.random.uniform(vc_range1_min, vc_range1_max)
                elif vc_range2[i]:
                    multiplier = np.random.uniform(vc_range2_min, vc_range2_max)
                else:
                    power_law_dist = powerlaw(a=vc_power_law_exponent, scale=15.0)
                    multiplier = power_law_dist.rvs()

                investment = (fund / n_investments)
                portfolio_return += investment * multiplier
                vc_returns[i] = multiplier

            growth_returns = np.zeros(growth_deals)
            growth_outcomes = np.random.normal(growth_distribution_mean, growth_distribution_std, growth_deals)
            growth_failures = growth_outcomes < growth_failure_rate
            growth_range1 = np.logical_and(~growth_failures, growth_outcomes < (growth_failure_rate + growth_range1_rate))
            growth_range2 = np.logical_and(~growth_range1, growth_outcomes < (growth_failure_rate + growth_range1_rate + growth_range2_rate))

            for i in range(growth_deals):
                if growth_failures[i]:
                    multiplier = 0
                elif growth_range1[i]:
                    multiplier = np.random.uniform(growth_range1_min, growth_range1_max)
                elif growth_range2[i]:
                    multiplier = np.random.uniform(growth_range2_min, growth_range2_max)
                else:
                    multiplier = np.random.normal(loc=growth_distribution_mean, scale=growth_distribution_std)

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
        prob_5x=pd.NamedAgg(column='portfolio_return', aggfunc=lambda x: np.mean(x >= 5 * fund))
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
    vc_range1_min = st.sidebar.number_input('VC Range1 Minimum', value=2.0)
    vc_range1_max = st.sidebar.number_input('VC Range1 Maximum', value=15.0)
    vc_range2_min = st.sidebar.number_input('VC Range2 Minimum', value=15.0)
    vc_range2_max = st.sidebar.number_input('VC Range2 Maximum', value=200.0)
    vc_power_law_exponent = st.sidebar.slider('Power Law Exponent for VC Deals', 1.0, 5.0, 2.5, step=0.1)

    st.sidebar.title('Growth Deals')
    growth_failure_rate = st.sidebar.slider('Growth Percentage of Failure', 0.0, 1.0, 0.1, step=0.01)
    growth_range1_min = st.sidebar.number_input('Growth Range1 Minimum', value=1.0)
    growth_range1_max = st.sidebar.number_input('Growth Range1 Maximum', value=3.0)
    growth_range2_min = st.sidebar.number_input('Growth Range2 Minimum', value=3.0)
    growth_range2_max = st.sidebar.number_input('Growth Range2 Maximum', value=20.0)

    df, summary = monte_carlo_simulation(n_runs, fund, n_investments,
                                         vc_failure_rate, vc_range1_min, vc_range1_max, vc_range2_min, vc_range2_max,
                                         growth_failure_rate, growth_range1_min, growth_range1_max,
                                         growth_range2_min, growth_range2_max)

    st.header('Simulation Results')
    st.subheader('Raw Data')
    st.dataframe(df)
    st.subheader('Summary Statistics')
    st.dataframe(summary)

    st.header('Portfolio Return Distribution')

    st.subheader('VC Deals')
    vc_chart_data = np.concatenate(df['vc_returns'].values)
    fig_vc, ax_vc = plt.subplots()
    sns.histplot(vc_chart_data, kde=True, ax=ax_vc, stat="probability")
    ax_vc.set_xlabel('Return')
    ax_vc.set_ylabel('Probability')
    st.pyplot(fig_vc)

    st.subheader('Growth Deals')
    growth_chart_data = np.concatenate(df['growth_returns'].values)
    fig_growth, ax_growth = plt.subplots()
    sns.histplot(growth_chart_data, kde=True, ax=ax_growth, stat="probability")
    ax_growth.set_xlabel('Return')
    ax_growth.set_ylabel('Probability')
    st.pyplot(fig_growth)

    st.subheader('Combined')
    fig_combined, ax_combined = plt.subplots()
    sns.histplot(vc_chart_data, kde=True, color='blue', label='VC Deals', ax=ax_combined, stat="probability")
    sns.histplot(growth_chart_data, kde=True, color='red', label='Growth Deals', ax=ax_combined, stat="probability")
    ax_combined.legend()
    ax_combined.set_xlabel('Return')
    ax_combined.set_ylabel('Probability')
    st.pyplot(fig_combined)

    st.header('Portfolio Return vs Number of Growth Deals')
    fig2, ax2 = plt.subplots()
    sns.lineplot(x='growth_deals', y='mean_return', data=summary, ax=ax2, label='Mean Return')
    ax2.fill_between(summary['growth_deals'], summary['percentile_25'], summary['percentile_75'], alpha=0.2)
    ax2.legend()
    ax2.set_xlabel('Number of Growth Deals')
    ax2.set_ylabel('Portfolio Return')
    st.pyplot(fig2)


if __name__ == "__main__":
    main()
