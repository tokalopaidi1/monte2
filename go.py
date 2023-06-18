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
                    power_law_dist = powerlaw(a=vc_power_law_exponent, scale=1.0)
                    multiplier = power_law_dist.rvs()

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
        prob_5x=pd.NamedAgg(column='portfolio_return', aggfunc=lambda x: np.mean(x >= 5 * fund))
    ).reset_index()

    return df, summary


def plot_distribution(df, title):
    plt.figure()
    sns.histplot(df, kde=True)
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.title(title)
    st.pyplot()


def main():
    st.title('Monte Carlo Simulation App')

    st.sidebar.title('Simulation Parameters')
    n_runs = st.sidebar.number_input('Number of Runs', value=1000, min_value=1)
    fund = st.sidebar.number_input('Initial Fund Amount', value=8000000, min_value=1)
    n_investments = st.sidebar.number_input('Number of Investments', value=25, min_value=1)

    st.sidebar.title('VC Deals')
    vc_failure_rate = st.sidebar.slider('VC Percentage of Failure', 0.0, 1.0, 0.2, step=0.01)
    vc_range1_rate = st.sidebar.slider('VC Percentage for 2x-15x', 0.0, 1.0, 0.5, step=0.01)
    vc_range2_rate = st.sidebar.slider('VC Percentage for 25x-170x', 0.0, 1.0, 0.3, step=0.01)
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
    plot_distribution(vc_chart_data, 'VC Deals')

    st.subheader('Growth Deals')
    growth_chart_data = np.concatenate(df['growth_returns'].values)
    plot_distribution(growth_chart_data, 'Growth Deals')

    st.subheader('Combined')
    combined_chart_data = np.concatenate([vc_chart_data, growth_chart_data])
    plot_distribution(combined_chart_data, 'Combined Deals')


if __name__ == '__main__':
    main()
