import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import powerlaw


def monte_carlo_simulation(n_runs, fund, n_investments, vc_failure_rate, vc_range1_rate, vc_range2_rate,
                           growth_failure_rate, growth_range1_rate, growth_range2_rate,
                           growth_distribution_mean, growth_distribution_std, vc_power_law_exponent):

    # ... same as before


def main():

    # ... same as before

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
    sns.histplot(vc_chart_data, kde=True, ax=ax_vc)
    ax_vc.set_xlabel('Return')
    ax_vc.set_ylabel('Frequency')
    st.pyplot(fig_vc)

    st.subheader('Growth Deals')
    growth_chart_data = np.concatenate(df['growth_returns'].values)
    fig_growth, ax_growth = plt.subplots()
    sns.histplot(growth_chart_data, kde=True, ax=ax_growth)
    ax_growth.set_xlabel('Return')
    ax_growth.set_ylabel('Frequency')
    st.pyplot(fig_growth)

    st.subheader('Combined')
    fig_combined, ax_combined = plt.subplots()
    sns.histplot(vc_chart_data, kde=True, color='blue', label='VC Deals', ax=ax_combined)
    sns.histplot(growth_chart_data, kde=True, color='green', label='Growth Deals', ax=ax_combined)
    ax_combined.set_xlabel('Return')
    ax_combined.set_ylabel('Frequency')
    ax_combined.legend()
    st.pyplot(fig_combined)


if __name__ == '__main__':
    main()
