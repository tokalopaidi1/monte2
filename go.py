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

    # Calculate mode separately
    modes = df.groupby('growth_deals')['roi'].apply(lambda x: mode(x)[0][0]).reset_index()
    modes.columns = ['growth_deals', 'mode']

    # Merge mode into summary
    summary = pd.merge(summary, modes, on='growth_deals')

    return df, summary


def main():
    st.title("Monte Carlo Simulation for Portfolio Returns")

    # Sidebar controls
    n_runs = st.sidebar.number_input("Number of simulations:", min_value=100, value=1000, step=100)
    fund = st.sidebar.number_input("Initial Fund:", min_value=100000, value=8000000, step=100000)
    n_investments = st.sidebar.number_input("Number of Investments:", min_value=1, value=20, step=1)

    st.sidebar.subheader("VC Investments")
    vc_failure_rate = st.sidebar.slider("VC Failure Rate:", min_value=0.0, max_value=1.0, value=0.65, step=0.01)
    vc_min_return = st.sidebar.number_input("VC Min Return Multiplier:", min_value=1.0, value=1.0, step=0.1)
    vc_max_return = st.sidebar.number_input("VC Max Return Multiplier:", min_value=1.0, value=10.0, step=0.1)
    vc_power_law_exponent = st.sidebar.number_input("VC Power Law Exponent:", min_value=0.1, max_value=5.0, value=0.85, step=0.05)

    # Rest of the code remains the same...

if __name__ == "__main__":
    main()
