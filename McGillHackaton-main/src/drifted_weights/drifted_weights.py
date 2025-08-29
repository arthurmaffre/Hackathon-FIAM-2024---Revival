import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# SIMULATED RESULTS TO REPLACE WITH LONG AND SHORT PORTFOLIO WEIGHTS
# Simulated monthly weights for rebalancing (replace with monthly weight matrix)
rebalance_dates = pd.date_range(start="2023-01-01", end="2023-03-31", freq='BME')  # Last business day of each month
weights = {
    'GOOGL': [0.8, -0.1, -0.4],
    'AAPL': [-0.1, 0.8, -0.9],
    'TSLA': [-0.3, 0.8, 1.8],
    'GME': [0.6, -0.5, 0.5]
} 
monthly_weights = pd.DataFrame(weights, index=rebalance_dates, columns=["AAPL", "GOOGL", "TSLA","GME"])
long_weights = monthly_weights.where(monthly_weights > 0, 0) 
short_weights = monthly_weights.where(monthly_weights < 0, 0)  
long_weights = long_weights.div(long_weights.abs().sum(axis=1), axis=0).fillna(0)
short_weights = short_weights.div(short_weights.abs().sum(axis=1), axis=0).fillna(0)

print(long_weights)
print(short_weights)


# SIMULATED LONG SHORT ALLOCATION TO REPLACE WITH SJM OUTPUT
start_date = '2000-01-01'
end_date = '2023-12-31'
month_end_dates = pd.date_range(start=start_date, end=end_date, freq='BME')

random_values = np.random.uniform(0.1, 0.5, size=len(month_end_dates))
short_allocation = pd.Series(data=random_values, index=month_end_dates)
print(short_allocation)


# Import the real stock prices data (transform into fucntion?)

daily_prices = pd.read_csv('../data/intermediate_data/preprocess_data/stocks_prices_df_preprocessed.csv',index_col=0)
daily_prices.index = pd.to_datetime(daily_prices.index)

# Tranform SJM Series into usable Dataframe

def transform_short_allocation(short_allocation):
    """
    Transforms the short_allocation series into a long_short_alloc DataFrame.

    Parameters:
    short_allocation (pd.Series): Series containing short allocation values.

    Returns:
    pd.DataFrame: DataFrame with long and short allocations.
    """

    long_short_alloc = pd.DataFrame({
        'long': 1 + short_allocation.abs(),
        'short': -short_allocation
    })
    long_short_alloc.index = pd.to_datetime(long_short_alloc.index)
    
    return long_short_alloc

# Implement function to transform series
long_short_alloc = transform_short_allocation(short_allocation)

#Integrate SJM into weighting (remove if this will be done in other step as discussed with Thomas)

def integrate_sjm_into_weighting(long_weights, short_weights, long_short_alloc):
    """
    Integrate SJM into weighting based on long and short weights and allocation.

    Parameters:
    long_weights (pd.DataFrame): DataFrame containing long weights with dates as index.
    short_weights (pd.DataFrame): DataFrame containing short weights with dates as index.
    long_short_alloc (pd.DataFrame): DataFrame containing long and short allocation with dates as index.

    Returns:
    pd.DataFrame: DataFrame containing overall portfolio weights.
    """
    # Ensure long and short portfolio dataframes have the columns aligned
    long_weights, short_weights = long_weights.align(short_weights, axis=1, join='outer', fill_value=0)

    # Align the start dates
    aligned_start_date = max(long_weights.index.min(), long_short_alloc.index.min(), short_weights.index.min())
    long_weights = long_weights.loc[aligned_start_date:]
    short_weights = short_weights.loc[aligned_start_date:]
    long_short_alloc = long_short_alloc.loc[aligned_start_date:]

    # Calculate the overall portfolio rebalancing weights
    adjusted_long_weights = long_weights.multiply(long_short_alloc['long'], axis=0)
    adjusted_short_weights = short_weights.multiply(abs(long_short_alloc['short']), axis=0)

    overall_portfolio_weights = adjusted_long_weights.add(adjusted_short_weights, fill_value=0)

    return overall_portfolio_weights

# Implement function to get overall portfolio weights with SJM integrated

Overall_portfolio_weights = integrate_sjm_into_weighting(long_weights, short_weights, long_short_alloc)


# Calculate the drifted weights

def calculate_drifted_weights(daily_prices, overall_portfolio_weights):
    """
    Calculate the drifted weights based on daily prices and overall portfolio weights.

    Parameters:
    daily_prices (pd.DataFrame): DataFrame containing daily prices with dates as index.
    overall_portfolio_weights (pd.DataFrame): DataFrame containing overall portfolio weights with dates as index.

    Returns:
    pd.DataFrame: DataFrame containing drifted weights.
    """
    # Calculate daily returns
    daily_returns = daily_prices.pct_change(fill_method=None)

    # Align the columns of the overall portfolio weights and daily returns
    overall_portfolio_weights_aligned, daily_returns_aligned = overall_portfolio_weights.align(daily_returns, axis=1, join='inner')
    overall_portfolio_weights_aligned.fillna(0, inplace=True)

    # Align the start dates of the dataframes
    aligned_start_date = max(overall_portfolio_weights_aligned.index.min(), daily_returns_aligned.index.min())
    overall_portfolio_weights_aligned = overall_portfolio_weights_aligned.loc[aligned_start_date:]
    daily_returns_aligned = daily_returns_aligned.loc[aligned_start_date:]

    # Initialize the drifted weights DataFrame
    drifted_weights = pd.DataFrame(index=daily_returns_aligned.index, columns=overall_portfolio_weights_aligned.columns)
    start_weights = overall_portfolio_weights_aligned.iloc[0]
    drifted_weights.iloc[0] = start_weights

    # Calculate the drifted weights
    for date in tqdm(daily_returns_aligned.index[1:], desc="Calculating drifted weights"):
        if date in overall_portfolio_weights_aligned.index:
            new_weights = overall_portfolio_weights_aligned.loc[date]
        else:
            return_factor = (1 + daily_returns_aligned.loc[date])
            previous_date = drifted_weights.index[drifted_weights.index.get_loc(date) - 1]
            previous_weights = drifted_weights.loc[previous_date]
            new_weights = previous_weights * return_factor
            new_weights[daily_returns_aligned.loc[date].isna()] = 0
            if abs(new_weights).sum() != 0:
                new_weights = new_weights / abs(new_weights).sum()

        drifted_weights.loc[date] = new_weights

    # Cut the DataFrame to only contain the month following the last rebalancing date
    cutoff_date = overall_portfolio_weights.index[-1] + pd.offsets.MonthBegin(1) + pd.offsets.MonthEnd(1) - pd.offsets.BDay(1)
    drifted_weights = drifted_weights.loc[:cutoff_date]

    return drifted_weights

# Implement the function

drifted_weights = calculate_drifted_weights(daily_prices, overall_portfolio_weights)

# Separate the long and short drifted weights (useful for the short and long sector allocation)

def separate_and_normalize_weights(drifted_weights):
    """
    Separate and normalize the long and short portfolio drifted weights.

    Parameters:
    drifted_weights (pd.DataFrame): DataFrame containing drifted weights.

    Returns:
    tuple: Two DataFrames containing normalized short and long drifted weights.
    """
    # Normalize function
    def normalize_row(row):
        row_sum = row.sum()
        if row_sum != 0:
            return row / row_sum
        return row

    # Separate and normalize short weights (displayed as positive values)
    short_drifted_weights = drifted_weights.where(drifted_weights < 0, 0)
    short_drifted_weights = short_drifted_weights.apply(normalize_row, axis=1)

    # Separate and normalize long weights
    long_drifted_weights = drifted_weights.where(drifted_weights > 0, 0)
    long_drifted_weights = long_drifted_weights.apply(normalize_row, axis=1)

    return short_drifted_weights, long_drifted_weights


short_weights, long_weights = separate_and_normalize_weights(drifted_weights)

# SCETOR BREAKDOWN OVER TIME 

# Import Sector data (transform into function?)

sector_data = pd.read_csv('../data/raw_data/mapping_stocks_country_df_preprocessed.csv', index_col=0)
sector_data = sector_data[['ticker_name', 'Sector']]
sector_data['Sector'] = sector_data['Sector'].str.split(',').str[0]
print(sector_data)


def calculate_sector_weights(weights_df, sector_data):
    """
    Calculate the normalized sector weights from a weights DataFrame and a sector DataFrame.
    
    Parameters:
    weights_df (pd.DataFrame): DataFrame containing weights with dates as index and tickers as columns.
    sector_data (pd.DataFrame): DataFrame containing sector information with 'ticker_name' and 'Sector' columns.
    
    Returns:
    pd.DataFrame: Normalized sector weights DataFrame.
    """

    melted_weights = weights_df.reset_index().melt(id_vars=weights_df.index.name, var_name='ticker', value_name='weight')
    melted_weights.rename(columns={weights_df.index.name: 'date'}, inplace=True)
    merged_data = melted_weights.merge(sector_data, left_on='ticker', right_on='ticker_name', how='left')
    sector_weights = merged_data.groupby(['date', 'Sector'])['weight'].sum().unstack(fill_value=0)
    sector_weights_normalized = sector_weights.div(sector_weights.sum(axis=1), axis=0).fillna(0)
    sector_weights_normalized.index = weights_df.index
    return sector_weights_normalized

# Implement function to calculate sector weights for long and short portfolios

short_sector_weights = calculate_sector_weights(short_drifted_weights, sector_data)
long_sector_weights = calculate_sector_weights(long_drifted_weights, sector_data)

# Plot the sector breakdown over time

def plot_stacked_area(df, title='Sector Breakdown of Portfolio Over Time', xlabel='Date', ylabel='Portfolio %', figsize=(12, 6)):
    """
    Create a stacked area plot for the sector weights DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing sector weights with dates as index and sectors as columns.
    title (str): Title of the plot.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    figsize (tuple): Size of the figure.
    """

    plt.figure(figsize=figsize)
    df.plot(kind='area', alpha=0.5, stacked=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Sectors')
    plt.tight_layout()
    plt.show()


# Implement function to show long and short portfolio sector weights
plot_stacked_area(short_sector_weights,title = 'Sector Breakdown of Short Portfolio Over Time')
plot_stacked_area(long_sector_weights, title = 'Sector Breakdown of Long Portfolio Over Time')



# Top 10 holdings

def top_10_holdings(drifted_weights):
    """
    Calculate the top 10 stock holdings based on average weights from drifted weights.

    Parameters:
    drifted_weights (pd.DataFrame): DataFrame containing the portfolio weights with tickers as columns.

    Returns:
    pd.DataFrame: A DataFrame containing the top 10 holdings, their average weights, and position (Long/Short).
    """
    average_weights = drifted_weights.mean()
    avg_weights_df = pd.DataFrame(average_weights).reset_index()
    avg_weights_df.columns = ['Ticker', 'Average Weight']
    avg_weights_df['Position'] = avg_weights_df['Average Weight'].apply(
        lambda x: 'Long' if x > 0 else 'Short'
    )
    avg_weights_df['Abs Weight'] = avg_weights_df['Average Weight'].abs()
    top_10 = avg_weights_df.sort_values(by='Abs Weight', ascending=False).head(10)
    top_10.reset_index(drop=True, inplace=True)
    top_10.index = top_10.index + 1  # Shift index to start from 1
    
    return top_10[['Ticker', 'Average Weight', 'Position']]

