import pandas as pd

class OptionFeatures:
    def __init__(self, option_data_pre_2010, option_data_post_2010, hackathon_data):
        """
        Initialise la classe avec les données des options.
        """
        self.option_data_pre_2010 = option_data_pre_2010.copy()
        self.option_data_post_2010 = option_data_post_2010.copy()
        self.hackathon_data = hackathon_data.copy()
        self.preprocess_data()

    def map_cusip_to_ticker(self):
        cusip_ticker_map = self.hackathon_data.groupby('cusip')['stock_ticker'].unique().to_dict()

        # Remove leading zeros from CUSIP keys
        self.cusip_ticker_map = {cusip.lstrip('0'): tickers for cusip, tickers in cusip_ticker_map.items()}
    
    def add_tickers_to_pre_2010_option_data(self):
        
        # Ensure the CUSIP column in the option DataFrame is treated as a string and remove leading zeros
        self.option_data_pre_2010['cusip'] = self.option_data_pre_2010['cusip'].astype(str).str.lstrip('0')
        
        # Define a function to extract the first ticker for each CUSIP from the map
        def get_ticker(cusip):
            tickers = self.cusip_ticker_map.get(cusip, None)
            return tickers[0] if tickers else None  # Choose the first ticker if there are multiple
        
        # Apply the function to add a 'ticker' column
        self.option_data_pre_2010['stock_ticker'] = self.option_data_pre_2010['cusip'].apply(get_ticker)
        
    def preprocess_data_pre_2010(self):
        """
        Prépare les données pour l'analyse.
        """
        # Convertir les colonnes de dates en format datetime
        self.option_data_pre_2010['date'] = pd.to_datetime(self.option_data_pre_2010['date'])
        self.option_data_pre_2010['expiration'] = pd.to_datetime(self.option_data_pre_2010['exdate'])

        # Add a column for the year and month
        self.option_data_pre_2010['year_month'] = self.option_data_pre_2010['date'].dt.to_period('M')

        # Add ticker column
        self.map_cusip_to_ticker()
        self.add_tickers_to_pre_2010_option_data()

        # Calculate the mid price
        self.option_data_pre_2010['mid_price'] = (self.option_data_pre_2010['best_bid'] + self.option_data_pre_2010['best_offer']) / 2

        # Ensure the volume column is treated as an integer
        self.option_data_pre_2010['volume'] = self.option_data_pre_2010['volume'].astype(int)
        
        # Remove options with 0 volume
        self.option_data_pre_2010 = self.option_data_pre_2010[self.option_data_pre_2010['volume'] > 0]
        
        # Calculer les jours jusqu'à l'échéance
        self.option_data_pre_2010['days_to_expiration'] = (self.option_data_pre_2010['expiration'] - self.option_data_pre_2010['date']).dt.days
        
        # Calculer le delta en valeur absolue
        self.option_data_pre_2010['abs_delta'] = self.option_data_pre_2010['delta'].abs()
        
        # Return only the necessary columns
        self.option_data_pre_2010 = self.option_data_pre_2010[['date', 'year_month', 'stock_ticker', 'exdate', 'cp_flag', 'strike_price', 
                                                               'best_bid', 'best_offer', 'mid_price', 'volume', 'impl_volatility', 
                                                               'delta', 'days_to_expiration', 'abs_delta']]

    def preprocess_data_post_2010(self):
        """
        Prépare les données pour l'analyse.
        """
        # Convertir les colonnes de dates en format datetime
        self.option_data_post_2010['date'] = pd.to_datetime(self.option_data_post_2010['date'])
        self.option_data_post_2010['expiration'] = pd.to_datetime(self.option_data_post_2010['exdate'])

        # Add a column for the year and month
        self.option_data_post_2010['year_month'] = self.option_data_post_2010['date'].dt.to_period('M')

        # Calculate the mid price
        self.option_data_post_2010['mid_price'] = (self.option_data_post_2010['best_bid'] + self.option_data_post_2010['best_offer']) / 2

        # Get ticker
        self.option_data_post_2010['stock_ticker'] = self.option_data_post_2010['symbol'].str.extract(r'([A-Z]+)')

        # Ensure the volume column is treated as an integer
        self.option_data_post_2010['volume'] = self.option_data_post_2010['volume'].astype(int)
        
        # Remove options with 0 volume
        self.option_data_post_2010 = self.option_data_post_2010[self.option_data_post_2010['volume'] > 0]
        
        # Calculer les jours jusqu'à l'échéance
        self.option_data_post_2010['days_to_expiration'] = (self.option_data_post_2010['expiration'] - self.option_data_post_2010['date']).dt.days
        
        # Calculer le delta en valeur absolue
        self.option_data_post_2010['abs_delta'] = self.option_data_post_2010['delta'].abs()
        
        # Return only the necessary columns
        self.option_data_post_2010 = self.option_data_post_2010[['date', 'year_month', 'stock_ticker', 'exdate', 'cp_flag', 'strike_price', 
                                                               'best_bid', 'best_offer', 'mid_price', 'volume', 'impl_volatility', 
                                                               'delta', 'days_to_expiration', 'abs_delta']]
        
    def preprocess_data(self):
        """
        Prépare les données pour l'analyse.
        """
        self.preprocess_data_pre_2010()
        self.preprocess_data_post_2010()
        
        # Concatenate the pre-2010 and post-2010 data
        self.data = pd.concat([self.option_data_pre_2010, self.option_data_post_2010])

        # Add year_month column to hackathon_data
        self.hackathon_data['date'] = pd.to_datetime(self.hackathon_data['date'], format='%Y%m%d')
        self.hackathon_data['year_month'] = self.hackathon_data['date'].dt.to_period('M')

        # Remove rows with nans
        self.data = self.data.dropna()
        
        # Define self.option_features DataFrame and give it date, year_month, and ticker columns of data
        self.option_features = self.data[['date', 'year_month', 'stock_ticker']].copy()
    
    def filter_options_data(self, delta_target=0.5, delta_tolerance=0.1, dte_target=30, dte_tolerance=5):
        """
        Filtre les options selon le delta et les jours jusqu'à l'échéance.
        """
        # Filtrer les options avec un delta proche de 0.5
        delta_min = delta_target - delta_tolerance
        delta_max = delta_target + delta_tolerance
        self.filtered_data = self.data[
            (self.data['abs_delta'] >= delta_min) & (self.data['abs_delta'] <= delta_max)
        ]
        
        # Filtrer les options avec environ 30 jours à l'échéance
        dte_min = dte_target - dte_tolerance
        dte_max = dte_target + dte_tolerance
        self.filtered_data = self.filtered_data[
            (self.filtered_data['days_to_expiration'] >= dte_min) & (self.filtered_data['days_to_expiration'] <= dte_max)
        ]
    
    # def calculate_cvol_pvol(self):
    #     """
    #     An, Ang, Bali and Cakici (2014)
    #     Calculate the monthly change in implied volatility for each day for ATM options.
    #     """
        
    #     # Create a new column 'date_prev_month' by subtracting one month
    #     self.option_features['date_prev_month'] = self.option_features['date'] - pd.DateOffset(months=1)
        
    #     # Prepare DataFrame for merging
    #     prev_month_data = self.option_features[['secid', 'cp_flag', 'date', 'impl_volatility']].copy()
    #     prev_month_data.rename(columns={'date': 'date_prev_month', 'impl_volatility': 'iv_prev_month'}, inplace=True)
        
    #     # Merge current data with previous month's data
    #     self.option_features = pd.merge(
    #         self.option_features,
    #         prev_month_data,
    #         on=['secid', 'cp_flag', 'date_prev_month'],
    #         how='left'
    #     )
        
    #     # Calculate the IV change
    #     self.option_features['cvol_pvol'] = self.option_features['impl_volatility'] - self.option_features['iv_prev_month']
        

    def calculate_monthly_avg_impl_volatility(self):
        """
        An, Ang, Bali and Cakici (2014)
        Calcule la volatilité implicite moyenne mensuelle des calls et puts ATM pour chaque action.
        """        
        # Group by 'stock_ticker' and 'month' to calculate the monthly average implied volatility
        monthly_avg_impl_volatility = self.filtered_data.groupby(['stock_ticker', 'year_month'])['impl_volatility'].mean().reset_index(name='monthly_avg_impl_volatility')

        # Merge the monthly average implied volatility back into self.option_features
        self.option_features = pd.merge(self.option_features, monthly_avg_impl_volatility, on=['stock_ticker', 'year_month'], how='left')
  
    
    def calculate_monthly_avg_skewness(self, delta_otm_put=-0.2, delta_otm_tolerance=0.05, delta_atm=0.5, delta_atm_tolerance=0.05, dte_target=30, dte_tolerance=5):
        """
        Xing, Zhang, and Zhao (2010)
        Skewness is defined as the difference between the implied volatility of out-of-the-money puts (delta around = -0.2)
        and the average implied volatility of at-the-money options (calls and puts with abs(delta) around 0.5).
        The skewness is calculated daily, and averaged for each month, then merged into self.option_features.
        """
        # Filter OTM puts with delta close to -0.2
        delta_otm_min = delta_otm_put - delta_otm_tolerance
        delta_otm_max = delta_otm_put + delta_otm_tolerance
        otm_puts = self.data[
            (self.data['cp_flag'] == 'P') &
            (self.data['delta'] >= delta_otm_min) &
            (self.data['delta'] <= delta_otm_max) &
            (self.data['days_to_expiration'] >= dte_target - dte_tolerance) &
            (self.data['days_to_expiration'] <= dte_target + dte_tolerance)
        ].copy()

        # Filter ATM options with abs(delta) close to 0.5
        delta_atm_min = delta_atm - delta_atm_tolerance
        delta_atm_max = delta_atm + delta_atm_tolerance
        atm_options = self.data[
            (self.data['abs_delta'] >= delta_atm_min) &
            (self.data['abs_delta'] <= delta_atm_max) &
            (self.data['days_to_expiration'] >= dte_target - dte_tolerance) &
            (self.data['days_to_expiration'] <= dte_target + dte_tolerance)
        ].copy()

        # Get avg implied volatility of ATM calls and puts
        avg_atm_iv = atm_options.groupby(['stock_ticker', 'date'])['impl_volatility'].mean().reset_index()
        avg_atm_iv = avg_atm_iv.rename(columns={'impl_volatility': 'avg_atm_iv'})

        # Get avg implied volatility of OTM puts
        otm_put_iv = otm_puts.groupby(['stock_ticker', 'date'])['impl_volatility'].mean().reset_index()
        otm_put_iv = otm_put_iv.rename(columns={'impl_volatility': 'otm_put_iv'})

        # Merge the data to compute skewness
        skewness_df = pd.merge(otm_put_iv, avg_atm_iv, on=['stock_ticker', 'date'], how='inner')

        # Compute skewness
        skewness_df['skewness'] = skewness_df['otm_put_iv'] - skewness_df['avg_atm_iv']

        # Create a 'month' column from the date
        skewness_df['year_month'] = skewness_df['date'].dt.to_period('M')

        # Calculate the monthly average skewness
        monthly_skewness = skewness_df.groupby(['stock_ticker', 'year_month'])['skewness'].mean().reset_index(name='monthly_avg_skewness')

        # Merge the monthly average skewness back into self.option_features
        self.option_features = pd.merge(self.option_features, monthly_skewness, on=['stock_ticker', 'year_month'], how='left')

    def calculate_monthly_avg_opt_baspread(self):
        """
        Calculates the average bid-ask spread % of options for each stock for each month.
        """
        # Calculate the bid-ask spread percentage for each option
        self.data['opt_baspread'] = (self.data['best_offer'] - self.data['best_bid']) / self.data['mid_price']

        # Group by 'stock_ticker' and 'month' to calculate the monthly average bid-ask spread across options
        monthly_avg_opt_baspread = self.data.groupby(['stock_ticker', 'year_month'])['opt_baspread'].mean().reset_index(name='monthly_avg_opt_baspread')

        # Merge the monthly average bid-ask spread into self.option_features
        self.option_features = pd.merge(self.option_features, monthly_avg_opt_baspread, on=['stock_ticker', 'year_month'], how='left')


    def add_option_features_to_hackathon_data(self):
        """
        Ajoute les caractéristiques des options à l'ensemble de données du hackathon.
        """
    
        # Merge the option features with the hackathon data
        self.hackathon_data = pd.merge(self.hackathon_data, self.option_features, on=['year_month', 'stock_ticker'], how='left')

        return self.hackathon_data
    

    def run_analysis(self):
        """
        Exécute toutes les étapes de l'analyse.
        """
        self.filter_options_data()
        self.calculate_monthly_avg_impl_volatility()
        # self.calculate_cvol_pvol() - bad predictor of stock excess return
        self.calculate_monthly_avg_skewness()
        self.calculate_monthly_avg_opt_baspread()
        
        # Keep only one row per date and ticker by dropping duplicates
        self.option_features = self.option_features.drop_duplicates(subset=['year_month', 'stock_ticker'])

        # Keep only the necessary columns
        self.option_features = self.option_features[['year_month', 'stock_ticker', 'monthly_avg_impl_volatility', 
                                                     'monthly_avg_skewness', 'monthly_avg_opt_baspread']]
