import pandas as pd
import numpy as np

class RiskNeutralMoments:
    def __init__(self, data):
        """
        Initialize the class with the options data.
        """
        self.data = data.copy()
        self.preprocess_data()
    
    def preprocess_data(self):
        """
        Prepare the data for analysis.
        """
        # Convert date columns to datetime format
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data['expiration'] = pd.to_datetime(self.data['exdate'])
        
        # Calculate days to expiration
        self.data['days_to_expiration'] = (self.data['expiration'] - self.data['date']).dt.days
        
        # Calculate time to maturity in years
        self.data['tau'] = self.data['days_to_expiration'] / 365.25
        
        # Calculate absolute delta
        self.data['abs_delta'] = self.data['delta'].abs()

        # Calculate mid price
        self.data['mid_price'] = (self.data['best_offer'] + self.data['best_bid']) / 2

        # Normalize strike price
        self.data['strike_price'] = self.data['strike_price'] / 1000

        # Get ticker
        self.data['ticker'] = self.data['symbol'].str.extract(r'([A-Z]+)')

        # Load price data from CSV
        price_data = pd.read_csv('../data/raw_data/hackathon_sample_v2.csv')

        price_data['date'] = pd.to_datetime(price_data['date'], format='%Y%m%d')
        
        self.data = pd.merge(
            self.data,
            price_data[['ticker', 'date', 'prc']],
            on=['ticker', 'date'],
            how='left'
        )

        self.data.rename(columns={'prc': 'underlying_price'}, inplace=True)

        # Remove observations with missing essential values
        self.data.dropna(subset=['impl_volatility', 'delta', 'days_to_expiration', 'mid_price', 'underlying_price'], inplace=True)
        
        # Remove options with zero or negative mid prices
        self.data = self.data[self.data['mid_price'] > 0]
        
    def filter_options(self, dte_target=30, dte_tolerance=5):
        """
        Filter options based on delta and days to expiration.
        """

        # Filter options with days to expiration close to dte_target
        dte_min = dte_target - dte_tolerance
        dte_max = dte_target + dte_tolerance
        self.filtered_data = self.data[
            (self.data['days_to_expiration'] >= dte_min) & (self.data['days_to_expiration'] <= dte_max)
        ]
        
        return self.filtered_data
    
    def compute_contract_prices(self, r):
        """
        Compute V_i,t(τ), W_i,t(τ), and X_i,t(τ) for each date, security, and tau.

        Parameters:
        - r: Risk-free interest rate (annual, continuous compounding)
        """
        self.contract_prices = []
        
        # Group data by 'secid' and 'date'
        grouped = self.filtered_data.groupby(['secid', 'date'])

        for (secid, date), group in grouped:

            # Get unique tau values for this group
            tau = group['tau'].unique()[0]

            # Ensure that S_t is consistent across all rows for this secid and date
            S_t = group['underlying_price'].unique()[0]
    
            # Separate OTM calls and puts
            otm_calls = group[(group['cp_flag'] == 'C') & (group['strike_price'] > S_t)]
            otm_puts = group[(group['cp_flag'] == 'P') & (group['strike_price'] <= S_t)]
                   
            # Compute V, W, X using your integration functions
            V = self.compute_integral(otm_calls, otm_puts, S_t, self.integrand_V_call, self.integrand_V_put)
            W = self.compute_integral(otm_calls, otm_puts, S_t, self.integrand_W_call, self.integrand_W_put)
            X = self.compute_integral(otm_calls, otm_puts, S_t, self.integrand_X_call, self.integrand_X_put)
 
            # Store results
            self.contract_prices.append({
                'secid': secid,
                'date': date,
                'tau': tau,
                'V': V,
                'W': W,
                'X': X,
                'S_t': S_t,
            })

        
    @staticmethod
    def integrand_V_call(K, S_t, call_prices):
        return ((2 * (1 - np.log(K / S_t))) / K**2) * call_prices
    
    @staticmethod    
    def integrand_V_put(K, S_t, put_prices):
        return ((2 * (1 + np.log(K / S_t))) / K**2) * put_prices
    
    @staticmethod
    def integrand_W_call(K, S_t, call_prices):
        ln_KF = np.log(K / S_t)
        return (((6 * ln_KF) - (3 * ln_KF**2)) / K**2) * call_prices
    
    @staticmethod
    def integrand_W_put(K, S_t, put_prices):
        ln_KF = np.log(K / S_t)
        return (((6 * ln_KF) + (3 * ln_KF**2)) / K**2) * put_prices
    
    @staticmethod
    def integrand_X_call(K, S_t, call_prices):
        ln_KF = np.log(K / S_t)
        return (((12 * ln_KF**2)-(4 * ln_KF**3)) / K**2) * call_prices
    
    @staticmethod
    def integrand_X_put(K, S_t, put_prices):
        ln_KF = np.log(K / S_t)
        return (((12 * ln_KF**2)+(4 * ln_KF**3)) / K**2) * put_prices
    
    def compute_integral(self, calls, puts, S_t, integrand_func_call, integrand_func_put):
        """
        Compute the integral over calls and puts using the trapezoidal rule.
        """
        integral = 0.0
        
        # Process puts
        if not puts.empty:
            puts = puts.sort_values('strike_price', ascending=False)
            K_put = puts['strike_price'].values
            prices_put = puts['mid_price'].values
            integrand_values_put = integrand_func_put(K_put, S_t, prices_put)
            integral_put = np.trapz(integrand_values_put, K_put)
            integral += integral_put
        
        # Process calls
        if not calls.empty:
            calls = calls.sort_values('strike_price')
            K_call = calls['strike_price'].values
            prices_call = calls['mid_price'].values
            integrand_values_call = integrand_func_call(K_call, S_t, prices_call)
            integral_call = np.trapz(integrand_values_call, K_call)
            integral += integral_call
        
        return integral
    
    def compute_risk_neutral_moments(self, r):
        """
        Compute the risk-neutral variance, skewness, and kurtosis.

        Parameters:
        - r: Risk-free interest rate (annual, continuous compounding)
        """
        self.moments = []
        
        for item in self.contract_prices:
            secid = item['secid']
            date = item['date']
            tau = item['tau']
            V = item['V']
            W = item['W']
            X = item['X']

            e_rt = np.exp(r * tau)
            
            # Compute mu
            mu = e_rt - 1 - ((e_rt * V) / 2) - ((e_rt * W) / 6) - ((e_rt * X) / 24)
            
            # Compute variance
            variance = e_rt * V - mu**2
              
            # Compute skewness
            numerator_skew = e_rt * W - 3 * mu * e_rt * V + 2 * mu**3
            skewness = numerator_skew / (variance ** 1.5)
            
            # Compute kurtosis
            numerator_kurt = e_rt * X - 4 * mu * e_rt * W + 6 * e_rt * mu**2 * V - mu**4
            kurtosis = numerator_kurt / (variance ** 2)
            
            self.moments.append({
                'secid': secid,
                'date': date,
                'variance': variance,
                'skewness': skewness,
                'kurtosis': kurtosis
            })
    
    def run_analysis(self, dte_target=30, dte_tolerance=3, r=0.01):
        """
        Execute all steps of the analysis.

        Parameters:
        - delta_target: Target delta value for filtering
        - delta_tolerance: Tolerance around the target delta
        - dte_target: Target days to expiration
        - dte_tolerance: Tolerance around the target days to expiration
        - r: Risk-free interest rate (annual, continuous compounding)
        """
        self.filter_options(dte_target, dte_tolerance)
        self.compute_contract_prices(r)
        self.compute_risk_neutral_moments(r)
        return pd.DataFrame(self.moments)
