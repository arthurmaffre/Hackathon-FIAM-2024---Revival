import pandas as pd


# Charget le dataset à partir d'un fichier CSV
dataset = pd.read_csv('../../data/raw_data/hackathon_sample_v2.csv')

# Assurer que la colonne 'date' est bien formatée
dataset['date'] = pd.to_datetime(dataset['date'], format='%Y%m%d').dt.to_period('M')

# Étape 2 : Détecter et supprimer les tickers avec des interruptions dans les dates
def has_interruption(group):
    # Créer une plage complète de dates pour ce ticker
    full_date_range = pd.period_range(start=group['date'].min(), end=group['date'].max(), freq='M')
    # Si la série n'a pas toutes les dates dans la plage, c'est une interruption
    return len(full_date_range) != len(group)

# Identifier les tickers avec des interruptions
tickers_with_interruptions = dataset.groupby('stock_ticker').filter(lambda x: has_interruption(x))['stock_ticker'].unique()

# Afficher les tickers qui seront supprimés
print(f"Tickers supprimés à cause des interruptions : {tickers_with_interruptions}")

# Filtrer le dataset pour supprimer les tickers avec interruptions
filtered_dataset = dataset.groupby('stock_ticker').filter(lambda x: not has_interruption(x))



# Identifier la date la plus ancienne et la plus récente dans le dataset
start_date = filtered_dataset['date'].min()
end_date = filtered_dataset['date'].max()

# Créer une plage de dates complète pour tout le dataset
full_date_range = pd.period_range(start=start_date, end=end_date, freq='M')

# Obtenir la liste des tickers uniques
tickers = filtered_dataset['stock_ticker'].unique()

# Créer un DataFrame avec toutes les combinaisons possibles de tickers et de dates
full_index = pd.MultiIndex.from_product([tickers, full_date_range], names=['stock_ticker', 'date'])
full_dataframe = pd.DataFrame(index=full_index).reset_index()

# Fusionner avec le dataset original pour combler les trous avec des NaN
merged_dataset = pd.merge(full_dataframe, filtered_dataset, on=['stock_ticker', 'date'], how='left')

print(merged_dataset.head)




#test
#liste des tickers à garder
tickers_to_keep = ['MSFT', 'INTC', 'TXN', 'MMM', 'PG']

#filtrer le dataframe pour ne garder que les tickers spécifiques
filtered_dataset = merged_dataset[merged_dataset['stock_ticker'].isin(tickers_to_keep)]




# Identifier le dernier enregistrement pour chaque ticker
last_entries = filtered_dataset.groupby('stock_ticker').tail(1)

# Filtrer les tickers dont 'stock_exret' est NaN dans la dernière entrée
tickers_to_drop = last_entries[last_entries['stock_exret'].isna()]['stock_ticker'].unique()
print(f"tickers drop: et n tickers {tickers_to_drop.shape} {tickers_to_drop}")
# Filtrer le dataset pour enlever les tickers correspondants
filtered_dataset = filtered_dataset[~filtered_dataset['stock_ticker'].isin(tickers_to_drop)]



#creer un multi-index avec 'stock_ticker' et 'date'
filtered_dataset.set_index(['stock_ticker', 'date'], inplace=True)
filtered_dataset = filtered_dataset.sort_index(level=['stock_ticker', 'date'])

# filtered_dataset.to_csv('test25.csv')



from sktime.forecasting.compose import DirRecTabularRegressionForecaster
from sklearn.ensemble import RandomForestRegressor
from sktime.forecasting.base import ForecastingHorizon
from sktime.split import temporal_train_test_split
from sktime.forecasting.arima import ARIMA
from sktime.performance_metrics.forecasting import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score
from sktime.forecasting.compose import make_reduction
from sktime.transformations.series.summarize import WindowSummarizer


#sélectionner la variable cible (par exemple, 'ret_eom')
y = filtered_dataset['stock_exret']


#sélectionner les variables eplicatives (exemple avec 'mspread', 'size_port', 'market_equity')
x = filtered_dataset[['mspread', 'market_equity', 'at_gr1', 'ni_be']]

y = pd.DataFrame(y)
x = pd.DataFrame(x)

# Supprimer les lignes avec des NaN dans y ou x
non_nan_index = y.dropna().index.intersection(x.dropna().index)  # Index sans NaN dans y et x



y_train, y_test, x_train, x_test = temporal_train_test_split(y, x)

x_test.to_csv('voirx.csv')
y_test.to_csv('voiry.csv')


kwargs = {
    "lag_feature": {
        "lag": [1],
        "mean": [[1, 3], [3, 6]],
        "std": [[1, 4]],
    }
}

summarizer = WindowSummarizer(**kwargs)

# Ajouter ces features aux variables explicatives x_train et x_test
x_train_augmented = summarizer.fit_transform(x_train)
x_test_augmented = summarizer.transform(x_test)


forecaster = make_reduction(
    RandomForestRegressor(n_estimators = 1000), window_length=20, strategy="recursive"
)

time_index = y_test.index.get_level_values(1)  # Le 2e niveau du MultiIndex correspond à la date
time_index = time_index.drop_duplicates()

# Créer un ForecastingHorizon à partir de cette partie temporelle
fh = ForecastingHorizon(time_index, is_relative=False)

forecaster.fit(y_train, X=x_train_augmented)

#y_pred = forecaster.predict(fh=fh, X=x_test_augmented)
y_pred = forecaster.predict(fh=fh, X=x_test_augmented)




#calculer les métriques de performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)  # Calcul du R² via scikit-learn

#afficher les résultats
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R^2: {r2}")
