import pandas as pd
from typing import Optional


class AddMoreVariables:
    """
    Classe pour enrichir un DataFrame de hackathon en le fusionnant avec des données macroéconomiques
    et des indicateurs d'options.
    """

    def __init__(self, hackathon_df: pd.DataFrame, freq: str = 'M'):
        """
        Initialise la classe avec les données de hackathon.

        Paramètres :
            hackathon_df (pd.DataFrame): Le DataFrame contenant les données du hackathon.
            freq (str): La fréquence pour ajuster l'index (par défaut 'M' pour fin de mois).
        """
        self.freq = freq
        self.hackathon_df = self._ensure_datetime_index(hackathon_df, self.freq)

    @staticmethod
    def _ensure_datetime_index(df: pd.DataFrame, freq: Optional[str] = None) -> pd.DataFrame:
        """
        Vérifie que l'index est un DatetimeIndex et ajuste la fréquence si nécessaire.

        Paramètres :
            df (pd.DataFrame): Le DataFrame à vérifier et convertir.
            freq (Optional[str]): La fréquence pour ajuster l'index ('M' pour mois, 'D' pour jours, etc.).

        Retourne :
            pd.DataFrame : DataFrame avec un index au format DatetimeIndex.
        """
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors='coerce')

        if df.index.hasnans:
            raise ValueError("Certains éléments de l'index n'ont pas pu être convertis en datetime.")

        if freq:
            df.index = df.index.to_period(freq).to_timestamp(freq)

        return df

    def add_macro_data(self, macro_data: pd.DataFrame) -> pd.DataFrame:
        """
        Fusionne les données macroéconomiques avec les données de hackathon sur l'index.

        Paramètres :
            macro_data (pd.DataFrame): Le DataFrame contenant les données macroéconomiques.

        Retourne :
            pd.DataFrame: DataFrame fusionné avec les variables explicatives ajoutées.
        """
        macro_data = self._ensure_datetime_index(macro_data, self.freq)

        merged_df = self.hackathon_df.merge(
            macro_data, left_index=True, right_index=True, how='left', suffixes=('', '_macro')
        )

        merged_df.index.name = 'date'

        return merged_df

    def add_option_indicators(self, hackathon_df: pd.DataFrame, option_indicators: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute les indicateurs d'options au DataFrame en fusionnant sur 'stock_ticker' et 'year_month'.

        Paramètres :
            hackathon_df (pd.DataFrame): Le DataFrame auquel ajouter les indicateurs d'options.
            option_indicators (pd.DataFrame): Le DataFrame contenant les indicateurs d'options.

        Retourne :
            pd.DataFrame: Le DataFrame avec les indicateurs d'options ajoutés.
        """
        if 'year_month' not in option_indicators.columns:
            raise KeyError("La colonne 'year_month' est manquante dans option_indicators.")

        df = hackathon_df.copy()
        original_index = df.index.copy()

        # Créer la colonne 'year_month' à partir de l'index
        df['year_month'] = df.index.to_period('M').strftime('%Y-%m')

        # Fusionner sur 'stock_ticker' et 'year_month'
        merged_df = df.merge(
            option_indicators, on=['stock_ticker', 'year_month'], how='left', suffixes=('', '_opt')
        )

        # Supprimer la colonne 'year_month' car elle n'est plus nécessaire après la fusion
        merged_df.drop(columns='year_month', inplace=True)

        # Réassigner l'index d'origine
        merged_df.index = original_index
        merged_df.index.name = 'date'

        return merged_df


if __name__ == '__main__':
    # Charger les données du hackathon
    hackathon_df = pd.read_csv(
        '../../data/intermediate_data/preprocess_data/hackathon_df_preprocessed.csv',
        index_col=0, parse_dates=True
    )

    # Charger les données macroéconomiques
    macro_data_scaled = pd.read_csv(
        '../../data/intermediate_data/preprocess_data/macro_data_scaled.csv',
        index_col=0, parse_dates=True
    )

    # Création d'une instance de AddMoreVariables
    add_more_variables = AddMoreVariables(hackathon_df)

    # Ajout des données macro
    merged_df = add_more_variables.add_macro_data(macro_data_scaled)

    # Charger les indicateurs d'options
    option_indicators = pd.read_csv(
        '../../data/intermediate_data/options_features/options_indicators.csv'
    )

    # Ajout des indicateurs d'options
    final_df = add_more_variables.add_option_indicators(merged_df, option_indicators)

    # Affichage des résultats
    print(final_df.head())
