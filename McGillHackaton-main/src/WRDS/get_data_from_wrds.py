import wrds
import pandas as pd

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class WRDSDataAPI(object):
    def __init__(self) -> None:
        self.conn = wrds.Connection()

    def list_all_libraries(self) -> list:
        return self.conn.list_libraries()

    def list_datasets_in_library(self, library: str) -> list:
        return self.conn.list_tables(library=library)

    def get_sample_data(self, library: str, dataset: str, obs: int = 10, rows: int = -1,
                        offset: int = 0, columns: any = None, coerce_float: any = None, index_col: any = None,
                        date_cols: any = None) -> pd.DataFrame:
        return self.conn.get_table(library=library, table=dataset, obs=obs, rows=rows, offset=offset, columns=columns,
                                   coerce_float=coerce_float, index_col=index_col, date_cols=date_cols)

    def describe_table(self, library: str, dataset: str) -> pd.DataFrame:
        return self.conn.describe_table(library=library, table=dataset)

    def get_data_by_sql_query(self, sql_query: str, params: dict = None, coerce_float: bool = True, date_cols: any = None,
                              index_col: any = None, chunksize: int = 500000, return_iter: bool = False) -> pd.DataFrame:
        return self.conn.raw_sql(sql=sql_query, params=params, coerce_float=coerce_float, date_cols=date_cols,
                                 index_col=index_col, chunksize=chunksize, return_iter=return_iter)

    def get_data(self, library: str, dataset: str, rows: int = -1, obs: any = None,
                 offset: int = 0, columns: any = None, coerce_float: any = None, index_col: any = None,
                 date_cols: any = None) -> pd.DataFrame:
        return self.conn.get_table(library=library, table=dataset, rows=rows, obs=obs, offset=offset, columns=columns,
                                   coerce_float=coerce_float, index_col=index_col, date_cols=date_cols)

    def get_options_data_after_2010(self, tickers_csv_path: str, start_year: int = 2000, end_year: int = 2023,
                                             columns: list = None, max_days_until_maturity_to_keep: int = 30) -> pd.DataFrame:
        """
        Récupère les données d'options de plusieurs années et applique les filtres spécifiés.

        Paramètres
        ----------
        tickers_csv_path : str
            Chemin vers le fichier CSV contenant les tickers (dans la première colonne).
        start_year : int
            Année de début (par défaut 2000).
        end_year : int
            Année de fin (par défaut 2023).
        columns : list
            Liste des colonnes à sélectionner. Par défaut, les colonnes spécifiées sont utilisées.
        max_days_until_maturity_to_keep : int
            Nombre maximal de jours jusqu'à l'échéance à conserver (par défaut 30).

        Retourne
        -------
        pandas.DataFrame
            Le DataFrame contenant les données filtrées.
        """
        if columns is None:
            columns = ['secid', 'date', 'symbol', 'cp_flag', 'exdate', 'impl_volatility', 'strike_price', 'best_bid',
                       'best_offer', 'volume', 'open_interest', 'delta', 'ticker']

        # Ajouter 'underlying_symbol' et 'time_to_maturity' aux colonnes sélectionnées
        select_columns = columns + [
            "split_part(symbol, ' ', 1) AS underlying_symbol",
            "(exdate - date) AS time_to_maturity"
        ]

        # Lire les tickers depuis le fichier CSV avec index_col=0
        tickers_df = pd.read_csv(tickers_csv_path, index_col=0)
        tickers_list = tickers_df.dropna().iloc[:, 0].tolist()
        tickers_list = [str(ticker).strip().upper() for ticker in tickers_list if str(ticker).strip()]

        # Vérifier la taille de la liste des tickers
        max_identifiers_per_query = 100  # Limite ajustable en fonction des contraintes du serveur et de la longueur de la requête
        all_data = pd.DataFrame()

        total_chunks = (len(tickers_list) + max_identifiers_per_query - 1) // max_identifiers_per_query
        chunk_counter = 0

        # Diviser la liste des tickers si nécessaire
        for i in range(0, len(tickers_list), max_identifiers_per_query):
            chunk_counter += 1
            subset_tickers = tickers_list[i:i + max_identifiers_per_query]

            # Vérifier les tickers pour éviter l'injection SQL
            safe_tickers = [ticker.replace("'", "''") for ticker in subset_tickers]

            # Construire la chaîne de tickers pour la clause IN
            tickers_str = "'" + "', '".join(safe_tickers) + "'"

            # Construire la clause SELECT
            select_clause = ', '.join(select_columns)

            # Liste des années
            years = range(start_year, end_year + 1)

            # Afficher le progrès
            print(f"Traitement du chunk {chunk_counter}/{total_chunks} : {len(subset_tickers)} tickers.")

            # Construire la partie UNION ALL de la requête pour chaque année
            union_queries = []
            for year in years:
                table_name = f"optionm_all.opprcd{year}"
                query_part = f"""
                SELECT {select_clause}
                FROM {table_name}
                WHERE split_part(symbol, ' ', 1) IN ({tickers_str})
                """
                union_queries.append(query_part)

            # Joindre toutes les sous-requêtes avec UNION ALL
            sql_union = "\nUNION ALL\n".join(union_queries)

            # Construire la requête finale avec le filtre sur time_to_maturity
            sql_query = f"""
            SELECT * FROM (
                {sql_union}
            ) AS combined_data
            WHERE volume <> 0
              AND open_interest <> 0
              AND impl_volatility > 0
              AND secid IS NOT NULL
              AND date IS NOT NULL
              AND symbol IS NOT NULL
              AND exdate IS NOT NULL
              AND impl_volatility IS NOT NULL
              AND strike_price IS NOT NULL
              AND best_bid IS NOT NULL
              AND best_offer IS NOT NULL
              AND delta IS NOT NULL
              AND (exdate - date) <= {max_days_until_maturity_to_keep}
            """

            # Exécuter la requête et récupérer les données
            try:
                print(f"Exécution de la requête pour le chunk {chunk_counter}/{total_chunks}...")
                data_chunk = self.conn.raw_sql(sql_query)
                if not data_chunk.empty:
                    all_data = pd.concat([all_data, data_chunk], ignore_index=True)
                    print(f"Données récupérées pour le chunk {chunk_counter}/{total_chunks}: {len(data_chunk)} lignes.")
                    print(data_chunk.head())
                    print(data_chunk['underlying_symbol'].nunique())
                    print(data_chunk['date'].min(), data_chunk['date'].max())
                else:
                    print(f"Aucune donnée récupérée pour le chunk {chunk_counter}/{total_chunks}.")
            except Exception as e:
                print(f"Erreur lors de la récupération des données pour les tickers {subset_tickers}: {e}")
                continue

        print("Extraction terminée.")
        return all_data

    def get_options_data_before_2010(self, cusip_csv_path: str, start_year: int = 2000, end_year: int = 2010,
                                     columns: list = None, max_days_until_maturity_to_keep: int = 30) -> pd.DataFrame:
        """
        Récupère les données d'options avant 2010 et applique les filtres spécifiés en utilisant les cusips.

        Paramètres
        ----------
        cusip_csv_path : str
            Chemin vers le fichier CSV contenant les CUSIPs (dans la première colonne).
        start_year : int
            Année de début (par défaut 2000).
        end_year : int
            Année de fin (par défaut 2010).
        columns : list
            Liste des colonnes à sélectionner. Par défaut, les colonnes spécifiées sont utilisées.
        max_days_until_maturity_to_keep : int
            Nombre maximal de jours jusqu'à l'échéance à conserver (par défaut 30).

        Retourne
        -------
        pandas.DataFrame
            Le DataFrame contenant les données filtrées.
        """
        if columns is None:
            columns = ['secid', 'date', 'cusip', 'cp_flag', 'exdate', 'impl_volatility', 'strike_price', 'best_bid',
                       'best_offer', 'volume', 'open_interest', 'delta', 'ticker']

        # Ajouter 'underlying_cusip' et 'time_to_maturity' aux colonnes sélectionnées
        select_columns = columns + [
            "cusip AS underlying_cusip",
            "(exdate - date) AS time_to_maturity"
        ]

        # Lire les CUSIPs depuis le fichier CSV avec index_col=0
        cusip_df = pd.read_csv(cusip_csv_path, index_col=0)
        cusip_list = cusip_df.dropna().iloc[:, 0].tolist()
        cusip_list = [str(cusip).strip().upper() for cusip in cusip_list if str(cusip).strip()]

        # Vérifier la taille de la liste des cusips
        max_identifiers_per_query = 100  # Limite ajustable en fonction des contraintes du serveur et de la longueur de la requête
        all_data = pd.DataFrame()

        total_chunks = (len(cusip_list) + max_identifiers_per_query - 1) // max_identifiers_per_query
        chunk_counter = 0

        # Diviser la liste des cusips si nécessaire
        for i in range(0, len(cusip_list), max_identifiers_per_query):
            chunk_counter += 1
            subset_cusips = cusip_list[i:i + max_identifiers_per_query]

            # Vérifier les CUSIPs pour éviter l'injection SQL
            safe_cusips = [cusip.replace("'", "''") for cusip in subset_cusips]

            # Construire la chaîne de CUSIPs pour la clause IN
            cusips_str = "'" + "', '".join(safe_cusips) + "'"

            # Construire la clause SELECT
            select_clause = ', '.join(select_columns)

            # Liste des années
            years = range(start_year, end_year + 1)

            # Afficher le progrès
            print(f"Traitement du chunk {chunk_counter}/{total_chunks} : {len(subset_cusips)} cusips.")

            # Construire la partie UNION ALL de la requête pour chaque année
            union_queries = []
            for year in years:
                table_name = f"optionm_all.opprcd{year}"
                query_part = f"""
                SELECT {select_clause}
                FROM {table_name}
                WHERE cusip IN ({cusips_str})
                """
                union_queries.append(query_part)

            # Joindre toutes les sous-requêtes avec UNION ALL
            sql_union = "\nUNION ALL\n".join(union_queries)

            # Construire la requête finale avec le filtre sur time_to_maturity
            sql_query = f"""
            SELECT * FROM (
                {sql_union}
            ) AS combined_data
            WHERE volume <> 0
              AND open_interest <> 0
              AND impl_volatility > 0
              AND secid IS NOT NULL
              AND date IS NOT NULL
              AND cusip IS NOT NULL
              AND exdate IS NOT NULL
              AND impl_volatility IS NOT NULL
              AND strike_price IS NOT NULL
              AND best_bid IS NOT NULL
              AND best_offer IS NOT NULL
              AND delta IS NOT NULL
              AND (exdate - date) <= {max_days_until_maturity_to_keep}
            """

            # Exécuter la requête et récupérer les données
            try:
                print(f"Exécution de la requête pour le chunk {chunk_counter}/{total_chunks}...")
                data_chunk = self.conn.raw_sql(sql_query)
                if not data_chunk.empty:
                    all_data = pd.concat([all_data, data_chunk], ignore_index=True)
                    print(f"Données récupérées pour le chunk {chunk_counter}/{total_chunks}: {len(data_chunk)} lignes.")
                    print(data_chunk.head())
                else:
                    print(f"Aucune donnée récupérée pour le chunk {chunk_counter}/{total_chunks}.")
            except Exception as e:
                print(f"Erreur lors de la récupération des données pour les cusips {subset_cusips}: {e}")
                continue

        print("Extraction terminée.")
        return all_data

    def get_stock_prices(self, permno_csv_path: str, start_date: str = '2000-01-01',
                         end_date: str = '2023-12-31', columns: list = None) -> pd.DataFrame:
        """
        Récupère les prix des actions pour les permno spécifiés entre deux dates.

        Paramètres
        ----------
        permno_csv_path : str
            Chemin vers le fichier CSV contenant les permno (dans la première colonne).
        start_date : str
            Date de début au format 'YYYY-MM-DD' (par défaut '2000-01-01').
        end_date : str
            Date de fin au format 'YYYY-MM-DD' (par défaut '2023-12-31').
        columns : list
            Liste des colonnes à sélectionner. Par défaut, ['date', 'permno', 'cusip', 'prc', 'ret'].

        Retourne
        -------
        pandas.DataFrame
            Le DataFrame contenant les prix des actions.
        """
        if columns is None:
            columns = ['date', 'permno', 'cusip', 'prc', 'ret']

        # Lire les permno depuis le fichier CSV
        permno_df = pd.read_csv(permno_csv_path, index_col=0)
        permno_list = permno_df.iloc[:, 0].dropna().tolist()
        permno_list = [str(permno).strip() for permno in permno_list if str(permno).strip()]

        # Vérifier la taille de la liste des permno
        max_identifiers_per_query = 100  # Nombre maximal de permno par requête
        all_data = pd.DataFrame()

        total_chunks = (len(permno_list) + max_identifiers_per_query - 1) // max_identifiers_per_query
        chunk_counter = 0

        for i in range(0, len(permno_list), max_identifiers_per_query):
            chunk_counter += 1
            subset_permno = permno_list[i:i + max_identifiers_per_query]

            # Préparer les paramètres pour la requête SQL
            params = {
                'permno': tuple(subset_permno),
                'start_date': start_date,
                'end_date': end_date
            }

            # Construire la clause SELECT
            select_clause = ', '.join(columns)

            # Construire la requête SQL avec permno
            sql_query = f"""
            SELECT {select_clause}
            FROM crsp.dsf
            WHERE permno IN %(permno)s
              AND date BETWEEN %(start_date)s AND %(end_date)s
            """

            print(f"Exécution de la requête pour le chunk {chunk_counter}/{total_chunks}...")

            try:
                data_chunk = self.conn.raw_sql(sql_query, params=params)
                if not data_chunk.empty:
                    all_data = pd.concat([all_data, data_chunk], ignore_index=True)
                    print(f"Données récupérées pour le chunk {chunk_counter}/{total_chunks}: {len(data_chunk)} lignes.")
                    print(data_chunk.head())
                    print(data_chunk['permno'].nunique())
                    print(data_chunk['date'].min(), data_chunk['date'].max())
                else:
                    print(f"Aucune donnée récupérée pour le chunk {chunk_counter}/{total_chunks}.")
            except Exception as e:
                print(f"Erreur lors de la récupération des données pour les permno {subset_permno}: {e}")
                continue

        print("Extraction des prix des actions terminée.")
        return all_data


if __name__ == '__main__':

    wrds_data = WRDSDataAPI()
    print(wrds_data.list_all_libraries())
    print(wrds_data.list_datasets_in_library(library='crsp'))
    print(wrds_data.describe_table(library='crsp', dataset='dsf'))
    print(wrds_data.get_sample_data(library='crsp', dataset='dsf', obs=10))

    # Les prix d'options sont contenus dans la librairie 'optionm_all' ainsi que dans les datasets qui commencent
    # par 'opprcd' et qui sont suivis de l'année de la donnée. Par exemple, 'opprcd2020' contient les prix d'options
    # de l'année 2020.

    print(wrds_data.list_datasets_in_library(library='optionm_all'))
    print(wrds_data.describe_table(library='optionm_all', dataset='opprcd2011'))
    print(wrds_data.get_sample_data(library='optionm_all', dataset='opprcd2011', obs=10))
    print(wrds_data.get_sample_data(library='optionm_all', dataset='opprcd2000', obs=10))

    # Chemin vers votre fichier CSV de tickers
    tickers_csv_path = '../../data/intermediate_data/stock_ticker.csv'
    permno_csv_path = '../../data/raw_data/permno.csv'
    cusip_csv_path = '../../data/intermediate_data/cusip.csv'

    #--------------------------------------------STOCK PRICES---------------------------------------------------#

    # # Récupérer les prix des actions
    # data_prices = wrds_data.get_stock_prices(
    #     permno_csv_path=permno_csv_path,
    #     start_date='2000-01-01',
    #     end_date='2023-12-31'
    # )
    #
    # # Afficher un échantillon des prix des actions
    # print(data_prices.head())
    # print(data_prices.info())
    # print(data_prices.tail())
    # print(data_prices.shape)
    #
    # # Enregistrer les prix des actions au format parquet
    # data_prices.to_parquet('../../data/raw_data/stock_prices.parquet')

    #--------------------------------------------OPTIONS DATA---------------------------------------------------#

    # Spécifier le nombre maximal de jours jusqu'à l'échéance à conserver
    max_days = 30  # Par exemple, 30 jours

    # Récupérer les données d'options de 2022 à 2022
    options_after_2010 = wrds_data.get_options_data_after_2010(
        tickers_csv_path=tickers_csv_path,
        start_year=2011,
        end_year=2023,
        max_days_until_maturity_to_keep=max_days
    )

    options_before_2010 = wrds_data.get_options_data_before_2010(
        cusip_csv_path=cusip_csv_path,
        start_year=2000,
        end_year=2010,
        max_days_until_maturity_to_keep=max_days
    )

    # Afficher un échantillon des données
    print(options_after_2010.head())
    print(options_after_2010.info())
    print(options_after_2010.tail())
    print(options_after_2010.shape)

    print(options_before_2010.head())
    print(options_before_2010.info())
    print(options_before_2010.tail())
    print(options_before_2010.shape)

    # Enregistrer les données au format parquet
    options_after_2010.to_parquet('../../data/raw_data/options_data_2011_2023.parquet')
    options_before_2010.to_parquet('../../data/raw_data/options_data_2000_2010.parquet')


