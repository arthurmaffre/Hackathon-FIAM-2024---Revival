import pandas as pd
import numpy as np


class FromPricesToReturns(object):

    @staticmethod
    def _verify_return_type(return_type: str) -> str:
        """
        Verifies if the provided return type is valid.

        Parameters
        ----------
        return_type : str
            The return type to be verified.

        Returns
        -------
        str
            The verified return type.
        """

        valid_return_types = ['arithmetic', 'logarithmic']

        if return_type.lower() not in valid_return_types:
            available_return_types = ", ".join(f"'{key}'" for key in valid_return_types)
            raise ValueError(
                f"Invalid return type: '{return_type}'. Available return types are: {available_return_types}."
            )

        return return_type

    def compute_returns(self, data: pd.DataFrame | pd.Series, return_type: str = 'arithmetic', binarize: bool = False) -> pd.DataFrame:
        """
        Compute the returns of the price data.

        Parameters
        ----------
        data : pd.DataFrame | pd.Series
            The price data to compute returns from.
        return_type : str
            The method to compute returns ('arithmetic' or 'logarithmic').
        binarize : bool
            Whether to binarize the returns.
        fractional_differentiation : bool
            Whether to apply fractional differentiation.

        Returns
        -------
        pd.DataFrame
            The returns of the price data.
        """


        return_type = self._verify_return_type(return_type=return_type)

        if return_type == 'arithmetic':
            returns = data.pct_change(fill_method=None)
        else:
            returns = np.log(data).diff()

        if binarize:
            returns = self.binarize_returns(returns=returns)

        return returns[1:]

    @staticmethod
    def binarize_returns(returns: pd.DataFrame, threshold: float = 0) -> pd.DataFrame:
        """
        Binarize the returns of the price data.

        Parameters
        ----------
        returns : pd.DataFrame
            The returns of the price data.
        threshold : float
            The threshold to binarize the returns.

        Returns
        -------
        pd.DataFrame
            The binarized returns of the price data.
        """
        binarized_returns = returns.applymap(lambda x: 1 if x > threshold else 0).astype(int)

        return binarized_returns


if __name__ == '__main__':
    data = pd.DataFrame({
        'price': [10, 11, 12, 13, 14]
    })

    from_prices_to_returns = FromPricesToReturns()
    returns = from_prices_to_returns.compute_returns(data=data, return_type='arithmetic', binarize=False)
    print(returns)
    #    price
    # 1    0.1
    # 2    0.090909
    # 3    0.083333
    # 4    0.076923
    returns = from_prices_to_returns.compute_returns(data=data, return_type='logarithmic', binarize=False)
    print(returns)
    #     price
    # 1     0.095310
    # 2     0.087011
    # 3     0.080043
    # 4     0.074108

