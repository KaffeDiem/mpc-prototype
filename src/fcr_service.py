from datetime import date, datetime, timedelta
from typing import List, Literal
from dataclasses import dataclass
import urllib.request
import urllib.error
import json
import logging
import numpy as np

class FCRService:
    def __init__(self):
        pass

    def get_fcr_prices(self) -> tuple[float, float]:
        """
        Get the historical FCR-D prices.
        It is a pay-as-clear market, so the prices are the ones that were actually paid.
        We choose to use the minimum prices, such that we are more likely to undershoot the price.

        Returns:
            tuple[float, float]: The minimum FCR-D down and up prices in EUR/kW.
        """
        url = f"https://api.energidataservice.dk/dataset/FcrNdDK2?limit=100"
        response = urllib.request.urlopen(url)
        data = json.loads(response.read())

        fcr_d_down_prices: np.ndarray = np.array([], dtype=float)
        fcr_d_up_prices: np.ndarray = np.array([], dtype=float)
        for r in data['records']:
            if r["AuctionType"] == "Total":
                if r["ProductName"] == "FCR-D ned":
                    fcr_d_down_prices = np.append(fcr_d_down_prices, r["PriceTotalEUR"])
                elif r["ProductName"] == "FCR-D upp":
                    fcr_d_up_prices = np.append(fcr_d_up_prices, r["PriceTotalEUR"])
        return fcr_d_down_prices[-1] / 1_000, fcr_d_up_prices[-1] / 1_000