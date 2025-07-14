from typing import Dict
from abc import ABC, abstractmethod
import time
from currency_converter import CurrencyConverter


class BaseTravelTool(ABC):
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        
    @abstractmethod
    async def execute(self, *args, **kwargs):
        pass

class CurrencyTool(BaseTravelTool):
    def __init__(self):
        super().__init__()
        self.converter = CurrencyConverter()
        
    async def execute(self, amount: float, from_currency: str, to_currency: str) -> Dict:
        """
        Convert currency using CurrencyConverter.
        """
        try:
            converted = self.converter.convert(amount, from_currency, to_currency)
            rate = self.converter.convert(1, from_currency, to_currency)
            
            return {
                'original_amount': amount,
                'converted_amount': round(converted, 2),
                'rate': round(rate, 4),
                'from': from_currency,
                'to': to_currency
            }
        except ValueError as e:
            print(f"Currency conversion error: {str(e)}")
            return {
                'original_amount': amount,
                'converted_amount': 'N/A',
                'rate': 'N/A',
                'from': from_currency,
                'to': to_currency,
                'error': str(e)
            }
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return {
                'original_amount': amount,
                'converted_amount': 'N/A',
                'rate': 'N/A',
                'from': from_currency,
                'to': to_currency,
                'error': str(e)
            }

if __name__ == "__main__":
    import asyncio
    currencyTool = CurrencyTool()
    currency = asyncio.run(currencyTool.execute(from_currency="MYR", to_currency="INR", amount=100))
    print(f"Currency: {currency}")