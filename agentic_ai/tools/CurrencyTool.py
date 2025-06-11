from forex_python.converter import CurrencyRates, RatesNotAvailableError
from typing import Dict
from abc import ABC, abstractmethod
import time

class BaseTravelTool(ABC):
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        
    @abstractmethod
    async def execute(self, *args, **kwargs):
        pass

class CurrencyTool(BaseTravelTool):
    def __init__(self):
        super().__init__()
        self.c = CurrencyRates()
        
    async def execute(self, amount: float, from_currency: str, to_currency: str) -> Dict:
        """
        Convert currency using forex-python (free).
        """
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                rate = self.c.get_rate(from_currency, to_currency)
                converted = self.c.convert(from_currency, to_currency, amount)
                
                return {
                    'original_amount': amount,
                    'converted_amount': round(converted, 2),
                    'rate': round(rate, 4),
                    'from': from_currency,
                    'to': to_currency
                }
            except RatesNotAvailableError as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"Error: Could not fetch currency rates after {max_retries} attempts")
                    return {
                        'original_amount': amount,
                        'converted_amount': 'N/A',
                        'rate': 'N/A',
                        'from': from_currency,
                        'to': to_currency,
                        'error': 'Currency rates not available'
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
    currency = asyncio.run(currencyTool.execute(100, "USD", "INR"))
    print(f"Currency: {currency}")