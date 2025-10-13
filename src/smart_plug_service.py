from dataclasses import dataclass
import urllib.request
import urllib.error
import json
import logging
import os
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Load .env from the project root (parent of src directory)
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass

logger = logging.getLogger(__name__)

@dataclass
class PlugStatus:
    is_on: bool
    power_watts: float  # Current active power in watts
    voltage: float
    current: float
    temperature_c: float

class SmartPlugService:
    def __init__(self, plug_ip: str | None = None) -> None:
        """
        Initialize the SmartPlugService.

        Args:
            plug_ip: IP address of the smart plug. If not provided, will read from PLUG_IP environment variable.

        Raises:
            ValueError: If no plug IP is provided or found in environment
        """
        self.plug_ip = plug_ip or os.getenv("PLUG_IP")

        if not self.plug_ip:
            raise ValueError(
                "Plug IP is required. Provide it as plug_ip parameter or set PLUG_IP environment variable."
            )

    def _retry_request(self, url: str, method_name: str) -> dict | None:
        """
        Perform HTTP request with retry logic.
        
        Args:
            url: The URL to request
            method_name: Name of the calling method (for logging)
        
        Returns:
            JSON response as dict, or None if all retries failed
        """
        max_retries = 3
        retry_delay = 30  # seconds
        timeout = 30  # seconds
        
        for attempt in range(1, max_retries + 1):
            try:
                req = urllib.request.Request(url)
                req.add_header("User-Agent", "SmartPlugService/1.0")
                req.add_header("Connection", "close")
                
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    data = json.loads(response.read().decode('utf-8'))
                    return data
                    
            except (urllib.error.HTTPError, urllib.error.URLError) as e:
                logger.warning(
                    f"{method_name} failed (attempt {attempt}/{max_retries}): {e}"
                )
                if attempt < max_retries:
                    logger.info(f"Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                else:
                    logger.error(
                        f"{method_name} failed after {max_retries} attempts. Continuing without action."
                    )
                    return None
                    
            except (KeyError, json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    f"{method_name} received invalid response (attempt {attempt}/{max_retries}): {e}"
                )
                if attempt < max_retries:
                    logger.info(f"Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                else:
                    logger.error(
                        f"{method_name} failed after {max_retries} attempts. Continuing without action."
                    )
                    return None
        
        return None

    def turn_on(self) -> bool:
        """
        Turn the relay on.

        Returns:
            True if the relay is on after the command, False if failed after retries
        """
        url = f"http://{self.plug_ip}/relay/0?turn=on"
        data = self._retry_request(url, "turn_on")
        
        if data is None:
            logger.warning("Failed to turn on relay after all retries. Assuming OFF state.")
            return False
        
        is_on = data.get('ison', False)
        logger.info(f"Relay turned on: {is_on}")
        return is_on

    def turn_off(self) -> bool:
        """
        Turn the relay off.

        Returns:
            True if the relay is off after the command (ison will be False), False if failed after retries
        """
        url = f"http://{self.plug_ip}/relay/0?turn=off"
        data = self._retry_request(url, "turn_off")
        
        if data is None:
            logger.warning("Failed to turn off relay after all retries. Assuming OFF state.")
            return False
        
        is_on = data.get('ison', False)
        logger.info(f"Relay turned off: {is_on}")
        return not is_on

    def get_status(self) -> PlugStatus:
        """
        Get the current status of the smart plug including power consumption.

        Returns:
            PlugStatus object with relay state, power consumption, and other metrics.
            Returns safe defaults (OFF state, 0 power) if request fails after retries.
        """
        url = f"http://{self.plug_ip}/rpc/Switch.GetStatus?id=0"
        data = self._retry_request(url, "get_status")
        
        if data is None:
            logger.warning("Failed to get plug status after all retries. Returning safe defaults (OFF state).")
            return PlugStatus(
                is_on=False,
                power_watts=0.0,
                voltage=0.0,
                current=0.0,
                temperature_c=0.0
            )
        
        status = PlugStatus(
            is_on=data.get('output', False),
            power_watts=data.get('apower', 0.0),
            voltage=data.get('voltage', 0.0),
            current=data.get('current', 0.0),
            temperature_c=data.get('temperature', {}).get('tC', 0.0)
        )
        
        logger.debug(f"Plug status: on={status.is_on}, power={status.power_watts}W")
        return status
