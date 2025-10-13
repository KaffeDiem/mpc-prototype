from dataclasses import dataclass
import urllib.request
import urllib.error
import json
import logging
import os
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

    def turn_on(self) -> bool:
        """
        Turn the relay on.

        Returns:
            True if the relay is on after the command

        Raises:
            Exception: If the API request fails
        """
        url = f"http://{self.plug_ip}/relay/0?turn=on"
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "SmartPlugService/1.0")
            req.add_header("Connection", "close")

            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))
                is_on = data.get('ison', False)
                logger.info(f"Relay turned on: {is_on}")
                return is_on

        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            raise Exception(f"Failed to turn on relay: {e}")
        except (KeyError, json.JSONDecodeError, ValueError) as e:
            raise Exception(f"Invalid response format: {e}")

    def turn_off(self) -> bool:
        """
        Turn the relay off.

        Returns:
            True if the relay is off after the command (ison will be False)

        Raises:
            Exception: If the API request fails
        """
        url = f"http://{self.plug_ip}/relay/0?turn=off"
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "SmartPlugService/1.0")
            req.add_header("Connection", "close")

            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))
                is_on = data.get('ison', False)
                logger.info(f"Relay turned off: {is_on}")
                return not is_on

        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            raise Exception(f"Failed to turn off relay: {e}")
        except (KeyError, json.JSONDecodeError, ValueError) as e:
            raise Exception(f"Invalid response format: {e}")

    def get_status(self) -> PlugStatus:
        """
        Get the current status of the smart plug including power consumption.

        Returns:
            PlugStatus object with relay state, power consumption, and other metrics

        Raises:
            Exception: If the API request fails or returns invalid data
        """
        url = f"http://{self.plug_ip}/rpc/Switch.GetStatus?id=0"
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "SmartPlugService/1.0")
            req.add_header("Connection", "close")

            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))

                status = PlugStatus(
                    is_on=data.get('output', False),
                    power_watts=data.get('apower', 0.0),
                    voltage=data.get('voltage', 0.0),
                    current=data.get('current', 0.0),
                    temperature_c=data.get('temperature', {}).get('tC', 0.0)
                )

                logger.debug(f"Plug status: on={status.is_on}, power={status.power_watts}W")
                return status

        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            raise Exception(f"Failed to get plug status: {e}")
        except (KeyError, json.JSONDecodeError, ValueError) as e:
            raise Exception(f"Invalid response format: {e}")
