"""
PyTaifex - Python wrapper for Taiwan Futures Exchange (TAIFEX) official TTB trading API.

This module provides a Python interface for interacting with the TAIFEX trading system
through the official TTB trading API. It includes classes for handling market data,
orders, positions, and account information.

Key Features:
- Real-time market data subscription
- Order management (create, modify, cancel)
- Position and account queries
- Comprehensive error handling
- Multi-process architecture for stability
"""

import importlib.util
import logging
import multiprocessing
import os
import queue
import sys
import threading
from enum import Enum
from logging.handlers import QueueHandler
from typing import Any, Callable


# Errors
class PyTaifexError(Exception):
    """Base exception for all PyTaifex errors."""

    pass


class SubscribeError(PyTaifexError):
    """Raised when subscription to market data fails."""

    pass


class OrderError(PyTaifexError):
    """Base class for order-related errors."""

    pass


class OrderCreationError(OrderError):
    """Raised when order creation fails."""

    pass


class OrderModificationError(OrderError):
    """Raised when order modification (price/quantity change) fails."""

    pass


class OrderCancellationError(OrderError):
    """Raised when order cancellation fails."""

    pass


class OrderQueryError(OrderError):
    """Raised when querying orders fails."""

    pass


class PositionQueryError(PyTaifexError):
    """Raised when querying positions fails."""

    pass


class AccountQueryError(PyTaifexError):
    """Raised when querying account information fails."""

    pass


class TTBConnectionError(PyTaifexError):
    """Raised when connection to TTB fails."""

    pass


class TTBTimeoutError(PyTaifexError):
    """Raised when operations timeout."""

    pass


class ValidationError(PyTaifexError):
    """Raised when input validation fails."""

    pass


# Enums
class TimeInForce(Enum):
    """Order time-in-force types for TAIFEX trading.

    Attributes:
        ROD: Rest of Day - Order remains active until end of trading day
        IOC: Immediate or Cancel - Execute immediately or cancel unfilled portion
        FOK: Fill or Kill - Execute completely or cancel entire order
    """

    ROD = "1"
    IOC = "2"
    FOK = "3"


class OrderSide(Enum):
    """Order side enumeration for buy/sell operations.

    Attributes:
        BUY: Buy order (long position)
        SELL: Sell order (short position)
    """

    BUY = "1"
    SELL = "2"


class OrderType(Enum):
    """Order type enumeration for market/limit orders.

    Attributes:
        MARKET: Market order - execute at current market price
        LIMIT: Limit order - execute at specified price or better
    """

    MARKET = "1"
    LIMIT = "2"


# Class
class QuoteData:
    """Represents real-time market quote data for a financial instrument.

    This class encapsulates market data received from the TAIFEX trading system,
    including price information, volume, bid/ask data, and timing information.

    Attributes:
        symbol (str): The trading symbol/instrument identifier
        name (str): The full name of the instrument
        open_ref (str): Closing price of last day.
        open_price (str): Opening price of today.
        high_price (str): Highest price of today.
        low_price (str): Lowest price of today.
        deno (float): Unknown, possibly related to price denomination
        price (str): Current/last traded price
        qty (str): Traded quantity
        change_price (str): Price change from previous data
        change_ratio (str): Percentage change from previous data
        bid_ps (str): Best bid price
        bid_pv (str): Best bid volume
        ask_ps (str): Best ask price
        ask_pv (str): Best ask volume
        tick_time (str): Timestamp of the quote data
        volume (str): Total trading volume
    """

    def __init__(self, data: dict):
        """Initialize QuoteData from a dictionary of market data.

        Args:
            data (dict): Dictionary containing market data fields from official TTB API
        """
        self.symbol: str = data.get("Symbol", "")
        self.name: str = data.get("Name", "")
        self.open_ref: str = data.get("OpenRef", "")
        self.open_price: str = data.get("OpenPrice", "")
        self.high_price: str = data.get("HighPrice", "")
        self.low_price: str = data.get("LowhPrice", "")
        self.deno: float = data.get("Deno", 0.0)
        self.price: str = data.get("Price", "")
        self.qty: str = data.get("Qty", "")
        self.change_price: str = data.get("Change", "")
        self.change_ratio: str = data.get("ChangeRatio", "")
        self.bid_ps: str = data.get("BidPs", "")
        self.bid_pv: str = data.get("BidPv", "")
        self.ask_ps: str = data.get("AskPs", "")
        self.ask_pv: str = data.get("AskPv", "")
        self.tick_time: str = data.get("TickTime", "")
        self.volume: str = data.get("Volume", "")

    def to_dict(self):
        """Convert QuoteData to dictionary format.

        Returns:
            dict: Dictionary representation of the quote data
        """
        return {
            "symbol": self.symbol,
            "name": self.name,
            "open_ref": self.open_ref,
            "open_price": self.open_price,
            "high_price": self.high_price,
            "low_price": self.low_price,
            "deno": self.deno,
            "price": self.price,
            "qty": self.qty,
            "change_price": self.change_price,
            "change_ratio": self.change_ratio,
            "bid_ps": self.bid_ps,
            "bid_pv": self.bid_pv,
            "ask_ps": self.ask_ps,
            "ask_pv": self.ask_pv,
            "tick_time": self.tick_time,
            "volume": self.volume,
        }

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"QuoteData({self.symbol}, {self.tick_time})"

    def __eq__(self, another):
        """Check equality based on symbol and tick time."""
        return self.symbol == another.symbol and self.tick_time == another.tick_time


class OrderData:
    """Represents order information from the TAIFEX trading system.

    This class encapsulates order data including order details, execution status,
    pricing information, and timing data.

    Attributes:
        order_number (str): Unique order number assigned by the system
        symbol_id (str): Trading symbol/instrument identifier
        symbol_name (str): Full name of the trading instrument
        status (str): Current order status (e.g., pending, filled, cancelled, but in Chinese)
        side (pytaifex.OrderSide): Order side (BUY or SELL)
        ofst_id (str): Unknown.
        order_price (str): Order price specified when placing the order
        order_qty (str): Total order quantity
        pending_qty (str): Remaining unfilled quantity
        filled_qty (str): Total filled quantity
        filled_price (str): Average filled price
        order_id (str): Internal order identifier
        order_date (str): Date when order was placed
        order_time (str): Time when order was placed
    """

    def __init__(self, order_dict: dict):
        """Initialize OrderData from a dictionary of order information.

        Args:
            order_dict (dict): Dictionary containing order data fields from official TTB API

        Raises:
            ValueError: If order side is not recognized
        """
        self.order_number: str = order_dict.get("ORDNO", "")
        self.symbol_id: str = order_dict.get("COMD_ID", "")
        self.symbol_name: str = order_dict.get("COMD_NAME", "")
        self.status: str = order_dict.get("STAT", "")
        if order_dict.get("BS", "") == "買別":
            self.side = OrderSide.BUY
        elif order_dict.get("BS", "") == "賣別":
            self.side = OrderSide.SELL
        else:
            raise ValueError(f"Unknown order side: {order_dict.get('BS', '')}")
        self.ofst_id: str = order_dict.get("OFST_ID", "")
        self.order_price: str = order_dict.get("ORDR_PRCE", "")
        self.order_qty: str = order_dict.get("VOLM", "")
        self.pending_qty: str = order_dict.get("LESS_VOLM", "")
        self.filled_qty: str = order_dict.get("DEAL_TOTL", "")
        self.filled_price: str = order_dict.get("DEAL_PRCE", "")
        self.order_id: str = order_dict.get("ORDR_ID", "")
        self.order_date: str = order_dict.get("ORDDT", "")
        self.order_time: str = order_dict.get("ORDTM", "")

    def to_dict(self):
        """Convert OrderData to dictionary format.

        Returns:
            dict: Dictionary representation of the order data
        """
        return {
            "order_number": self.order_number,
            "symbol_id": self.symbol_id,
            "symbol_name": self.symbol_name,
            "status": self.status,
            "side": "買別" if self.side.value == "1" else "賣別",
            "ofst_id": self.ofst_id,
            "order_price": self.order_price,
            "order_qty": self.order_qty,
            "pending_qty": self.pending_qty,
            "filled_qty": self.filled_qty,
            "filled_price": self.filled_price,
            "order_id": self.order_id,
            "order_date": self.order_date,
            "order_time": self.order_time,
        }

    def __str__(self):
        return f"OrderData({self.order_number}, {self.symbol_id}, {self.status}, {self.side}, {self.order_time})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, another):
        """Check equality based on order number and order ID."""
        return self.order_number == another.order_number and self.order_id == another.order_id


class PositionData:
    """Represents position information from the TAIFEX trading system.

    This class encapsulates position data including deal information, instrument details,
    profit/loss calculations, and trading metadata. Supports both single-leg and spread positions.

    Attributes:
        deal_id (str): Unique deal/position identifier
        order_kind (str): Unknown, possibly related to original order type
        symbol1_id (str): Primary instrument symbol identifier
        side1 (pytaifex.OrderSide): Position side for primary instrument (BUY/SELL)
        type1 (str): Type of primary instrument
        symbol2_id (str): Secondary instrument symbol (for spreads)
        side2 (pytaifex.OrderSide): Position side for secondary instrument (for spreads)
        type2 (str): Type of secondary instrument (for spreads)
        hold (str): Position holding quantity
        deal_price (str): Execution price
        settle_price (str): Current settlement price
        floating_profit_loss (str): Unrealized profit/loss
        currency (str): Currency code (ISO 4217)
        symbol1_name (str): Full name of primary instrument
        symbol2_name (str): Full name of secondary instrument (for spreads)
        trade_hour (str): Trading hour when position was established
        trade_day (str): Trading day when position was established
    """

    def __init__(self, position_dict: dict):
        """Initialize PositionData from a dictionary of position information.

        Args:
            position_dict (dict): Dictionary containing position data fields from TTB API
        """
        self.deal_id: str = position_dict.get("DealID", "")
        self.order_kind: str = position_dict.get("OrderKind", "")
        self.symbol1_id: str = position_dict.get("ComdID1", "")
        self.side1: OrderSide = OrderSide.BUY if position_dict.get("ComdBS1", "") == "B" else OrderSide.SELL
        self.type1: str = position_dict.get("ComdType1", "")
        self.symbol2_id: str = position_dict.get("ComdID2", "")
        self.side2: OrderSide = OrderSide.BUY if position_dict.get("ComdBS2", "") == "B" else OrderSide.SELL
        self.type2: str = position_dict.get("ComdType2", "")
        self.hold: str = position_dict.get("Hold", "")
        self.deal_price: str = position_dict.get("DealPrice", "")
        self.settle_price: str = position_dict.get("SettlePrice", "")
        self.floating_profit_loss: str = position_dict.get("FloatingProfitLoss", "")
        self.currency: str = position_dict.get("Curr4217", "")
        self.symbol1_name: str = position_dict.get("ComdName1", "")
        self.symbol2_name: str = position_dict.get("ComdName2", "")
        self.trade_hour: str = position_dict.get("TradeHour", "")
        self.trade_day: str = position_dict.get("TradeDay", "")

    def to_dict(self):
        """Convert PositionData to dictionary format.

        Returns:
            dict: Dictionary representation of the position data
        """
        return {
            "deal_id": self.deal_id,
            "order_kind": self.order_kind,
            "symbol1_id": self.symbol1_id,
            "side1": "買別" if self.side1.value == "1" else "賣別",
            "type1": self.type1,
            "symbol2_id": self.symbol2_id,
            "side2": "買別" if self.side2.value == "1" else "賣別",
            "type2": self.type2,
            "hold": self.hold,
            "deal_price": self.deal_price,
            "settle_price": self.settle_price,
            "floating_profit_loss": self.floating_profit_loss,
            "currency": self.currency,
            "symbol1_name": self.symbol1_name,
            "symbol2_name": self.symbol2_name,
            "trade_hour": self.trade_hour,
            "trade_day": self.trade_day,
        }

    def __str__(self):
        return f"PositionData({self.deal_id}, {self.symbol1_id}, {self.symbol2_id}, {self.trade_day})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, another):
        """Check equality based on deal ID."""
        return self.deal_id == another.deal_id


def _load_pyc_internal(pyc_file_path: str, logger: logging.Logger):
    """Utility function to Load TTB pyc module in worker process.

    This function dynamically loads the official TTB compiled Python module from a .pyc file.
    It performs validation, creates module specifications, and executes
    the module code in the current process.

    Args:
        pyc_file_path (str): Path to the TTB .pyc file
        logger (logging.Logger): Logger instance for error reporting

    Returns:
        module: The loaded TTB module instance

    Raises:
        Exception: If .pyc file is not found or doesn't have .pyc extension
        ImportError: If module creation or loading fails
    """
    logging.info(f"Loading pyc file: {pyc_file_path}")

    if not os.path.exists(pyc_file_path):
        logger.error(f".pyc file not found at '{pyc_file_path}'")
        raise Exception(f".pyc file not found at '{pyc_file_path}'")
    if not pyc_file_path.endswith(".pyc"):
        logger.error(f"The provided file '{pyc_file_path}' does not have a .pyc extension.")
        raise Exception(f"The provided file '{pyc_file_path}' does not have a .pyc extension.")

    module_name = "TTBHelp"

    try:
        # 1. Create a module spec from the .pyc file
        spec = importlib.util.spec_from_file_location(module_name, pyc_file_path)

        if spec is None:
            logger.error(
                f"Could not create module spec for TTBHelp from '{pyc_file_path}'. "
                + "Check Python version compatibility."
            )
            raise ImportError(
                f"Could not create module spec for TTBHelp from '{pyc_file_path}'. "
                + "Check Python version compatibility."
            )

        # 2. Create a module from the spec
        ttb_module_internal = importlib.util.module_from_spec(spec)

        if ttb_module_internal is None:
            logger.error("Error: Could not create module from spec for TTBHelp.")
            raise ImportError("Error: Could not create module from spec for TTBHelp.")

        # 3. Add the module to sys.modules, so it can be imported from other places
        sys.modules[module_name] = ttb_module_internal

        # 4. Execute the module's code
        if spec.loader:
            spec.loader.exec_module(ttb_module_internal)
            logger.info(f"Successfully loaded TTBHelp from '{pyc_file_path}'")
            return ttb_module_internal
        else:
            logger.error("Error: No loader found in spec for TTBHelp. Cannot execute module.")
            if module_name in sys.modules:
                del sys.modules[module_name]
            raise ImportError("Error: No loader found in spec for TTBHelp. Cannot execute module.")

    except Exception as e:
        logger.error(f"An unexpected error occurred while loading TTBHelp from '{pyc_file_path}': {e}")
        if module_name in sys.modules:
            del sys.modules[module_name]
        raise


def _ttb_worker_function(
    pyc_file_path: str,
    host: str,
    zmq_port: int,
    data_q_out: multiprocessing.Queue,
    control_q_in: multiprocessing.Queue,
    response_q_out: multiprocessing.Queue,
    log_q_for_main_process: multiprocessing.Queue,
    parent_logger_name: str,
):
    """Worker function for TTB operations in a separate process.

    This function runs in a separate process and handles all TTB operations including
    market data subscription, order management, and API communication. It communicates
    with the main process through queues for commands, responses, and data.

    Args:
        pyc_file_path (str): Path to the TTB .pyc module file
        host (str): TTB server host URL
        zmq_port (int): ZeroMQ port for TTB communication
        data_q_out (multiprocessing.Queue): Queue for sending market data to main process
        control_q_in (multiprocessing.Queue): Queue for receiving commands from main process
        response_q_out (multiprocessing.Queue): Queue for sending API responses to main process
        log_q_for_main_process (multiprocessing.Queue): Queue for sending log messages to main process
        parent_logger_name (str): Name of the parent logger for log hierarchy
    """
    worker_logger = logging.getLogger(f"{parent_logger_name}.TTBWorker")

    for h in worker_logger.handlers[:]:  # clear alll exists handlers
        worker_logger.removeHandler(h)

    queue_log_handler = QueueHandler(log_q_for_main_process)
    worker_logger.addHandler(queue_log_handler)
    worker_logger.propagate = False

    # send all logs to main process
    # the actual log filter is in main process
    worker_logger.setLevel(logging.DEBUG)
    worker_logger.info(f"TTB Worker started (PID: {os.getpid()}), logging to main process.")

    _ttb_instance = None
    try:
        ttb_module_loaded = _load_pyc_internal(pyc_file_path, worker_logger)

        class TTBProcessInternal(ttb_module_loaded.TTBModule):
            def __init__(
                self,
                host: str,
                zmq_port: int,
                output_queue: multiprocessing.Queue,
                process_logger: logging.Logger,
            ):
                super().__init__(host, zmq_port)
                self._output_queue = output_queue
                self._process_logger = process_logger
                self._process_logger.info(f"TTBProcessInternal initialized with host: {host}, zmq_port: {zmq_port}")

            def SHOWQUOTEDATA(self, obj: Any):  # noqa: N802, special method name in pyc file
                self._process_logger.debug(f"SHOWQUOTEDATA called, data: {obj}")
                try:
                    self._output_queue.put(obj)
                except Exception as e:
                    self._process_logger.error(f"Error during SHOWQUOTEDATA putting data in queue: {e}", exc_info=True)

        _ttb_instance = TTBProcessInternal(host, zmq_port, data_q_out, worker_logger)
        worker_logger.info("Internal TTB instance initialized and running.")
        data_q_out.put(
            {
                "type": "info",
                "source": "worker_initialization_runtime",
                "message": "Internal TTB instance initialized and running.",
                "success": True,
            }
        )

        running = True
        while running:
            try:
                command_dict = control_q_in.get(timeout=0.1)
                if not isinstance(command_dict, dict) or "command" not in command_dict:
                    worker_logger.error(f"Invalid command received: {command_dict}")
                    continue
                if command_dict.get("command") == "shutdown":
                    worker_logger.info("Received shutdown command, exiting TTB worker.")
                    running = False
                elif command_dict.get("command") == "subscribe":
                    symbols = command_dict.get("symbols", [])
                    worker_logger.info(f"Received subscribe command for symbol: {', '.join(symbols)}")
                    resp = _ttb_instance.QUOTEDATA(",".join(symbols))
                    response_q_out.put(resp)
                elif command_dict.get("command") == "create_order":
                    order_dict = command_dict.get("order_dict", {})
                    worker_logger.info(f"Received create order command: {order_dict}")
                    resp = _ttb_instance.NEWORDER(order_dict)
                    response_q_out.put(resp)
                elif command_dict.get("command") == "change_price":
                    order_dict = command_dict.get("order_dict", {})
                    worker_logger.info("Received change price command.")
                    resp = _ttb_instance.REPLACEPRICE(command_dict.get("order_dict", {}))
                    response_q_out.put(resp)
                elif command_dict.get("command") == "change_qty":
                    order_dict = command_dict.get("order_dict", {})
                    worker_logger.info("Received change qty command.")
                    resp = _ttb_instance.REPLACEQTY(command_dict.get("order_dict", {}))
                    response_q_out.put(resp)
                elif command_dict.get("command") == "get_orders":
                    worker_logger.info("Received get orders command.")
                    resp = _ttb_instance.QUERYRESTOREREPORT()
                    response_q_out.put(resp)
                elif command_dict.get("command") == "cancel_order":
                    order_dict = command_dict.get("order_dict", {})
                    worker_logger.info("Received cancel order command.")
                    resp = _ttb_instance.CANCELORDER(command_dict.get("order_dict", {}))
                    response_q_out.put(resp)
                elif command_dict.get("command") == "get_positions":
                    worker_logger.info("Received get positions command.")
                    resp = _ttb_instance.QUERYRESTOREFILLREPORT()
                    response_q_out.put(resp)
                elif command_dict.get("command") == "get_accounts":
                    worker_logger.info("Received get accounts command.")
                    resp = _ttb_instance.QUERYMARGIN()
                    response_q_out.put(resp)
                else:
                    worker_logger.error(f"Unknown command received: {command_dict}")
            except queue.Empty:
                pass
            except Exception as e:
                worker_logger.error(f"Error in TTB worker while processing command: {e}", exc_info=True)
                try:
                    data_q_out.put(
                        {
                            "type": "error",
                            "source": "worker_control_loop",
                            "message": f"worker process queue error: {e!s}",
                        }
                    )
                except Exception as q_err:
                    worker_logger.error(f"Error putting error message in queue: {q_err}")

                running = False

            if not running:
                break

    except Exception as e:
        worker_logger.critical(f"Serious error in TTB worker: {e}", exc_info=True)
        try:
            data_q_out.put(
                {
                    "type": "critical_error",
                    "source": "worker_initialization_runtime",
                    "message": str(e),
                    "details": repr(e),
                }
            )
        except Exception as q_err:
            worker_logger.error(f"Error putting critical error message in queue: {q_err}", exc_info=True)

    finally:
        worker_logger.info("Closing")
        logging.shutdown()


class TTB:
    """Main interface for TTB API operations.

    This class provides a high-level Python interface for interacting with the TAIFEX
    trading system through the official TTB API. It manages market data subscriptions, order
    operations, position queries, and account information retrieval using a multi-process
    architecture for stability and performance.

    The class uses separate processes for TTB operations to isolate potential crashes
    and ensure the main application remains stable. Communication between processes
    is handled through queues for commands, responses, and real-time data.

    Key Features:
    - Real-time market data subscription with callback support
    - Complete order lifecycle management (create, modify, cancel)
    - Position and account information queries
    - Robust error handling with specific exception types
    - Multi-process architecture for stability
    - Comprehensive logging support
    - Context manager support for resource cleanup

    Example:
        ```python
        from pytaifex import TTB, QuoteData, OrderSide, TimeInForce, OrderType

        def on_quote(data: QuoteData):
            print(f"Received quote: {data}, symbol: {data.symbol}")
            print(f"{data.to_dict()}")

        with TTB("local/TTBHelp.pyc") as client:
            client.register_quote_callback(on_quote)
            client.subscribe(["TXFF5"])
            client.create_order("TXFF5", OrderSide.BUY, "17000",
                               TimeInForce.ROD, OrderType.LIMIT, "1", False)
        ```

    Attributes:
        logger (logging.Logger): Logger instance for this TTB instance
        pyc_file_path (str): Path to the TTB .pyc module file
        host (str): TTB server host URL
        zmq_port (int): ZeroMQ port for TTB communication
    """

    def __init__(
        self,
        pyc_file_path: str,
        host: str = "http://localhost:8080",
        zmq_port: int = 51141,
        logger: logging.Logger | None = None,
        timeout: int = 5,
    ):
        """Initialize TTB wrapper with connection parameters.

        Args:
            pyc_file_path (str): Path to the TTB .pyc module file
            host (str, optional): TTB server host URL. Defaults to "http://localhost:8080"
            zmq_port (int, optional): ZeroMQ port for TTB communication. Defaults to 51141
            logger (logging.Logger, optional): Custom logger instance. If None, creates default logger
            timeout (int, optional): Timeout in seconds for worker initialization. Defaults to 5

        Raises:
            pytaifex.TTBTimeoutError: If worker process fails to initialize within timeout
            Exception: If TTB module loading or initialization fails
        """
        if logger is not None:
            self.logger = logger
        else:
            # Create a logger if not given
            self.logger = logging.getLogger(self.__class__.__name__ + f"_{id(self)}")
            if not self.logger.handlers:
                default_handler = logging.StreamHandler()
                formatter = logging.Formatter("%(levelname)s [%(name)s]: %(message)s")
                default_handler.setFormatter(formatter)
                default_handler.setLevel(logging.INFO)
                self.logger.setLevel(logging.INFO)
                self.logger.addHandler(default_handler)

        self.pyc_file_path = pyc_file_path
        self.host = host
        self.zmq_port = zmq_port

        self.__quote_callbacks: list[Callable[[Any], None]] = []
        self.__data_queue: multiprocessing.Queue = multiprocessing.Queue()
        self.__control_queue: multiprocessing.Queue = multiprocessing.Queue()
        self.__response_queue: multiprocessing.Queue = multiprocessing.Queue()

        self.__log_processing_queue: multiprocessing.Queue = multiprocessing.Queue(-1)  # infinite size
        self.__log_listener_thread: threading.Thread | None = None
        self.__log_listener_stop_event: threading.Event = threading.Event()

        self.__worker_process: multiprocessing.Process | None = None
        self.__listener_thread: threading.Thread | None = None
        self.__listener_stop_event: threading.Event = threading.Event()

        self.logger.info("Initializing wrapper of TTB")
        try:
            self.__start_log_listener_thread()
            self.__start_worker(timeout)
            self.__start_data_listener_thread()
            self.logger.info("TTB wrapper initialized.")
        except Exception as e:
            self.logger.critical(f"Error initializing TTB wrapper: {e}", exc_info=True)
            self.shutdown(timeout=2)
            raise

    def register_quote_callback(self, callback: Callable[[QuoteData], None]):
        """Register a callback function for real-time market data.

        The callback function will be called whenever new market data is received
        for subscribed symbols. Multiple callbacks can be registered.

        Args:
            callback (Callable[[pytaifex.QuoteData], None]): Function to call with pytaifex.QuoteData objects
        """
        self.logger.info(f"Registering quote callback: {callback.__name__}")
        self.__quote_callbacks.append(callback)

    def subscribe(self, symbols: list[str]):
        """Subscribe to real-time market data for specified symbols.

        Initiates subscription to market data feeds for the given trading symbols.
        Data will be delivered through registered callbacks when available.

        Args:
            symbols (list[str]): List of trading symbols to subscribe to (e.g., ["TXFF5", "MTXF5"]),
                "TXF5" is "TXF" + "F" + "5", "TXF" is the symbol, "F" is the month (start from "A" for January),
                "5" is the year(2025's last digit 5).

        Raises:
            pytaifex.SubscribeError: If subscription fails due to invalid symbols or API errors
            pytaifex.TTBTimeoutError: If subscription request times out

        Note:
            The TTB API may not return explicit confirmation for subscription requests.
            Check the logs and callback data to verify successful subscription.
        """
        self.logger.info(f"Subscribing to symbols: {', '.join(symbols)}")
        try:
            self.__control_queue.put({"command": "subscribe", "symbols": symbols})
            response = self.__response_queue.get(timeout=5)
            self.logger.debug(f"Subscribe response: {response}")
            if response is None:
                self.logger.info("Subscribe command sent successfully.")
                self.logger.warning("Note: TTB official API does not return response for subscribe command.")
                self.logger.warning(
                    "You should see data in callback soon if subscription is successful and there is data available."
                )
                self.logger.warning(
                    "If you don't see data in callback, it is possible that:\n"
                    + "\t1. The symbol you subscribed is not available.\n"
                    + "\t2. The symbol you subscribed is not trading.\n"
                    + "\t3. The symbol you subscribed is not in the correct format.\n"
                    + "\t4. You did'nt subscribe to the symbol in the official TTB GUI at the same time."
                )
                return
            if not isinstance(response, dict):
                raise SubscribeError(f"Unexpected response: {response}")
            if response.get("Code") != "0000":
                raise SubscribeError(response.get("ErrMsg", "No ErrMsg"))
        except queue.Empty as e:
            self.logger.error("Timeout waiting for subscribe response.")
            raise TTBTimeoutError("Timeout waiting for subscribe response.") from e
        except Exception as e:
            self.logger.error(f"Unexpected error subscribing: {e}", exc_info=True)
            raise
        return None

    def create_order(
        self,
        symbol1: str,
        side1: OrderSide,
        price: str,
        time_in_force: TimeInForce,
        order_type: OrderType,
        order_qty: str,
        day_trade: bool,
        symbol2: str | None = None,
        side2: OrderSide | None = None,
    ):
        """Create a new trading order.

        Submits a new order to the TAIFEX trading system. Supports both single-leg
        orders and spread orders (when symbol2 and side2 are provided).

        Args:
            symbol1 (str): Primary trading symbol (e.g., "TXFF5")
            side1 (pytaifex.OrderSide): Order side for primary symbol (BUY or SELL)
            price (str): Order price as string
            time_in_force (pytaifex.TimeInForce): Order time-in-force (ROD, IOC, or FOK)
            order_type (pytaifex.OrderType): Order type (MARKET or LIMIT)
            order_qty (str): Order quantity as string
            day_trade (bool): Whether this is a day trade order
            symbol2 (str, optional): Secondary symbol for spread orders
            side2 (pytaifex.OrderSide, optional): Order side for secondary symbol in spread orders

        Raises:
            pytaifex.OrderCreationError: If order creation fails due to validation or API errors
            pytaifex.TTBTimeoutError: If order creation request times out

        Example:
            ```python
            # Single-leg limit order
            ttb.create_order("TXFF5", OrderSide.BUY, "17000",
                           TimeInForce.ROD, OrderType.LIMIT, "1", False)

            # Spread order
            ttb.create_order("TXFF5", OrderSide.BUY, "50", TimeInForce.ROD,
                           OrderType.LIMIT, "1", False, "TX01", OrderSide.SELL)
            ```
        """
        self.logger.info(f"Creating order for {symbol1} at price {price} with quantity {order_qty}.")
        order_dict = {
            "Symbol1": symbol1,
            "Price": price,
            "TimeInForce": time_in_force.value,
            "Side1": side1.value,
            "OrderType": order_type.value,
            "OrderQty": order_qty,
            "DayTrade": "1" if day_trade else "0",
            "Symbol2": symbol2 if symbol2 is not None else "",
            "Side2": side2.value if side2 is not None else "",
            "PositionEffect": "",
        }
        try:
            self.__control_queue.put({"command": "create_order", "order_dict": order_dict})
            response = self.__response_queue.get(timeout=5)
            self.logger.debug(f"Create order response: {response}")
            if response is None:
                raise OrderCreationError("Order creation command sent successfully but received None as response.")
            if not isinstance(response, dict) or "Code" not in response:
                raise OrderCreationError(f"Unexpected response: {response}")
            if response.get("Code") != "0000":
                raise OrderCreationError(
                    f"error creating order ({response.get('Code', 'No code')}): "
                    + f"{response.get('ErrMsg', 'No ErrMsg')}"
                )
            self.logger.info("Order created successfully.")
        except queue.Empty as e:
            self.logger.error("Timeout waiting for create order response.")
            raise TTBTimeoutError("Timeout waiting for create order response.") from e
        except Exception as e:
            self.logger.error(f"Unexpected error creating order: {e}", exc_info=True)
            raise
        return None

    def change_price(self, order_number: str, new_price: str):
        """Modify the price of an existing order.

        Args:
            order_number (str): Order number to modify
            new_price (str): New price for the order

        Raises:
            pytaifex.OrderModificationError: If price change fails
            pytaifex.TTBTimeoutError: If request times out
        """
        self.logger.info(f"Changing price of order {order_number} to {new_price}.")
        order_dict = {"OrdNo": order_number, "Price": new_price}
        try:
            self.__control_queue.put({"command": "change_price", "order_dict": order_dict})
            response = self.__response_queue.get(timeout=5)
            if response is None:
                raise OrderModificationError("Change price command sent successfully but received None as response.")
            if not isinstance(response, dict) or "Code" not in response:
                raise OrderModificationError(f"Unexpected response: {response}")
            if response.get("Code") != "0000":
                raise OrderModificationError(
                    f"error changing price ({response.get('Code', 'No code')}): "
                    + f"{response.get('ErrMsg', 'No ErrMsg')}"
                )
            self.logger.info("Price changed successfully.")
        except queue.Empty as e:
            self.logger.error("Timeout waiting for change price response.")
            raise TTBTimeoutError("Timeout waiting for change price response.") from e
        except Exception as e:
            self.logger.error(f"Unexpected error changing price: {e}", exc_info=True)
            raise

    def change_qty(self, order_number: str, new_qty: str):
        """Modify the quantity of an existing order.

        Args:
            order_number (str): Order number to modify
            new_qty (str): New quantity for the order

        Raises:
            pytaifex.OrderModificationError: If quantity change fails
            pytaifex.TTBTimeoutError: If request times out
        """
        self.logger.info(f"Changing quantity of order {order_number} to {new_qty}.")
        order_dict = {"OrdNo": order_number, "UdpQty": new_qty}
        try:
            self.__control_queue.put({"command": "change_qty", "order_dict": order_dict})
            response = self.__response_queue.get(timeout=5)
            if response is None:
                raise OrderModificationError("Change quantity command sent successfully but received None as response.")
            if not isinstance(response, dict) or "Code" not in response:
                raise OrderModificationError(f"Unexpected response: {response}")
            if response.get("Code") != "0000":
                raise OrderModificationError(
                    f"error changing quantity ({response.get('Code', 'No code')}): "
                    + f"{response.get('ErrMsg', 'No ErrMsg')}"
                )
            self.logger.info("Quantity changed successfully.")
        except queue.Empty as e:
            self.logger.error("Timeout waiting for change quantity response.")
            raise TTBTimeoutError("Timeout waiting for change quantity response.") from e
        except Exception as e:
            self.logger.error(f"Unexpected error changing quantity: {e}", exc_info=True)
            raise

    def get_orders(self, include_done: bool = False):
        """Retrieve current orders from the trading system.

        Args:
            include_done (bool, optional): Whether to include completed orders. Defaults to False

        Returns:
            list[pytaifex.OrderData]: List of OrderData objects representing current orders

        Raises:
            pytaifex.OrderQueryError: If order query fails
            pytaifex.TTBTimeoutError: If request times out
        """
        self.logger.info("Getting orders.")
        try:
            self.__control_queue.put({"command": "get_orders"})
            response = self.__response_queue.get(timeout=5)
            self.logger.debug(f"Get orders response: {response}")
            if response is None:
                raise OrderQueryError("Get orders command sent successfully but received None as response.")
            if not isinstance(response, dict) or "Code" not in response or "Data" not in response:
                raise OrderQueryError(f"Unexpected response: {response}")
            if response.get("Code") != "0000":
                raise OrderQueryError(
                    f"error getting orders ({response.get('Code', 'No code')}): "
                    + f"{response.get('ErrMsg', 'No ErrMsg')}"
                )
            self.logger.info("Orders retrieved successfully.")
            return [
                OrderData(order_dict)
                for order_dict in response.get("Data", [])
                if include_done or order_dict.get("LESS_VOLM") != "0"
            ]
        except queue.Empty as e:
            self.logger.error("Timeout waiting for get orders response.")
            raise TTBTimeoutError("Timeout waiting for get orders response.") from e
        except Exception as e:
            self.logger.error(f"Unexpected error getting orders: {e}", exc_info=True)
            raise

    def cancel_order(self, order_number: str):
        """Cancel an existing order.

        Args:
            order_number (str): Order number to cancel

        Raises:
            pytaifex.OrderCancellationError: If order cancellation fails
            pytaifex.TTBTimeoutError: If request times out
        """
        self.logger.info(f"Cancelling order {order_number}.")
        order_dict = {"OrdNo": order_number}
        try:
            self.__control_queue.put({"command": "cancel_order", "order_dict": order_dict})
            response = self.__response_queue.get(timeout=5)
            if response is None:
                raise OrderCancellationError("Cancel order command sent successfully but received None as response.")
            if not isinstance(response, dict) or "Code" not in response:
                raise OrderCancellationError(f"Unexpected response: {response}")
            if response.get("Code") != "0000":
                raise OrderCancellationError(
                    f"error cancelling order ({response.get('Code', 'No code')}): "
                    + f"{response.get('ErrMsg', 'No ErrMsg')}"
                )
            self.logger.info("Order cancelled successfully.")
        except queue.Empty as e:
            self.logger.error("Timeout waiting for cancel order response.")
            raise TTBTimeoutError("Timeout waiting for cancel order response.") from e
        except Exception as e:
            self.logger.error(f"Unexpected error cancelling order: {e}", exc_info=True)
            raise

    def get_positions(self):
        """Retrieve current positions from the trading system.

        Returns:
            list[pytaifex.PositionData]: List of pytaifex.PositionData objects representing current positions

        Raises:
            pytaifex.PositionQueryError: If position query fails
            pytaifex.TTBTimeoutError: If request times out
        """
        self.logger.info("Getting positions.")
        try:
            self.__control_queue.put({"command": "get_positions"})
            response = self.__response_queue.get(timeout=5)
            self.logger.debug(f"Get positions response: {response}")
            if response is None:
                raise PositionQueryError("Get positions command sent successfully but received None as response.")
            if not isinstance(response, dict) or "Code" not in response or "Data" not in response:
                raise PositionQueryError(f"Unexpected response: {response}")
            if response.get("Code") != "0000":
                raise PositionQueryError(
                    f"error getting positions ({response.get('Code', 'No code')}): "
                    + f"{response.get('ErrMsg', 'No ErrMsg')}"
                )
            self.logger.info("Positions retrieved successfully.")
            return [PositionData(position_dict) for position_dict in response.get("Data", [])]
        except queue.Empty as e:
            self.logger.error("Timeout waiting for get positions response.")
            raise TTBTimeoutError("Timeout waiting for get positions response.") from e
        except Exception as e:
            self.logger.error(f"Unexpected error getting positions: {e}", exc_info=True)
            raise

    def get_accounts(self):
        """Retrieve account information from the trading system.

        Returns:
            list[dict]: List of dictionaries containing account information

        Raises:
            pytaifex.AccountQueryError: If account query fails
            pytaifex.TTBTimeoutError: If request times out
        """
        self.logger.info("Querying accounts.")
        try:
            self.__control_queue.put({"command": "get_accounts"})
            response = self.__response_queue.get(timeout=5)
            self.logger.debug(f"Query accounts response: {response}")
            if response is None:
                raise AccountQueryError("Query accounts command sent successfully but received None as response.")
            if not isinstance(response, dict) or "Code" not in response or "Data" not in response:
                raise AccountQueryError(f"Unexpected response: {response}")
            if response.get("Code") != "0000":
                raise AccountQueryError(
                    f"error querying accounts ({response.get('Code', 'No code')}): "
                    + f"{response.get('ErrMsg', 'No ErrMsg')}"
                )
            self.logger.info("Accounts queried successfully.")
            return response.get("Data", [])
        except queue.Empty as e:
            self.logger.error("Timeout waiting for query accounts response.")
            raise TTBTimeoutError("Timeout waiting for query accounts response.") from e
        except Exception as e:
            self.logger.error(f"Unexpected error querying accounts: {e}", exc_info=True)
            raise

    def is_worker_alive(self) -> bool:
        """Check if the worker process is alive and running.

        Returns:
            bool: True if worker process is alive, False otherwise
        """
        return self.__worker_process is not None and self.__worker_process.is_alive()

    def shutdown(self, timeout: int = 5):
        """Shutdown the TTB wrapper and clean up resources.

        Stops all background threads and processes, closes queues, and performs
        cleanup operations. This method is automatically called when using the
        TTB class as a context manager.

        Args:
            timeout (int, optional): Maximum time to wait for graceful shutdown. Defaults to 5
        """
        self.logger.info("Shutting down TTB wrapper.")

        # Stop data listener thread
        if self.__listener_thread is not None and self.__listener_thread.is_alive():
            self.logger.debug("Stopping data listener thread.")
            self.__listener_stop_event.set()

            self.__listener_thread.join(timeout=max(1, timeout // 3))
            if self.__listener_thread.is_alive():
                self.logger.warning("Data listener thread did not stop within timeout.")

        # Stop worker process
        if self.__worker_process is not None and self.__worker_process.is_alive():
            self.logger.debug("Stopping worker process.")
            try:
                self.__control_queue.put({"command": "shutdown"})
                self.__worker_process.join(timeout=max(1, timeout // 3))
                if self.__worker_process.is_alive():
                    self.logger.warning("Worker process did not stop within timeout. Forcing termination.")
                    self.__worker_process.terminate()
                    self.__worker_process.join(timeout=1)
            except Exception as e:
                self.logger.error(f"Error shutting down worker process: {e}", exc_info=True)
                if self.__worker_process and self.__worker_process.is_alive():
                    self.logger.warning("Worker process is still alive after error. Forcing termination.")
                    self.__worker_process.terminate()
                    self.__worker_process.join(timeout=1)
        elif self.__worker_process is not None:
            self.logger.debug("Worker process is not alive, no need to stop.")
        else:
            self.logger.debug("Worker process is not started, no need to stop.")

        # Stop log listener thread
        if self.__log_listener_thread is not None and self.__log_listener_thread.is_alive():
            self.logger.debug("Stopping log listener thread.")
            self.__log_listener_stop_event.set()

            self.__log_listener_thread.join(timeout=max(1, timeout // 3))
            if self.__log_listener_thread.is_alive():
                self.logger.warning("Log listener thread did not stop within timeout.")

        # Clear and close all queues
        self.logger.debug("Clearing and close all queues.")
        queue_to_close = [self.__data_queue, self.__control_queue, self.__response_queue, self.__log_processing_queue]
        for q in queue_to_close:
            if q is not None:
                try:
                    q.close()
                except Exception as e:
                    self.logger.error(f"Error closing queue: {e}", exc_info=True)

        self.logger.info("TTB wrapper shutdown.")

    def __enter__(self):
        """Context manager entry method."""
        return self

    def __exit__(self, *_):
        """Context manager exit method."""
        self.shutdown()

    def __del__(self):
        """Destructor to ensure proper cleanup."""
        needs_shutdown = False
        if self.__worker_process and self.__worker_process.is_alive():
            needs_shutdown = True
        if self.__listener_thread and self.__listener_thread.is_alive():
            needs_shutdown = True
        if self.__log_listener_thread and self.__log_listener_thread.is_alive():
            needs_shutdown = True

        if needs_shutdown:
            # call logger in shutdown may cause problem.
            print("TTB wrapper is being garbage collected. Shutting down.")
            self.shutdown(timeout=2)

    def __log_listener_processor(self):
        """Process log records from worker process in the main process.

        This method runs in a separate thread and handles log records sent from
        the worker process through the log queue. It forwards these records to
        the main logger for proper handling and formatting.
        """
        self.logger.debug("__log_listener_processor started")
        while not self.__log_listener_stop_event.is_set():
            try:
                log_record: logging.LogRecord = self.__log_processing_queue.get(timeout=0.2)

                self.logger.handle(log_record)
            except queue.Empty:
                continue
            except Exception as e:
                # Cannot use log here, preventing infinite logging circle
                print(f"TTB __log_listener_processor error: {e}")

        # Clear remaining log_record in the queue
        while True:
            try:
                log_record: logging.LogRecord = self.__log_processing_queue.get_nowait()
                if log_record is None:
                    continue
                self.logger.handle(log_record)
            except queue.Empty:
                break
            except Exception:
                pass

        self.logger.debug("__log_listener_processor stopped")

    def __start_log_listener_thread(self):
        """Start the log listener thread for processing worker logs."""
        if self.__log_listener_thread and self.__log_listener_thread.is_alive():
            self.logger.warning("Logging listener thread already started")
            return

        self.__log_listener_stop_event.clear()
        self.__log_listener_thread = threading.Thread(
            target=self.__log_listener_processor,
            name="TTBLogListenerThread",
            daemon=True,
        )
        self.__log_listener_thread.start()
        self.logger.debug("Logging listener thread started.")

    def __start_worker(self, timeout):
        """Start the worker process for TTB operations.

        Args:
            timeout (int): Timeout for worker initialization
        """
        if self.__worker_process and self.__worker_process.is_alive():
            self.logger.warning("Worker process already started")
            return

        self.logger.info("Starting worker process.")
        self.__worker_process = multiprocessing.Process(
            target=_ttb_worker_function,
            args=(
                self.pyc_file_path,
                self.host,
                self.zmq_port,
                self.__data_queue,
                self.__control_queue,
                self.__response_queue,
                self.__log_processing_queue,
                self.logger.name,
            ),
            daemon=True,
        )
        self.__worker_process.start()
        self.logger.info(f"Worker process started. PID: {self.__worker_process.pid}")

        # wait for the worker to initialize
        try:
            while True:
                data = self.__data_queue.get(timeout=timeout)
                if (
                    isinstance(data, dict)
                    and data.get("type") == "info"
                    and data.get("source") == "worker_initialization_runtime"
                    and data.get("success") is True
                ):
                    break
        except queue.Empty as e:
            self.logger.error("Timeout waiting for worker to initialize.")
            raise TTBTimeoutError("Timeout waiting for worker to initialize.") from e
        except Exception as e:
            self.logger.error(f"Error waiting for worker to initialize: {e}")
            raise

    def __data_listener(self):
        """Listen for data from worker process and dispatch to callbacks."""
        self.logger.debug("__data_listener started")
        while not self.__listener_stop_event.is_set():
            try:
                data = self.__data_queue.get(timeout=0.5)
                if isinstance(data, dict) and data.get("type") in ["error", "critical_error"]:
                    source = data.get("source", "unknown_worker_source")
                    message = data.get("message", "No message")
                    details = data.get("details", "")
                    self.logger.error(f"Error from worker: {source}, {message}, {details}")
                    if data.get("type") == "critical_error":
                        self.logger.critical("Critical error in worker.")
                    continue
                for callback in self.__quote_callbacks:
                    try:
                        callback(QuoteData(data))
                    except Exception as e:
                        self.logger.error(f"Error in callback: {e}", exc_info=True)
            except queue.Empty:
                continue
            except (EOFError, BrokenPipeError) as e:
                self.logger.error(f"Error in data listener: {e} (worker process might be down)", exc_info=True)
                self.__listener_stop_event.set()
                break
            except Exception as e:
                self.logger.error(f"Unexpected error in data listener: {e}", exc_info=True)
        self.logger.debug("__data_listener stopped")

    def __start_data_listener_thread(self):
        """Start the data listener thread for processing market data."""
        if self.__listener_thread and self.__listener_thread.is_alive():
            self.logger.warning("Data listener thread already started")
            return
        self.__listener_stop_event.clear()
        self.__listener_thread = threading.Thread(
            target=self.__data_listener,
            name="TTBDataListenerThread",
            daemon=True,
        )
        self.__listener_thread.start()
        self.logger.debug("Data listener thread started.")
