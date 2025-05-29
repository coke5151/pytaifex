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
class SubscribeError(Exception):
    pass


class OrderError(Exception):
    pass


# Enums
class TimeInForce(Enum):
    ROD = "1"
    IOC = "2"
    FOK = "3"


class OrderSide(Enum):
    BUY = "1"
    SELL = "2"


class OrderType(Enum):
    MARKET = "1"
    LIMIT = "2"


# Class
class QuoteData:
    def __init__(self, data: dict):
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
        return self.symbol == another.symbol and self.tick_time == another.tick_time


class OrderData:
    def __init__(self, order_dict: dict):
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
        return self.order_number == another.order_number and self.order_id == another.order_id


def _load_pyc_internal(pyc_file_path: str, logger: logging.Logger):
    """
    Load pyc module in worker process.
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
    """
    Worker function for TTB.
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
                elif command_dict.get("command") == "query_orders":
                    worker_logger.info("Received query orders command.")
                    resp = _ttb_instance.QUERYRESTOREREPORT()
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
    def __init__(
        self,
        pyc_file_path: str,
        host: str = "http://localhost:8080",
        zmq_port: int = 51141,
        logger: logging.Logger | None = None,
        timeout: int = 5,
    ):
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
        self.logger.info(f"Registering quote callback: {callback.__name__}")
        self.__quote_callbacks.append(callback)

    def subscribe(self, symbols: list[str]):
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
            raise TimeoutError("Timeout waiting for subscribe response.") from e
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
                raise OrderError("Order creation command sent successfully but received None as response.")
            if not isinstance(response, dict) or "Code" not in response:
                raise OrderError(f"Unexpected response: {response}")
            if response.get("Code") != "0000":
                raise OrderError(
                    f"error creating order ({response.get('Code', 'No code')}): "
                    + f"{response.get('ErrMsg', 'No ErrMsg')}"
                )
            self.logger.info("Order created successfully.")
        except queue.Empty as e:
            self.logger.error("Timeout waiting for create order response.")
            raise TimeoutError("Timeout waiting for create order response.") from e
        except Exception as e:
            self.logger.error(f"Unexpected error creating order: {e}", exc_info=True)
            raise
        return None

    def change_price(self, order_number: str, new_price: str):
        self.logger.info(f"Changing price of order {order_number} to {new_price}.")
        order_dict = {"OrdNo": order_number, "Price": new_price}
        try:
            self.__control_queue.put({"command": "change_price", "order_dict": order_dict})
            response = self.__response_queue.get(timeout=5)
            if response is None:
                raise OrderError("Change price command sent successfully but received None as response.")
            if not isinstance(response, dict) or "Code" not in response:
                raise OrderError(f"Unexpected response: {response}")
            if response.get("Code") != "0000":
                raise OrderError(
                    f"error changing price ({response.get('Code', 'No code')}): "
                    + f"{response.get('ErrMsg', 'No ErrMsg')}"
                )
            self.logger.info("Price changed successfully.")
        except queue.Empty as e:
            self.logger.error("Timeout waiting for change price response.")
            raise TimeoutError("Timeout waiting for change price response.") from e
        except Exception as e:
            self.logger.error(f"Unexpected error changing price: {e}", exc_info=True)
            raise

    def change_qty(self, order_number: str, new_qty: str):
        self.logger.info(f"Changing quantity of order {order_number} to {new_qty}.")
        order_dict = {"OrdNo": order_number, "UdpQty": new_qty}
        try:
            self.__control_queue.put({"command": "change_qty", "order_dict": order_dict})
            response = self.__response_queue.get(timeout=5)
            if response is None:
                raise OrderError("Change quantity command sent successfully but received None as response.")
            if not isinstance(response, dict) or "Code" not in response:
                raise OrderError(f"Unexpected response: {response}")
            if response.get("Code") != "0000":
                raise OrderError(
                    f"error changing quantity ({response.get('Code', 'No code')}): "
                    + f"{response.get('ErrMsg', 'No ErrMsg')}"
                )
            self.logger.info("Quantity changed successfully.")
        except queue.Empty as e:
            self.logger.error("Timeout waiting for change quantity response.")
            raise TimeoutError("Timeout waiting for change quantity response.") from e
        except Exception as e:
            self.logger.error(f"Unexpected error changing quantity: {e}", exc_info=True)
            raise

    def query_orders(self):
        self.logger.info("Querying orders.")
        try:
            self.__control_queue.put({"command": "query_orders"})
            response = self.__response_queue.get(timeout=5)
            self.logger.debug(f"Query orders response: {response}")
            if response is None:
                raise OrderError("Query orders command sent successfully but received None as response.")
            if not isinstance(response, dict) or "Code" not in response or "Data" not in response:
                raise OrderError(f"Unexpected response: {response}")
            if response.get("Code") != "0000":
                raise OrderError(
                    f"error querying orders ({response.get('Code', 'No code')}): "
                    + f"{response.get('ErrMsg', 'No ErrMsg')}"
                )
            self.logger.info("Orders queried successfully.")
            return [OrderData(order_dict) for order_dict in response.get("Data", [])]
        except queue.Empty as e:
            self.logger.error("Timeout waiting for query orders response.")
            raise TimeoutError("Timeout waiting for query orders response.") from e
        except Exception as e:
            self.logger.error(f"Unexpected error querying orders: {e}", exc_info=True)
            raise

    def is_worker_alive(self) -> bool:
        return self.__worker_process is not None and self.__worker_process.is_alive()

    def shutdown(self, timeout: int = 5):
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
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()

    def __del__(self):
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
        """
        Running in the main process, deal with the log from worker process.
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
            raise TimeoutError("Timeout waiting for worker to initialize.") from e
        except Exception as e:
            self.logger.error(f"Error waiting for worker to initialize: {e}")
            raise

    def __data_listener(self):
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
