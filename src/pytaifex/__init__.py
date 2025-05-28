import importlib.util
import logging
import multiprocessing
import os
import queue
import sys
import threading
from logging.handlers import QueueHandler
from typing import Any, Callable


class SubscribeError(Exception):
    pass


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
    response_q_in: multiprocessing.Queue,
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
                if command_dict.get("command") == "subscribe":
                    symbols = command_dict.get("symbols", [])
                    worker_logger.info(f"Received subscribe command for symbol: {', '.join(symbols)}")
                    resp = _ttb_instance.QUOTEDATA(",".join(symbols))
                    response_q_in.put(resp)
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

    def register_quote_callback(self, callback: Callable[[Any], None]):
        self.logger.info(f"Registering quote callback: {callback.__name__}")
        self.__quote_callbacks.append(callback)

    def subscribe(self, symbols: list[str]):
        self.logger.info(f"Subscribing to symbols: {', '.join(symbols)}")
        try:
            self.__control_queue.put({"command": "subscribe", "symbols": symbols})
            response = self.__response_queue.get(timeout=5)
            if response is None:
                self.logger.info("Subscribed successfully.")
                return  # official ttb behavior, no response code for subscribe
            if not isinstance(response, dict):
                raise SubscribeError(f"Unexpected response: {response}")
            if response.get("Code") != "0000":
                raise SubscribeError(response.get("Message", "No message"))
        except queue.Empty as e:
            self.logger.error("Timeout waiting for subscribe response.")
            raise TimeoutError("Timeout waiting for subscribe response.") from e
        except Exception as e:
            self.logger.error(f"Unexpected error subscribing: {e}", exc_info=True)
            raise
        return None

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
                        callback(data)
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
