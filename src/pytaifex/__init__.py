import importlib.util
import logging
import os
import sys


class TTB:
    def __init__(self, pyc_file_path: str, logger: logging.Logger | None = None):
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(levelname)s: %(message)s")
            handler.setFormatter(formatter)
            handler.setLevel(logging.INFO)
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(handler)

        self.__load_pyc(pyc_file_path)

    def __load_pyc(self, pyc_file_path: str):
        self.ttb = None
        self.logger.info(f"Initializing pyc to load module TTBHelp from: {pyc_file_path}")

        if not os.path.exists(pyc_file_path):
            self.logger.error(f".pyc file not found at '{pyc_file_path}'")
            return
        if not pyc_file_path.endswith(".pyc"):
            self.logger.error(f"The provided file '{pyc_file_path}' does not have a .pyc extension.")

        try:
            # 1. Create a module spec from the .pyc file
            spec = importlib.util.spec_from_file_location("TTBHelp", pyc_file_path)

            if spec is None:
                self.logger.error(
                    f"Could not create module spec for TTBHelp from '{pyc_file_path}'. "
                    + "Check Python version compatibility."
                )
                raise Exception(
                    f"Could not create module spec for TTBHelp from '{pyc_file_path}'. "
                    + "Check Python version compatibility."
                )

            # 2. Create a module from the spec
            self.ttb = importlib.util.module_from_spec(spec)

            if self.ttb is None:
                self.logger.error("Error: Could not create module from spec for TTBHelp.")
                raise Exception("Error: Could not create module from spec for TTBHelp.")

            # 3. Add the module to sys.modules, so it can be imported from other places
            sys.modules["TTBHelp"] = self.ttb

            # 4. Execute the module's code
            if spec.loader:
                spec.loader.exec_module(self.ttb)
                self.logger.info(f"Successfully loaded TTBHelp from '{pyc_file_path}'")
                return self.ttb
            else:
                self.logger.error("Error: No loader found in spec for TTBHelp. Cannot execute module.")
                self.ttb = None
                raise Exception("Error: No loader found in spec for TTBHelp. Cannot execute module.")

        except ImportError as e:
            self.logger.error(f"ImportError loading TTBHelp from '{pyc_file_path}': {e}")
            self.ttb = None
            if "TTBHelp" in sys.modules:
                del sys.modules["TTBHelp"]  # If partially loaded, remove from sys.modules
            raise Exception(f"ImportError loading TTBHelp from '{pyc_file_path}': {e}") from e
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while loading TTBHelp from '{pyc_file_path}': {e}")
            self.ttb = None  # 清理
            if "TTBHelp" in sys.modules:
                del sys.modules["TTBHelp"]
            raise Exception(f"An unexpected error occurred while loading TTBHelp from '{pyc_file_path}': {e}") from e
