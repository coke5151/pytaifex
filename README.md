# Python Project Template
A modern Python project template leveraging `uv` for dependency management and integrating various development tools for code quality assurance.

## Features
- Fast and reliable package management with `uv`
- Integrated code quality tools settings:
  - Ruff for linting and formatting
  - Mypy for static type checking
  - VSCode integration
  - Jupyter notebook

## Project Structure
```
.
├── notebooks/         # Jupyter notebook files
├── .vscode/          # VSCode configuration
├── README.md         # Project documentation
├── LICENSE           # License
├── .gitignore       # Git ignore patterns
├── mypy.ini         # Mypy configuration
├── ruff.toml        # Ruff configuration
└── pyrightconfig.json # Pyright configuration
```

## Development Setup
1. Install `uv`:
   ```bash
   # macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   ```bash
   # Windows
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
2. Initialize Project:
   ```bash
   # For applications
   uv init --python <version>
   # Example: uv init --python 3.13

   # For libraries (eg. PyPI distribution)
   uv init --lib
   ```
3. Install Packages:
   ```bash
   # Install development tools globally (each tool has its own venv)
   uv tool install mypy ruff

   # Install development dependencies (jupyterlab for example)
   uv add jupyterlab --dev  # --dev for development dependencies

   # Install project dependencies (pandas and requests for example)
   uv add pandas requests  # example packages
   ```
4. Run Code:
   ```bash
   uv run main.py  # Run main.py in virtual environment
   uv run jupyter lab  # Launch Jupyter Lab
   ```

## Recommanded VSCode/Cursor Extension
- Must-have
	- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
	- [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance)
	- [Python Debugger](https://marketplace.visualstudio.com/items?itemName=ms-python.debugpy)
	- [Mypy Type Checker](https://marketplace.visualstudio.com/items?itemName=ms-python.mypy-type-checker)
	- [Ruff](https://marketplace.cursorapi.com/items?itemName=charliermarsh.ruff)
	- [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
- Nice to have:
	- [autoDocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring)
	- [Error Lens](https://marketplace.visualstudio.com/items?itemName=usernamehw.errorlens)
	- [GitLens](https://marketplace.visualstudio.com/items?itemName=eamodio.gitlens)
	- [Git Graph](https://marketplace.visualstudio.com/items?itemName=mhutchie.git-graph)

## License
See [LICENSE](./LICENSE) for details.
