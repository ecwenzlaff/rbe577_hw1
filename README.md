This is a 'uv' managed project, so when 'uv' is installed, the code can be run (and data generated) by running 'uv run control_allocator.py' from the command line/terminal (assuming that the pyproject.toml, uv.lock, and .python-version files are all within the same directory). Also, since this is a 'uv' managed project, if you're working in VSCode, make sure you run 'uv venv --seed' and 'uv sync' before you point VSCode to the .venv as the interpreter. Without this command, the .venv's 'activate' script won't function the same as a .venv's script created through "python -m venv"

Alternatively, this script can also be run with a virtual environment using the requirements.txt file. Please see below for requirements:

Python Version: 3.10.8

pip freeze:
contourpy==1.3.2
cycler==0.12.1
filelock==3.19.1
fonttools==4.60.0
fsspec==2025.9.0
Jinja2==3.1.6
kiwisolver==1.4.9
MarkupSafe==3.0.2
matplotlib==3.10.6
mpmath==1.3.0
networkx==3.4.2
numpy==2.2.6
packaging==25.0
pillow==11.3.0
pyparsing==3.2.4
python-dateutil==2.9.0.post0
six==1.17.0
sympy==1.14.0
torch==2.8.0+cu126
typing_extensions==4.15.0