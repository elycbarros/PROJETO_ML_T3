"""Entrada principal: EDA, preparação, experimentos e avaliação reproduzíveis."""
from pathlib import Path
import runpy
from pipeline_credito import run_all

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("02_eda_graficos.py")), run_name="__main__")
    run_all()
