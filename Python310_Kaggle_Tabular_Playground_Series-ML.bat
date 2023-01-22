@echo off
start /min cmd /k "O: & cd O:\source\repos\Kaggle_Tabular_Playground_Series-ML"
start /min cmd /k "O: & cd O:\source\repos\venv\Python310\Scripts & activate & cd /d O:\source\repos\Kaggle_Tabular_Playground_Series-ML & python -m jupyter lab"