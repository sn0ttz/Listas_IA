import nbformat
from nbconvert import PythonExporter

# Ler o notebook
with open('DecisionTree_Restaurante.ipynb', 'r', encoding='utf-8') as f:
    notebook = nbformat.read(f, as_version=4)

# Converter para Python
exporter = PythonExporter()
source_code, _ = exporter.from_notebook_node(notebook)

# Salvar o arquivo Python
with open('DecisionTree_Restaurante.py', 'w', encoding='utf-8') as f:
    f.write(source_code)
