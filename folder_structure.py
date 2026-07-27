import os
from pathlib import Path

project_source_name='src'

Files_Folder_list=[
    f"{project_source_name}/__init__.py",
    f"{project_source_name}/data/__init__.py",
    f"{project_source_name}/data/loader.py",  
    f"{project_source_name}/data/preprocessor.py",
    f"{project_source_name}/data/conditional_variable.py",
    f"{project_source_name}/data/dataset.py",
    f"{project_source_name}/models/__init__.py",
    f"{project_source_name}/models/encoder.py",
    f"{project_source_name}/models/decoder.py",
    f"{project_source_name}/models/physics.py",
    f"{project_source_name}/models/cvae.py",
    f"{project_source_name}/loss/__init__.py",
    f"{project_source_name}/loss/reconstruction_loss.py",
    f"{project_source_name}/training/__init__.py",
    f"{project_source_name}/training/trainer.py",
    f"{project_source_name}/evaluation/metrics.py",
    f"{project_source_name}/evaluation/visualizer_plots.py",
    f"{project_source_name}/utils/__init__.py",
    f"{project_source_name}/utils/config.py",
    f"{project_source_name}/utils/logger.py",
    f"{project_source_name}/constants/__init__.py",
    "configs/base_config.yaml",
    "configs/experiment_01.yaml",
    "experiments_mlruns/mlruns/",
    "notebooks/01_data_exploration.ipynb",
    "notebooks/02_preprocessing.ipynb",
    "notebooks/03_result_visualization.ipynb",
    "tests/test_loader.py",
    "tests/test_models.py",
    "tests/test_physics.py",
    "pipeline/preprocess.py",
    "pipeline/train.py",
    "outputs/weights_checkpoints/",
    "outputs/figures/",
    "requirements.txt",
    "setup.py",
    "README.md"]


for filepath in Files_Folder_list:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
    else:
        print(f"file is already present at: {filepath}")