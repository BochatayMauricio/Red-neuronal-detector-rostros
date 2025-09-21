import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent / "model_laboratory" / "dataset"

def setup():
    """Configura los directorios necesarios para el dataset"""
    global ROOT_DIR
    
    LABELS_DIR = ROOT_DIR / "labels"
    IMAGES_DIR = ROOT_DIR / "images"
    
    # Crear directorios necesarios
    LABELS_DIR.mkdir(exist_ok=True)
    IMAGES_DIR.mkdir(exist_ok=True)
    
    # Crear subdirectorios para train/val
    (LABELS_DIR / "train").mkdir(exist_ok=True)
    (LABELS_DIR / "val").mkdir(exist_ok=True)
    (IMAGES_DIR / "train").mkdir(exist_ok=True)
    (IMAGES_DIR / "val").mkdir(exist_ok=True)

    print("ROOT_DIR:", ROOT_DIR.absolute())
    print("Directorios creados exitosamente")

setup()
