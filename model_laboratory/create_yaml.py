from pathlib import Path
import config_directories

def setup():
    """Crea el archivo data.yaml necesario para entrenar con YOLO"""
    ROOT = Path(__file__).parent.parent / "model_laboratory"
    data_yaml = ROOT / "data.yaml"
    train_p = (config_directories.ROOT_DIR / "images" / "train").resolve()
    val_p   = (config_directories.ROOT_DIR / "images" / "val").resolve()

    # Crear el contenido del archivo YAML
    yaml_content = f"""train: {train_p}
val: {val_p}
nc: 1
names: ['face']
"""
    
    data_yaml.write_text(yaml_content, encoding='utf-8')
    print("Creado:", data_yaml)
    print("Contenido del archivo:")
    print(data_yaml.read_text(encoding='utf-8'))

if __name__ == "__main__":
    setup()