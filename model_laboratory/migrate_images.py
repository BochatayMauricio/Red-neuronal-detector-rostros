# Celda D: crear images/train val y linkear o copiar imágenes
from pathlib import Path
import os, shutil
import config_directories
    
def setup():
    """Migra las imágenes de WIDER Face a la estructura necesaria para YOLO"""
    ROOT = Path(config_directories.ROOT_DIR)
    
    IMAGES_ROOT = ROOT / "images"
    IMAGES_TRAIN = IMAGES_ROOT / "train"
    IMAGES_VAL   = IMAGES_ROOT / "val"
    
    # Asegurar que los directorios existen
    IMAGES_ROOT.mkdir(parents=True, exist_ok=True)
    IMAGES_TRAIN.mkdir(parents=True, exist_ok=True)
    IMAGES_VAL.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Directorios de imágenes creados en: {IMAGES_ROOT}")

    LIMIT_IMAGES = None  # mismo límite que antes si querés

    def link_or_copy(src_dir, dst_dir, limit=None):
        """Enlaza o copia imágenes desde WIDER_train/WIDER_val a images/train o images/val"""
        cnt = 0
        src_path = Path(src_dir)
        
        if not src_path.exists():
            print(f"Directorio fuente no existe: {src_dir}")
            return 0
            
        for p in src_path.rglob("*.jpg"):
            if limit and cnt >= limit:
                break
            dst = Path(dst_dir) / p.name
            if dst.exists():
                cnt += 1
                continue
            try:
                # En Windows, usar copy en lugar de symlink por problemas de permisos
                if os.name == 'nt':  # Windows
                    shutil.copy2(str(p), str(dst))
                else:  # Linux/Mac
                    os.symlink(str(p), str(dst))
            except Exception as e:
                print(f"Error copiando {p}: {e}")
                try:
                    shutil.copy2(str(p), str(dst))
                except Exception as e2:
                    print(f"Error en copia de respaldo {p}: {e2}")
                    continue
            cnt += 1
            
        print(f"{cnt} imágenes procesadas desde {src_dir} -> {dst_dir}")
        return cnt

    # Intentar distintas ubicaciones que WIDER puede usar
    train_sources = ROOT / "WIDER_train" / "images"
    val_sources = ROOT / "WIDER_val" / "images"
    
    # Procesar imágenes de entrenamiento
    if train_sources.exists():
        print(f"✓ Encontradas imágenes de entrenamiento en: {train_sources}")
        link_or_copy(train_sources, IMAGES_TRAIN, limit=LIMIT_IMAGES)
    
    # Procesar imágenes de validación
    if val_sources.exists():
        print(f"✓ Encontradas imágenes de validación en: {val_sources}")
        link_or_copy(val_sources, IMAGES_VAL, limit=LIMIT_IMAGES)

if __name__ == "__main__":
    setup()
