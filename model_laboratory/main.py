import sys
from pathlib import Path

# Importar los módulos
import config_directories
import Wider_parser
import create_yaml
import migrate_images
import train

def main():
    print("="*60)
    print("GENERANDO MODELO - DETECTOR DE ROSTROS YOLO")
    print("="*60)
    
    try:
        print("\n[1/5] Configurando directorios...")
        config_directories.setup()
        print("✓ Directorios configurados")
        
        print("\n[2/5] Parseando anotaciones WIDER Face a formato YOLO...")
        Wider_parser.setup()
        print("✓ Anotaciones parseadas")
        
        print("\n[3/5] Migrando imágenes a estructura YOLO...")
        migrate_images.setup()
        print("✓ Imágenes migradas")
        
        print("\n[4/5] Creando archivo data.yaml...")
        create_yaml.setup()
        print("✓ Archivo data.yaml creado")
        
        print("\n[5/5] Iniciando entrenamiento del modelo...")
        success = train.setup()
        
        if success:
            print("✓ ¡Entrenamiento completado exitosamente!")
            print("\nEl modelo entrenado se encuentra en:")
            print("runs/detect/wider_face_exp/weights/")
        else:
            print("✗ Error durante el entrenamiento")
            return False
            
    except Exception as e:
        print(f"\n✗ Error durante la ejecución: {e}")
        print(f"Tipo de error: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETADO")
    print("="*60)
    return True

def check_requirements():
    """Verifica que se tengan las dependencias necesarias"""
    missing_deps = []
    
    try:
        import PIL
    except ImportError:
        missing_deps.append("Pillow")
    
    try:
        import ultralytics
    except ImportError:
        missing_deps.append("ultralytics")
    
    if missing_deps:
        print("Dependencias faltantes:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nInstala con:")
        print(f"pip install {' '.join(missing_deps)}")
        return False
    
    return True

if __name__ == "__main__":
    print("Verificando dependencias...")
    if not check_requirements():
        sys.exit(1)
    
    print("Iniciando pipeline...")
    success = main()
    
    if not success:
        sys.exit(1)


