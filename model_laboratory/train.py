# Parámetros de entrenamiento
EPOCHS = 5
BATCH = 16
IMGSZ = 320

from pathlib import Path
import subprocess
import config_directories

def setup():
    """Entrena el modelo YOLO con el dataset WIDER Face"""
    ROOT = Path(__file__).parent.parent / "model_laboratory"
    data_yaml = ROOT / "data.yaml"
    
    # Verificar que el archivo data.yaml existe
    if not data_yaml.exists():
        print(f"Error: No se encuentra el archivo {data_yaml}")
        print("Asegúrate de haber ejecutado create_yaml.setup() primero")
        return False
    
    # Verificar que las carpetas de imágenes existen y tienen contenido
    train_images = config_directories.ROOT_DIR / "images" / "train"
    val_images = config_directories.ROOT_DIR / "images" / "val"
    
    if not train_images.exists() or len(list(train_images.glob("*.jpg"))) == 0:
        print(f"Error: No se encontraron imágenes de entrenamiento en {train_images}")
        return False
        
    if not val_images.exists() or len(list(val_images.glob("*.jpg"))) == 0:
        print(f"Error: No se encontraron imágenes de validación en {val_images}")
        return False
    
    print(f"Iniciando entrenamiento con {len(list(train_images.glob('*.jpg')))} imágenes de entrenamiento")
    print(f"y {len(list(val_images.glob('*.jpg')))} imágenes de validación")
    
    try:
        # Intentar importar ultralytics
        from ultralytics import YOLO
        
        # Crear el modelo
        model = YOLO('yolov8n.pt')  # Descarga automáticamente si no existe
        
        # Configurar paths para resultados
        project_path = Path(__file__).parent.parent / "runs" / "detect"
        
        # Entrenar el modelo
        results = model.train(
            data=str(data_yaml),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            project=str(project_path),
            name="wider_face_exp",
            exist_ok=True
        )
        
        print("¡Entrenamiento completado exitosamente!")
        print(f"Modelo guardado en: {project_path}/wider_face_exp/weights/")
        return True
        
    except ImportError:
        print("Error: ultralytics no está instalado.")
        print("Instálalo con: pip install ultralytics")
        return False
        
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
        
        # Fallback: usar comando de línea si ultralytics está instalado
        print("Intentando con comando de línea...")
        try:
            cmd = [
                "yolo", "task=detect", "mode=train", 
                "model=yolov8n.pt", 
                f"data={str(data_yaml)}", 
                f"epochs={EPOCHS}", 
                f"imgsz={IMGSZ}", 
                f"batch={BATCH}", 
                f"project={project_path}",
                "name=wider_face_exp",
                "exist_ok=True"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).parent.parent))
            print("Código de retorno:", result.returncode)
            
            if result.stdout:
                print("Salida:", result.stdout)
            if result.stderr:
                print("Errores:", result.stderr)
                
            return result.returncode == 0
            
        except FileNotFoundError:
            print("Error: comando 'yolo' no encontrado.")
            print("Asegúrate de que ultralytics esté instalado correctamente.")
            return False

if __name__ == "__main__":
    setup()

