# Detector de Rostros con YOLOv8n (WIDER Face)

Este repositorio contiene un pipeline completo para entrenar un detector de rostros con YOLOv8n usando el dataset WIDER Face, y un script de demostración para ejecutar la detección en tiempo real con un modelo ya entrenado.

## Requisitos

- Python 3.9 o superior
- pip reciente

## Instalación de dependencias (comandos)

Actualizar pip y herramientas base:
```bash
python -m pip install --upgrade pip setuptools wheel
```

Instalar dependencias principales:
```bash
pip install ultralytics opencv-python Pillow
```

Instalar PyTorch (elige la adecuada para tu sistema; consulta https://pytorch.org/get-started/locally/):
- CPU (genérico):
```bash
pip install torch torchvision torchaudio
```

## Si solo querés ver cómo funciona la red (demo rápida)

Hay un script en el directorio raíz que usa un modelo ya entrenado para hacer inferencia con la webcam.

- Archivo: `detector_model.py`
- Ejecutar:
```bash
python detector_model.py
```
- Controles: se abre una ventana con la detección cuadro a cuadro. Presioná `q` para salir.

Notas:
- El script carga el modelo desde `runs/detect/wider_face_exp/weights/best.pt`. 
- Si ese archivo no existe, colocá tus pesos entrenados en esa ruta o ajustá la variable `MODEL_PATH` dentro de `detector_model.py`.

## Volver a entrenar el modelo (pipeline completo)

El entrenamiento y la preparación del dataset están automatizados dentro de `model_laboratory`. Antes de ejecutar, prepará el dataset WIDER Face como se explica abajo.

### 1) Preparar el dataset WIDER Face

- Descargar WIDER Face desde: http://shuoyang1213.me/WIDERFACE/
- Crear la carpeta `model_laboratory/dataset`
- Dentro de `model_laboratory/dataset`, colocar las carpetas extraídas del dataset SIN CAMBIAR LOS NOMBRES. Deben ser 4 en total:
  - `WIDER_train`
  - `WIDER_val`
  - `WIDER_test`
  - `wider_face_split`

La estructura esperada (resumen):

```
model_laboratory/
  dataset/
    WIDER_train/
      images/...
    WIDER_val/
      images/...
    WIDER_test/
      images/...
    wider_face_split/
      wider_face_train_bbx_gt.txt
      wider_face_val_bbx_gt.txt
      wider_face_test_filelist.txt
```

El pipeline creará de forma automática la estructura YOLO (labels/images con train/val) dentro de `model_laboratory/dataset/`.

### 2) Ejecutar el pipeline de entrenamiento

Desde la raíz del repositorio:
```bash
cd model_laboratory
python main.py
```

Este script ejecuta, en orden:
1. `config_directories.setup()` — prepara directorios base (`images/` y `labels/` con `train/` y `val/`) dentro de `model_laboratory/dataset/`.
2. `Wider_parser.setup()` — convierte las anotaciones de WIDER Face al formato YOLO y genera los `.txt` en `labels/train` y `labels/val`.
3. `migrate_images.setup()` — migra (copia o crea enlaces) las imágenes de `WIDER_train/images` y `WIDER_val/images` a `images/train` y `images/val`.
4. `create_yaml.setup()` — crea `model_laboratory/data.yaml` con la configuración del dataset (1 clase: `face`).
5. `train.setup()` — entrena con YOLOv8n usando los parámetros de `model_laboratory/train.py`.

Al finalizar, los pesos se guardan en:
```
runs/detect/wider_face_exp/weights/
```

### Ajustar hiperparámetros de entrenamiento

En `model_laboratory/train.py` podés modificar:
```python
EPOCHS = 5
BATCH  = 16
IMGSZ  = 320 o 640 depende de tu hardware. Mejor si es 640
```
El entrenamiento usa el checkpoint base `yolov8n.pt` (descarga automática si no está presente).

## Breve explicación del modelo (YOLOv8n)

- YOLOv8n (nano) es la variante más liviana de la familia YOLOv8 de Ultralytics, diseñada para alta velocidad y uso en dispositivos con recursos limitados.
- Se utiliza para detección de objetos; en este proyecto se entrena para una única clase: `face`.
- Ventajas:
  - Inferencia rápida y bajo consumo de memoria.
  - Entrenamiento sencillo con la API de Ultralytics.
- Flujo general:
  - Preparación del dataset (anotaciones + estructura YOLO).
  - `data.yaml` define rutas de `train` y `val`, número de clases y nombres.
  - Entrenamiento con `YOLO('yolov8n.pt').train(...)`.
  - Generación de pesos optimizados (`best.pt`) para inferencia.

## Comandos útiles

- Ejecutar demo (webcam):
```bash
python detector_model.py
```

- Entrenar desde cero:
```bash
cd model_laboratory
python main.py
```

- Reinstalar dependencias clave:
```bash
pip install --upgrade ultralytics opencv-python Pillow
```

## Créditos

- Dataset: WIDER Face — http://shuoyang1213.me/WIDERFACE/
- Framework: Ultralytics YOLO — https://github.com/ultralytics/ultralytics
