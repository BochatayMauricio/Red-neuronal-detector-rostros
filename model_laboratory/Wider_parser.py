from pathlib import Path
from PIL import Image
import config_directories

def setup():
    """Parsea los archivos de anotaciones WIDER Face a formato YOLO"""
    ROOT = Path(config_directories.ROOT_DIR)
    
    # Encontrar el directorio que contiene wider_face_split
    SPLIT = ROOT / "wider_face_split"
    
    TRAIN_GT = SPLIT / 'wider_face_train_bbx_gt.txt'
    VAL_GT   = SPLIT / 'wider_face_val_bbx_gt.txt'

    OUT_LABELS = ROOT / "labels"
    OUT_TRAIN = OUT_LABELS / "train"
    OUT_VAL   = OUT_LABELS / "val"
    OUT_TRAIN.mkdir(parents=True, exist_ok=True)
    OUT_VAL.mkdir(parents=True, exist_ok=True)

    LIMIT_IMAGES = None  # poné None para procesar todo, usa un número pequeño para probar

    def is_int_string(s):
        try:
            int(s)
            return True
        except:
            return False

    def parse_wider_file(gt_path, images_root, out_dir, limit=None, max_lookahead=10):
        print("Parseando:", gt_path)
        with open(gt_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_lines = f.readlines()
        
        
        lines = [l.rstrip("\n").rstrip("\r") for l in raw_lines] # cantidad de lineas del original
        i = 0
        count = 0
        total_lines = len(lines)
        while i < total_lines:
            # buscar próxima línea no-vacía que presumimos es el path de la imagen
            while i < total_lines and lines[i].strip() == "":
                i += 1
            if i >= total_lines:
                break
            img_rel = lines[i].strip()
            i += 1
            if img_rel == "":
                continue

            # buscar la línea siguiente que contenga el número de bounding boxes
            # (saltamos vacías). Si la línea no es un entero, buscamos hacia adelante
            # hasta max_lookahead líneas para recuperar un entero.
            # Esto soluciona problemas por líneas extra o saltos de formato.
            num = None
            look_i = i
            while look_i < total_lines and (lines[look_i].strip() == ""):
                look_i += 1
            if look_i < total_lines:
                candidate = lines[look_i].strip()
                if is_int_string(candidate):
                    num = int(candidate)
                    i = look_i + 1
                else:
                    # intentar buscar un enterno en los próximos max_lookahead renglones
                    found = False
                    for j in range(look_i, min(total_lines, look_i + max_lookahead)):
                        if lines[j].strip() == "":
                            continue
                        if is_int_string(lines[j].strip()):
                            num = int(lines[j].strip())
                            i = j + 1
                            found = True
                            print(f"[RECUP] Imagen {img_rel}: número de bboxes encontrado en línea {j} (lookahead).")
                            break
                    if not found:
                        # no se pudo recuperar: emitimos warning y saltamos esta imagen
                        print(f"[WARN] No se encontró número de bboxes tras imagen '{img_rel}' (línea ~{look_i}). Saltando entrada.")
                        # intentamos continuar sin consumir más líneas (ya i apunta después de img_rel)
                        continue
            else:
                print(f"[WARN] fin de archivo inesperado tras imagen '{img_rel}'")
                break

            # ahora leemos 'num' líneas de bboxes, saltando vacías si las hubiera
            bboxes = []
            read = 0
            while read < num and i < total_lines:
                line = lines[i].strip()
                i += 1
                if line == "":
                    continue
                parts = line.split()
                if len(parts) < 4:
                    # línea inválida: emitir warning y continuar
                    print(f"[WARN] bbox inválida para '{img_rel}' (esperadas >=4 columnas): '{line[:80]}'")
                    continue
                try:
                    x, y, w, h = map(float, parts[:4])
                    bboxes.append((x, y, w, h))
                except Exception as e:
                    print(f"[WARN] error parseando bbox para '{img_rel}': {e} -- line: '{line[:80]}'")
                    continue
                read += 1

            # localizar la imagen en el árbol images_root (puede venir con subcarpetas)
            img_path = Path(images_root) / img_rel
            if not img_path.exists():
                found = None
                # buscar por nombre de archivo en images_root
                for p in Path(images_root).rglob(Path(img_rel).name):
                    found = p
                    break
                if found:
                    img_path = found
                else:
                    print(f"[WARN] Imagen no encontrada: {img_rel} -> saltando")
                    continue

            # obtener dimensiones y escribir fichero .txt en formato YOLO
            try:
                iw, ih = Image.open(img_path).size
            except Exception as e:
                print(f"[WARN] no se pudo abrir imagen {img_path}: {e}")
                continue

            out_file = Path(out_dir) / (img_path.stem + ".txt")
            try:
                with open(out_file, "w", encoding="utf-8") as out:
                    for (x,y,w,h) in bboxes:
                        x_c = x + w/2.0
                        y_c = y + h/2.0
                        x_cn = x_c / iw
                        y_cn = y_c / ih
                        w_n = w / iw
                        h_n = h / ih
                        def clamp(v): return max(0.0, min(1.0, v))
                        out.write(f"0 {clamp(x_cn):.6f} {clamp(y_cn):.6f} {clamp(w_n):.6f} {clamp(h_n):.6f}\n")
                count += 1
            except Exception as e:
                print(f"[ERROR] escribiendo label para {img_path}: {e}")

            if limit and count >= limit:
                break

        print(f"Generados {count} labels en {out_dir} (líneas de entrada: {total_lines})")

    # Ejecutar parseo 
    if TRAIN_GT.exists():
        parse_wider_file(str(TRAIN_GT), str(ROOT / "WIDER_train" / "images"), OUT_TRAIN, limit=LIMIT_IMAGES)
    else:
        print("No existe train gt:", TRAIN_GT)

    if VAL_GT.exists():
        parse_wider_file(str(VAL_GT), str(ROOT / "WIDER_val" / "images"), OUT_VAL, limit=LIMIT_IMAGES)
    else:
        print("No existe val gt:", VAL_GT)

if __name__ == "__main__":
    setup()