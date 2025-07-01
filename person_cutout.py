#!/usr/bin/env python3
"""
person_cutout.py

Recorta la/s persona/s en la imagen de entrada usando Mask R-CNN y rellena el fondo con un color elegido.
Solo se mantienen los píxeles de clase “persona”; todo lo demás queda pintado del color de fondo.

Uso en terminal:
    python person_cutout.py input.jpg output.png --fill_color "#00FF00" --mask_thresh 0.5 --resize 800

Esto pintará todo el fondo de verde (#00FF00) y dejará solo la(s) persona(s) detectada(s).

Dependencias:
    pip install torch torchvision pillow
    (Opcionalmente tqdm, etc.)
"""

import argparse
import os
from PIL import Image, ImageDraw, ImageColor
import torch
import torchvision

def parse_color(color_str):
    """
    Parsea color dado como:
    - Cadena hex #RRGGBB o #RRGGBBAA
    - Nombre de color PIL (“red”, “blue”, etc.)
    Devuelve tupla RGBA (r,g,b,a) con alpha=255 si no se especifica.
    """
    if color_str is None:
        return None
    try:
        # ImageColor.getcolor puede aceptar #RGB, #RRGGBB, nombres, etc.
        rgba = ImageColor.getcolor(color_str, "RGBA")
        return rgba
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Color inválido '{color_str}': {e}")

def load_model(device):
    """
    Carga Mask R-CNN preentrenado en COCO (detecta personas: label=1).
    """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    model.eval()
    return model

def get_person_mask(model, img_pil, device, mask_thresh=0.5, resize_to=None):
    """
    Ejecuta Mask R-CNN sobre la imagen y devuelve una máscara booleana de la/s persona/s.
    - img_pil: PIL.Image (RGB o RGBA).
    - mask_thresh: umbral de probabilidad para considerar píxel como parte de la persona.
    - resize_to: si se especifica entero >0, redimensiona la imagen manteniedo aspect ratio para que el lado mayor sea resize_to.
      Esto acelera el procesamiento; luego se reescala la máscara al tamaño original.
    Devuelve: máscara PIL de tipo 'L' (0 o 255) del tamaño original de img_pil, donde 255 indica píxeles de persona.
    """
    # Convertir PIL a tensor
    # Opción de redimensionar para inferencia:
    original_size = img_pil.size  # (width, height)
    img_to_process = img_pil
    resized = False
    if resize_to is not None and resize_to > 0:
        w, h = img_pil.size
        # Determinar factor para lado mayor
        max_side = max(w, h)
        if max_side > resize_to:
            scale = resize_to / max_side
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_to_process = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            resized = True

    # Transformar a tensor y normalizar
    # torchvision espera imagen en [0,1] en 3 canales, sin normalizar manualmente para Mask R-CNN
    # (Mask R-CNN preprocesa internamente normalización).
    img_tensor = torchvision.transforms.functional.to_tensor(img_to_process).to(device)

    with torch.no_grad():
        outputs = model([img_tensor])
    output = outputs[0]
    # output es dict con 'masks', 'labels', 'scores', 'boxes'
    masks = output.get('masks')    # tensor [N, 1, H, W], con valores entre 0 y 1 (probabilidad)
    labels = output.get('labels')  # tensor [N], int con etiqueta COCO
    scores = output.get('scores')  # tensor [N], confidencias

    if masks is None or len(masks) == 0:
        # Ninguna detección
        # Devolver máscara de todo False
        return Image.new('L', original_size, 0)

    # Filtrar solo etiquetas de persona: en COCO person es label == 1
    person_indices = (labels == 1).nonzero(as_tuple=False).squeeze(1)
    if len(person_indices) == 0:
        return Image.new('L', original_size, 0)

    # Combine máscaras de persona con umbral
    # masks[person_indices] tiene forma [M, 1, h_proc, w_proc]
    # Hacemos OR sobre instancias:
    mask_proc = None
    for idx in person_indices.tolist():
        mask_i = masks[idx, 0]  # tensor [h_proc, w_proc], valores [0,1]
        # Aplicar umbral
        mask_bin = (mask_i >= mask_thresh)
        if mask_proc is None:
            mask_proc = mask_bin
        else:
            mask_proc = mask_proc.logical_or(mask_bin)
    if mask_proc is None:
        return Image.new('L', original_size, 0)

    # mask_proc es tensor bool de tamaño de img_to_process
    mask_proc = mask_proc.to(torch.uint8)  # 0/1
    # Convertir a PIL y reescalar a tamaño original si fue redimensionada
    mask_pil = Image.fromarray((mask_proc.cpu().numpy() * 255).astype('uint8'), mode='L')
    if resized:
        # Reescalar máscara a tamaño original con interpolación cercana
        mask_pil = mask_pil.resize(original_size, resample=Image.Resampling.NEAREST)
    return mask_pil

def composite_person_on_color(img_pil, mask_pil, fill_color):
    """
    Crea nueva imagen RGBA:
    - fill_color: tupla RGBA (r,g,b,a) o RGB (r,g,b) interpretado con alpha=255.
    - mask_pil: PIL L con 0 fondo /255 persona.
    - img_pil: PIL RGB/RGBA de entrada.
    Resultado: nueva PIL RGBA donde:
        Si mask=255: pixel de img_pil (mantiene RGBA original o añade alpha=255 si RGB).
        Si mask=0: pixel = fill_color.
    """
    # Asegurar img en RGBA
    img_rgba = img_pil.convert("RGBA")
    w, h = img_rgba.size

    # Crear fondo
    # Si fill_color es tupla de longitud 3 o 4:
    if len(fill_color) == 3:
        fill = (fill_color[0], fill_color[1], fill_color[2], 255)
    else:
        fill = fill_color  # asume RGBA
    background = Image.new("RGBA", (w, h), fill)

    # Convertir mask a modo 'L' mismo tamaño
    mask = mask_pil.convert("L")
    # El método: paste img sobre background usando mask como alpha
    # Pero mask=255 para persona, queremos conservar persona: pegamos la persona sobre fill.
    # background.paste(img_rgba, box=(0,0), mask=mask) funcionará:
    #   donde mask=255, toma píxeles de img_rgba; donde mask=0, deja background.
    result = background.copy()
    result.paste(img_rgba, (0,0), mask)
    return result

def main():
    parser = argparse.ArgumentParser(description="Recorta personas de la imagen y rellena el fondo con un color.")
    parser.add_argument("input", help="Ruta de la imagen de entrada")
    parser.add_argument("output", help="Ruta de la imagen de salida (PNG para mantener transparencia si se desea)")
    parser.add_argument("--fill_color", "-f", type=parse_color, required=True,
                        help="Color de fondo para todo lo que NO sea persona. Ej: '#00FF00', 'blue', '#FF00FF80'")
    parser.add_argument("--mask_thresh", "-t", type=float, default=0.5,
                        help="Umbral para máscara de persona (0–1). Valores más bajos incluyen más píxeles; por defecto 0.5.")
    parser.add_argument("--resize", "-r", type=int, default=None,
                        help="Redimensionar lado mayor de la imagen a este valor antes de procesar para acelerar, luego volver a tamaño original.")
    parser.add_argument("--device", "-d", type=str, default=None,
                        help="Dispositivo para PyTorch: 'cpu' o 'cuda'. Por defecto usa 'cuda' si está disponible, sino 'cpu'.")
    args = parser.parse_args()

    # Verificar existencia
    if not os.path.isfile(args.input):
        print(f"[ERROR] No existe archivo de entrada: {args.input}", file=sys.stderr)
        exit(1)

    # Cargar imagen
    img = Image.open(args.input).convert("RGB")

    # Dispositivo
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Usando dispositivo: {device}")

    # Cargar modelo (Mask R-CNN)
    model = load_model(device)

    # Obtener máscara de persona
    print("Procesando segmentación de persona (Mask R-CNN)... Esto puede tardar en CPU.")
    mask = get_person_mask(model, img, device, mask_thresh=args.mask_thresh, resize_to=args.resize)

    # Componer imagen final
    result = composite_person_on_color(img, mask, args.fill_color)

    # Guardar resultado
    # Si deseas transparencia fuera de la persona en vez de color de fondo, podrías ajustar fill_color alpha=0.
    # Pero aquí asumimos fondo opaco.
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    result.save(args.output, format="PNG")
    print(f"[OK] Imagen guardada en {args.output}")

if __name__ == "__main__":
    main()
