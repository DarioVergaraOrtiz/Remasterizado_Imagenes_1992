#!/usr/bin/env python3
"""
person_cutout_with_border.py

Recorta la(s) persona(s) en la imagen de entrada usando Mask R-CNN y rellena el fondo con un color elegido.
Además, dibuja un contorno (borde) alrededor de la silueta de la persona con color y grosor configurables.

Uso en terminal:
    python person_cutout_with_border.py input.jpg output.png \
        --fill_color "#00FF00" \
        --outline_color "#FF0000" --outline_width 5 \
        [--mask_thresh 0.5] [--resize 800]

- fill_color: color de fondo de todo lo que NO sea persona.
- outline_color: color del contorno alrededor de la silueta.
- outline_width: grosor en píxeles del contorno (aproximado con vecindad cuadrada).
- mask_thresh: umbral de máscara de Mask R-CNN.
- resize: redimensionar lado mayor antes de segmentar (para acelerar).
- device: opcionalmente “cpu” o “cuda”.

Dependencias:
    pip install torch torchvision pillow
"""

import argparse
import os
import sys
from PIL import Image, ImageDraw, ImageColor, ImageFilter, ImageChops
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
    Ejecuta Mask R-CNN sobre la imagen y devuelve una máscara booleana de la(s) persona(s).
    Devuelve máscara PIL 'L' (0 o 255) del tamaño original: 255 indica píxeles de persona.
    Igual que antes: redimensiona antes si resize_to, luego remapea.
    """
    original_size = img_pil.size  # (width, height)
    img_to_process = img_pil
    resized = False
    if resize_to is not None and resize_to > 0:
        w, h = img_pil.size
        max_side = max(w, h)
        if max_side > resize_to:
            scale = resize_to / max_side
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_to_process = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            resized = True

    img_tensor = torchvision.transforms.functional.to_tensor(img_to_process).to(device)

    with torch.no_grad():
        outputs = model([img_tensor])
    output = outputs[0]
    masks = output.get('masks')    # [N,1,H,W]
    labels = output.get('labels')  # [N]
    scores = output.get('scores')  # [N]
    if masks is None or len(masks) == 0:
        return Image.new('L', original_size, 0)

    # Filtrar solo etiquetas de persona (COCO label=1)
    person_indices = (labels == 1).nonzero(as_tuple=False).squeeze(1)
    if len(person_indices) == 0:
        return Image.new('L', original_size, 0)

    # Combinar máscaras de persona con OR:
    mask_proc = None
    for idx in person_indices.tolist():
        mask_i = masks[idx, 0]  # tensor [h_proc, w_proc]
        mask_bin = (mask_i >= mask_thresh)
        if mask_proc is None:
            mask_proc = mask_bin
        else:
            mask_proc = mask_proc.logical_or(mask_bin)
    if mask_proc is None:
        return Image.new('L', original_size, 0)

    mask_proc = mask_proc.to(torch.uint8)
    mask_pil = Image.fromarray((mask_proc.cpu().numpy() * 255).astype('uint8'), mode='L')
    if resized:
        mask_pil = mask_pil.resize(original_size, resample=Image.Resampling.NEAREST)
    return mask_pil

def add_outline(mask_pil, outline_width):
    """
    Genera una máscara de contorno a partir de la máscara binaria de la persona.
    - mask_pil: PIL L, 0 fondo, 255 persona.
    - outline_width: grosor en píxeles (>=1).
    Retorna una nueva máscara 'L' de mismo tamaño donde los píxeles del contorno valen 255, y resto 0.
    Usamos dilatación aproximada con ImageFilter.MaxFilter de tamaño (outline_width*2+1).
    outline_mask = dilated_mask - original_mask.
    """
    if outline_width <= 0:
        return Image.new('L', mask_pil.size, 0)
    # Asegurar modo L y binario (0 o 255)
    mask = mask_pil.convert("L")
    # Tomamos umbral para asegurar binaridad (por si llegan valores intermedios)
    mask = mask.point(lambda p: 255 if p >= 128 else 0)
    # Tamaño de kernel para MaxFilter: un entero impar
    kernel_size = outline_width * 2 + 1
    try:
        dilated = mask.filter(ImageFilter.MaxFilter(kernel_size))
    except Exception:
        # En versiones antiguas de PIL ImageFilter.MaxFilter puede requerir un tamaño fijo; 
        # pero normalmente acepta el tamaño como argumento.
        dilated = mask.filter(ImageFilter.MaxFilter(size=kernel_size))
    # border_mask = dilated - mask
    border = ImageChops.subtract(dilated, mask)
    # Binarizar de nuevo (0 o 255)
    border = border.point(lambda p: 255 if p >= 128 else 0)
    return border

def composite_with_outline(img_pil, mask_pil, fill_color, outline_color=None, outline_width=0):
    """
    Crea nueva imagen RGBA:
    - fill_color: tupla RGBA o RGB (se completa alpha=255).
    - mask_pil: PIL L con 0 fondo /255 persona.
    - outline_color: tupla RGBA o RGB para el contorno; si None o outline_width<=0, no dibuja contorno.
    - outline_width: grosor en píxeles.
    Flujo:
    1. background: color fill_color.
    2. Si outline: crear mask_outline = add_outline(mask_pil, outline_width).
       Pegar sobre background en los píxeles donde mask_outline=255 el color outline_color.
    3. Pegar persona: sobre el resultado, en máscara mask_pil.
    """
    img_rgba = img_pil.convert("RGBA")
    w, h = img_rgba.size

    # Preparar fill
    if len(fill_color) == 3:
        fill = (fill_color[0], fill_color[1], fill_color[2], 255)
    else:
        fill = fill_color
    background = Image.new("RGBA", (w, h), fill)

    result = background.copy()

    # Dibujar contorno si se pide
    if outline_color is not None and outline_width and outline_width > 0:
        # Obtener máscara del contorno
        mask_outline = add_outline(mask_pil, outline_width)
        # Crear capa con color de contorno
        if len(outline_color) == 3:
            ocol = (outline_color[0], outline_color[1], outline_color[2], 255)
        else:
            ocol = outline_color
        outline_layer = Image.new("RGBA", (w, h), ocol)
        # Pegar outline_layer sobre result usando mask_outline como máscara
        result.paste(outline_layer, (0,0), mask_outline)

    # Finalmente, pegar la persona sobre el contorno/fondo
    # mask_pil ya es L con 0/255
    result.paste(img_rgba, (0,0), mask_pil.convert("L"))

    return result

def main():
    parser = argparse.ArgumentParser(
        description="Recorta personas de la imagen y rellena el fondo con un color, añadiendo contorno a la silueta."
    )
    parser.add_argument("input", help="Ruta de la imagen de entrada")
    parser.add_argument("output", help="Ruta de la imagen de salida (PNG recomendado)")
    parser.add_argument("--fill_color", "-f", type=parse_color, required=True,
                        help="Color de fondo para todo lo que NO sea persona. Ej: '#00FF00', 'blue', '#FF00FF80'")
    parser.add_argument("--mask_thresh", "-t", type=float, default=0.5,
                        help="Umbral para máscara de persona (0–1).")
    parser.add_argument("--resize", "-r", type=int, default=None,
                        help="Redimensionar lado mayor antes de segmentar para acelerar (p.ej. 800).")
    parser.add_argument("--outline_color", "-o", type=parse_color, default=None,
                        help="Color del contorno alrededor de la persona. Ej: '#FF0000'. Si no se especifica, no dibuja contorno.")
    parser.add_argument("--outline_width", "-w", type=int, default=0,
                        help="Grosor en píxeles del contorno alrededor de la silueta. Si 0, no dibuja contorno.")
    parser.add_argument("--device", "-d", type=str, default=None,
                        help="Dispositivo para PyTorch: 'cpu' o 'cuda'. Por defecto usa 'cuda' si está disponible, sino 'cpu'.")
    args = parser.parse_args()

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

    # Cargar modelo Mask R-CNN
    print("Cargando modelo Mask R-CNN...")
    model = load_model(device)

    # Obtener máscara de persona
    print("Procesando segmentación de persona (Mask R-CNN)...")
    mask = get_person_mask(model, img, device, mask_thresh=args.mask_thresh, resize_to=args.resize)

    # Componer con outline si se indicó
    print("Construyendo imagen final con fondo y contorno...")
    result = composite_with_outline(
        img_pil=img,
        mask_pil=mask,
        fill_color=args.fill_color,
        outline_color=args.outline_color,
        outline_width=args.outline_width
    )

    # Guardar resultado
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    result.save(args.output, format="PNG")
    print(f"[OK] Imagen guardada en {args.output}")

if __name__ == "__main__":
    main()
