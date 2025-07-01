#!/usr/bin/env python3
"""
circular_image.py

Script para procesar imágenes: recorta centrado a cuadrado, redimensiona,
aplica máscara circular, añade borde de color y fondo interior (relleno) configurable,
además de permitir padding interno para mostrar fill_color alrededor de la foto.

Uso en la terminal:
    python circular_image.py input.jpg output.png --size 200 \
        --border_color "#FF4500" --border_width 10 --fill_color "#FFFFFF" --inner_padding 10

Esto generará una imagen circular interior de 200×200 px (área total interior), con:
- Padding interno de 10 px: la imagen del candidato se redimensiona a 180×180 y se centra,
  dejando 10 px de fondo (fill_color) alrededor.
- Borde exterior de 10 px en color #FF4500.
- Color de relleno interior #FFFFFF.
- Tamaño final = 200 + 2*10 = 220 px de diámetro (incluyendo borde).
"""

import os
import sys
from PIL import Image, ImageDraw, ImageColor
import argparse

class CircularImageProcessor:
    """
    Clase para procesar imágenes y generar versiones recortadas en círculo,
    con borde de color, fondo interior (fill_color) configurable, tamaño configurable,
    y padding interno para dejar espacio de fondo alrededor de la imagen.
    """

    def __init__(self, border_color="#FF4500", border_width=10, output_size=None, fill_color=None, inner_padding=0):
        """
        Inicializa el procesador.

        :param border_color: Color del borde. Puede ser:
                             - Cadena hexadecimal: "#RRGGBB" o "#RRGGBBAA" o sin '#', p.e. "FF4500".
                             - Nombre de color reconocido por PIL: "red", "blue", etc.
                             - Tupla RGBA: (r, g, b) o (r, g, b, a).
        :param border_width: Grosor del borde en píxeles (entero >= 0).
        :param output_size: Tamaño del diámetro interior total de la imagen circular (fill + imagen interior).
                            Si es None, se toma el menor lado de la imagen original.
                            Si es un entero, se usa ese valor tanto para ancho como alto.
                            Si es una tupla/list con al menos 2 valores, se toma min(w, h).
        :param fill_color: Color de fondo interior al círculo. Igual reglas que border_color.
                           Si None, queda transparente.
        :param inner_padding: Padding interno en píxeles entre la imagen recortada y el borde interior.
                              La imagen del candidato se redimensionará a (output_size - 2*inner_padding).
                              Debe ser >= 0 y tal que output_size >= 2*inner_padding.
        """
        # Validar border_width
        if border_width is None or border_width < 0:
            raise ValueError("border_width debe ser un entero >= 0")
        self.border_width = border_width

        # Parsear y validar border_color
        self.border_color = self._parse_color(border_color) if border_color is not None else None

        # output_size (puede ser None o entero o tupla/list)
        if output_size is not None:
            if isinstance(output_size, int):
                if output_size <= 0:
                    raise ValueError("output_size debe ser un entero positivo o None")
                self.output_size = output_size
            elif isinstance(output_size, (tuple, list)) and len(output_size) >= 2:
                # lo almacenamos sin transformación: extraeríamos later el mínimo
                self.output_size = tuple(output_size)
            else:
                raise ValueError("output_size debe ser None, un entero positivo o una tupla/list de largo >= 2")
        else:
            self.output_size = None

        # Parsear y validar fill_color
        self.fill_color = self._parse_color(fill_color) if fill_color is not None else None

        # inner_padding
        if inner_padding is None or inner_padding < 0:
            raise ValueError("inner_padding debe ser un entero >= 0")
        self.inner_padding = inner_padding

    def _parse_color(self, color):
        """
        Convierte color (string o tupla) a tupla RGBA válida.
        Si es string, usa ImageColor.getcolor.
        Si es tupla de 3 o 4 elementos, la convierte a RGBA (añade 255 si solo RGB).
        """
        if isinstance(color, str):
            # Acepta "#RRGGBB", "#RRGGBBAA", nombres de color, etc.
            try:
                # getcolor con "RGBA" devuelve tupla (r,g,b,a)
                rgba = ImageColor.getcolor(color, "RGBA")
                return rgba
            except Exception as e:
                raise ValueError(f"No se pudo parsear fill/border color '{color}': {e}")
        elif isinstance(color, (tuple, list)):
            if len(color) == 3:
                r, g, b = color
                return (r, g, b, 255)
            elif len(color) == 4:
                return tuple(color)
            else:
                raise ValueError("Color como tupla debe tener 3 (RGB) o 4 (RGBA) elementos")
        else:
            raise ValueError("Color debe ser string (hex, nombre) o tupla/list de 3 o 4 enteros")

    def process(self, input_path, output_path):
        """
        Procesa la imagen de entrada, genera versión circular con borde, fondo interior y padding,
        y guarda en output_path.

        :param input_path: Ruta de la imagen de entrada.
        :param output_path: Ruta donde se guardará la imagen resultante (PNG recomendado).
        """
        # Verificar que el archivo existe
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"No se encontró el archivo de entrada: {input_path}")

        # Carga la imagen
        img = Image.open(input_path).convert("RGBA")
        width, height = img.size

        # Determinar recorte cuadrado centrado según menor lado
        min_side = min(width, height)
        left = (width - min_side) // 2
        upper = (height - min_side) // 2
        img_cropped = img.crop((left, upper, left + min_side, upper + min_side))

        # Determinar tamaño interior total (diámetro interior) para fill + imagen interior
        if self.output_size is not None:
            if isinstance(self.output_size, int):
                size = self.output_size
            else:
                # tupla/list: tomar mínimo
                size = min(self.output_size[0], self.output_size[1])
        else:
            size = min_side

        # Verificar inner_padding válido
        if size < 2 * self.inner_padding:
            raise ValueError(f"output_size ({size}) debe ser >= 2*inner_padding ({2*self.inner_padding})")

        # Calcular tamaño real de la imagen dentro del círculo (contenido)
        image_inner_size = size - 2 * self.inner_padding
        if image_inner_size <= 0:
            raise ValueError("inner_padding demasiado grande: no queda espacio para la imagen interior")

        # Redimensionar recorte original a tamaño image_inner_size
        img_resized = img_cropped.resize((image_inner_size, image_inner_size), Image.Resampling.LANCZOS)

        # Crear máscara circular para la imagen interior
        mask = Image.new('L', (image_inner_size, image_inner_size), 0)
        draw_mask = ImageDraw.Draw(mask)
        draw_mask.ellipse((0, 0, image_inner_size, image_inner_size), fill=255)

        # Imagen circular recortada con transparencia fuera de la máscara
        img_circular = Image.new('RGBA', (image_inner_size, image_inner_size), (0,0,0,0))
        img_circular.paste(img_resized, (0, 0), mask)

        # Crear fondo con borde y fill interior
        border_size = size + 2 * self.border_width  # diámetro total con borde exterior
        background = Image.new('RGBA', (border_size, border_size), (0, 0, 0, 0))
        draw_bg = ImageDraw.Draw(background)

        # Dibuja círculo de borde completo (diámetro border_size)
        if self.border_color is not None and self.border_width > 0:
            draw_bg.ellipse((0, 0, border_size, border_size), fill=self.border_color)

        # Dibuja círculo de fill interior si se especifica (diámetro = size)
        if self.fill_color is not None:
            inner_left = self.border_width
            inner_top = self.border_width
            inner_right = self.border_width + size
            inner_bottom = self.border_width + size
            draw_bg.ellipse((inner_left, inner_top, inner_right, inner_bottom), fill=self.fill_color)

        # Pegar la imagen circular centrada con padding interno
        paste_x = self.border_width + self.inner_padding
        paste_y = self.border_width + self.inner_padding
        background.paste(img_circular, (paste_x, paste_y), img_circular)

        # Asegurar carpeta de salida existente
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Guardar como PNG para mantener transparencia
        background.save(output_path, format='PNG')
        print(f"[OK] Guardada imagen circular con borde, fill y padding en: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Procesar imagen para recortarla en círculo con borde, fondo interior, y padding interno."
    )
    parser.add_argument("input", help="Ruta de la imagen de entrada")
    parser.add_argument("output", help="Ruta donde se guardará la imagen resultante (PNG recomendado)")
    parser.add_argument(
        "--size", "-s", type=int, default=None,
        help="Diámetro interior total en píxeles (fill + imagen interior). Si no se indica, usa el menor lado de la imagen original."
    )
    parser.add_argument(
        "--border_color", "-c", type=str, default="#FF4500",
        help="Color del borde en formato hexadecimal (#RRGGBB o nombre de color)."
    )
    parser.add_argument(
        "--border_width", "-b", type=int, default=10,
        help="Grosor del borde en píxeles (>=0)."
    )
    parser.add_argument(
        "--fill_color", "-f", type=str, default=None,
        help="Color de fondo interior al círculo, en formato hexadecimal o nombre. Si no se especifica, queda transparente."
    )
    parser.add_argument(
        "--inner_padding", "-p", type=int, default=0,
        help="Padding interno en píxeles entre el borde interior y la imagen. Deja espacio para mostrar fill_color alrededor de la foto."
    )
    args = parser.parse_args()

    try:
        processor = CircularImageProcessor(
            border_color=args.border_color,
            border_width=args.border_width,
            output_size=args.size,
            fill_color=args.fill_color,
            inner_padding=args.inner_padding
        )
        processor.process(args.input, args.output)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
