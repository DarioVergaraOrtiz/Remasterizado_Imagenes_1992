# Estandarizado de Imágenes de Candidatos 1992

Este repositorio contiene dos scripts en Python para procesar las fotografías de los candidatos presidenciales de Ecuador 1992:

1. **`person_cutout_with_border.py`**  
   Segmenta a la(s) persona(s) en una foto, rellena el fondo con color sólido y dibuja un contorno alrededor de la silueta.

2. **`circular_image.py`**  
   Recorta la imagen centrada, genera una máscara circular, aplica relleno interior, padding y un borde exterior.

---

## Requisitos

- Python 3.8 o superior  
- [Pillow](https://pypi.org/project/Pillow/)  
- [Torch](https://pypi.org/project/torch/) y [torchvision](https://pypi.org/project/torchvision/)  
- (Opcional) CUDA si quieres acelerar Mask R‑CNN

Instala todo con:

```bash
pip install pillow torch torchvision


Uso
1. person_cutout_with_border.py
Este script extrae la(s) persona(s) de la imagen, rellena el fondo y añade un contorno.

python person_cutout_with_border.py INPUT.jpg OUTPUT_cutout.png \
  --fill_color "#FFFFFF" \
  --outline_color "#000000" --outline_width 3 \
  --mask_thresh 0.5 --resize 800
INPUT.jpg → foto original

OUTPUT_cutout.png → resultado en PNG (fondo plano + contorno)

--fill_color (-f): color de fondo (hex o nombre)

--outline_color (-o): color del contorno

--outline_width (-w): grosor del contorno (en px)

--mask_thresh (-t): umbral de máscara (0–1)

--resize (-r): redimensiona lado mayor antes de segmentar

2. circular_image.py
Este script toma la imagen segmentada y la transforma en un círculo con borde y relleno interior.

python circular_image.py INPUT_cutout.png OUTPUT_circle.png \
  --size 140 --inner_padding 5 \
  --border_color "#003366" --border_width 5 \
  --fill_color "#EFEFEF"
INPUT_cutout.png → salida de person_cutout_with_border.py

OUTPUT_circle.png → imagen circular final

--size (-s): diámetro interior (fill + imagen) en px

--inner_padding (-p): padding entre imagen y borde interior

--border_color (-c): color del borde exterior

--border_width (-b): grosor del borde exterior

--fill_color (-f): color de fondo interior al círculo

Ejemplo de flujo
Extraer silueta:

python person_cutout_with_border.py raw.jpg cutout.png -f white -o black -w 2 -t 0.6 -r 600
Generar versión circular de 105 px (95 px + 2×5 px padding + 2×5 px borde):

python circular_image.py cutout.png circle_105.png -s 95 -p 5 -b 5 -c "#FF4500" -f "#FFFFFF"
Generar versión circular de 150 px (140 px + 2×5 px padding + 2×5 px borde):

python circular_image.py cutout.png circle_150.png -s 140 -p 5 -b 5 -c "#FF4500" -f "#FFFFFF"

Estructura del repositorio
.
├── person_cutout_with_border.py
├── circular_image.py
├── samples/
│   ├── raw.jpg
│   ├── cutout.png
│   └── circle_105.png
└── README.md