# TFG_GUI — Sistema de detección y seguimiento con GUI (YOLOv8 + ByteTrack/KCF + FastSAM)

Repositorio del código desarrollado para el TFG. El objetivo es ofrecer una **aplicación de escritorio** (PySide6) que integra:
- **Detección** con YOLOv8 (Ultralytics).
- **Seguimiento** multi-objeto (**MOT**) con ByteTrack.
- **Seguimiento** individual (**SOT**) con Multi-KCF (OpenCV contrib).
- **Segmentación** asistida (FastSAM) para selección/edición de ROI.
- **HUD/Overlay** informativo y utilidades de telemetría/ROI.

> ⚠️ Los **pesos de modelos** (`*.pt`) no están incluidos en el repositorio. Abajo se explica cómo descargarlos/colocarlos.

---

## Tabla de contenidos
1. [Características](#características)
2. [Demo rápida](#demo-rápida)
3. [Requisitos](#requisitos)
4. [Instalación](#instalación)
5. [Modelos (pesos)](#modelos-pesos)
6. [Ejecutar la aplicación](#ejecutar-la-aplicación)
7. [Estructura del proyecto](#estructura-del-proyecto)
8. [Configuración (`config/models.yaml`)](#configuración-configmodelsyaml)
9. [Controles y ayuda](#controles-y-ayuda)
10. [Solución de problemas](#solución-de-problemas)
11. [Nota adicional](#nota-adicional)
12. [Licencia](#licencia)
13. [Cita](#cita)

---

## Características
- **Detección** en tiempo real con YOLOv8 (modelos Ultralytics).
- **Seguimiento MOT** con ByteTrack para IDs consistentes por objeto.
- **Seguimiento SOT** con **Multi-KCF** (OpenCV contrib).
- **Segmentación/selección** con **FastSAM** para refinar o crear ROIs.
- **Interfaz PySide6** con overlay informativo (clase, ID, bbox, etc.).
- Arquitectura **modular** en `core/` (detectores, trackers, UI, utilidades).

---

## Demo rápida
1. Instala dependencias (ver [Instalación](#instalación)).
2. Descarga/coloca los pesos (ver [Modelos](#modelos-pesos)).
3. Lanza la GUI:
   ```bash
   python gui_app.py

---

## Requisitos
- **Python:** 3.10 o superior.
- **Sistema operativo:** Windows 10/11 (probado). Linux/macOS deberían funcionar con las mismas dependencias si OpenCV y Qt se instalan correctamente.
- **CPU:** suficiente para inferencia en tiempo real con modelos ligeros (p. ej., YOLOv8n).  
  **GPU NVIDIA (opcional):** acelera sustancialmente. Instala PyTorch con la build de CUDA adecuada desde la web oficial de PyTorch y usa `device: "cuda"` en `config/models.yaml`.
- **RAM:** ≥ 8 GB recomendado (16 GB si se utiliza **FastSAM** u otros modelos pesados).
- **Almacenamiento:** espacio adicional para vídeos de entrada/salida y para los pesos de los modelos (no incluidos en el repo).
- **Cámara / vídeo:** cámara web/RTSP o rutas a archivos de vídeo compatibles.
- **Dependencias principales:** se instalan con `pip install -r requirements.txt` (incluye `ultralytics`, `opencv-contrib-python`, `PySide6`, `torch`, etc.).  
  > Nota: **KCF/Multi-KCF** requiere *opencv-contrib*, ya contemplado en `requirements.txt`.
- **Herramientas opcionales:**
  - `ffmpeg` (códecs/vídeo adicionales).
  - GStreamer (si se usan fuentes RTSP específicas en Linux).
- **Entorno recomendado:** virtual (`venv`) o **Conda** para aislar dependencias.

---

## Instalación

> Puedes usar **pip + venv** (rápido y ligero) o **Conda** (si ya trabajas con Anaconda). Ambas opciones instalan lo mismo desde `requirements.txt`.

### Opción A — pip + entorno virtual
```bash
# 1) Crear y activar entorno virtual
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 2) Actualiza pip y wheel
python -m pip install --upgrade pip wheel

# 3) Instala dependencias del proyecto
pip install -r requirements.txt
```
### Opción B — Conda
```bash
# 1) Crear y activar entorno
conda create -n tfg_gui python=3.10 -y
conda activate tfg_gui

# 2) Instalar dependencias del proyecto
pip install -r requirements.txt
```

---

## Modelos (pesos)
Los pesos **no están incluidos** en el repositorio (se ignoran con `.gitignore` por su tamaño).  
Debes descargarlos manualmente y colocarlos en la carpeta `config/models/` o actualizar las rutas en `config/models.yaml`.

### Modelos recomendados
- **YOLOv8** (ej. `yolov8n.pt`) — descarga desde [Ultralytics](https://github.com/ultralytics/ultralytics).
- **FastSAM** (ej. `FastSAM-s.pt`) — descarga desde [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM).
- **Modelo personalizado del TFG** (ej. `best.pt`) — entrenado específicamente para este proyecto. No se distribuye en este repositorio (disponible bajo petición).

---

## Ejecutar la aplicación

```bash
python gui_app.py
```
---

## Estructura del proyecto
El repositorio está organizado de forma modular para facilitar la comprensión y la extensión del sistema:

- **`core/`** → Código principal dividido en módulos:
  - `detector_yolov8.py` → detección con YOLOv8 (Ultralytics).
  - `tracker_mot_bytetrack.py` → seguimiento multiobjeto con ByteTrack.
  - `tracker_sot_multikcf.py` → seguimiento individual con Multi-KCF.
  - `tracker_utils.py` → utilidades para tracking.
  - `bounding_box.py` → operaciones sobre cajas delimitadoras (bboxes).
  - `roi_manual.py` → selección manual de ROI.
  - `roi_segmentation.py` → segmentación con FastSAM.
  - `ui_overlay.py` → overlay/HUD con información en pantalla.
  - `ui_mouse.py` → interacción con ratón.
  - `telemetry.py` → telemetría e información auxiliar.
  - `comms.py`, `compat.py`, `coord_utils.py` → módulos auxiliares.
  - `main.py` → lógica principal de ejecución (si aplica).
- **`config/`** → Archivos de configuración:
  - `models.yaml` → rutas y parámetros de los modelos.
  - `models/` → carpeta para los pesos (`*.pt`), ignorada en git.
- **`gui_app.py`** → script principal que lanza la interfaz gráfica.
- **`requirements.txt`** → dependencias del proyecto.
- **`.gitignore`** → reglas para ignorar archivos en Git (como pesos, entornos, temporales).
- **`README.md`** → documentación principal del proyecto.

---

## Configuración (`config/models.yaml`)
El archivo `config/models.yaml` centraliza las rutas a los modelos y los parámetros principales del sistema.  
Su estructura típica es:

```yaml
# Rutas a modelos
yolo:    "config/models/yolov8n.pt"
fastsam: "config/models/FastSAM-s.pt"
custom:  "config/models/best.pt"   # modelo entrenado en el TFG (opcional)

# Parámetros de detección
detector:
  conf_threshold: 0.25
  iou_threshold: 0.45
  device: "cpu"    # usar "cuda" si se dispone de GPU NVIDIA

# Seguimiento
tracker:
  mot: "bytetrack"   # opciones: "bytetrack" | "none"
  sot: "multikcf"    # opciones: "multikcf" | "none"

# HUD/Overlay
overlay:
  show_ids: true
  show_class: true
  show_coords: true
```

---

## Controles y ayuda
La aplicación incluye un panel de **Info/Teclas** donde se listan los atajos y controles disponibles.  
Entre las acciones principales destacan:

- **Selección de ROI** con el ratón.  
- Activación y desactivación de **detección**.  
- Activación y desactivación de **seguimiento MOT (ByteTrack)** y **SOT (Multi-KCF)**.  
- Uso de **FastSAM** para segmentación y refinamiento de regiones.  
- Alternar la visibilidad de elementos en el overlay (IDs, clases, coordenadas).  

> Si el texto mostrado en la ventana de ayuda se corta, redimensiona la ventana o utiliza la barra de desplazamiento.

---

## Solución de problemas
- **`ModuleNotFoundError: <paquete>`**  
  Alguna librería no está incluida en `requirements.txt`. Instálala manualmente y añádela.  
  Paquetes habituales:
  - `filterpy` → usado en ciertas implementaciones de ByteTrack.  
  - `lap` → requerido por algunos algoritmos de MOT.  

- **KCF no disponible (`cv2.legacy` no encontrado)**  
  Asegúrate de instalar **opencv-contrib-python** (ya incluido en `requirements.txt`).  
  Evita usar únicamente `opencv-python`.

- **Error de Qt/PySide6 (“plugin platform windows”)**  
  Reinstala `PySide6` en un entorno limpio y asegúrate de ejecutar siempre desde la consola del entorno activo.

- **La GPU no se utiliza**  
  Verifica que PyTorch con soporte CUDA esté correctamente instalado:  
  ```python
  import torch
  print(torch.cuda.is_available())

---

## Nota adicional
Además de ejecutar la aplicación desde `gui_app.py`, también es posible lanzar el sistema directamente mediante `main.py` desde un terminal:

```bash
python core/main.py
```

---

## Licencia
Este proyecto se distribuye bajo la licencia **MIT**.  
Consulta el archivo `LICENSE` incluido en el repositorio para más detalles.  

Los modelos externos utilizados (YOLOv8, FastSAM, etc.) mantienen las licencias de sus respectivos autores.

---

## Cita
Si este software te resulta útil, por favor utiliza la siguiente referencia:

```bibtex
@software{TFG_GUI_2025,
  author  = {Sergio Sánchez Ramos},
  title   = {TFG_GUI — Sistema de detección y seguimiento con GUI},
  year    = {2025},
  url     = {https://github.com/SergioSR02/TFG_GUI}
}
