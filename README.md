# Monitor de ergonomía postural con YOLO Pose

Detecta en tiempo real, con la webcam, si estás sentado con mala postura frente al computador y te avisa.
Usa el modelo pre-entrenado `yolo26n-pose.pt` (COCO Keypoints) de Ultralytics, sin entrenar nada.

## Qué mide

| Problema | Cómo se detecta |
|---|---|
| Hombros desnivelados | Ángulo de la línea hombro-hombro vs. la horizontal |
| Cabeza ladeada | Ángulo de la línea ojo-ojo vs. la horizontal |
| Encorvado / cabeza baja | Distancia vertical hombros→nariz ÷ ancho de hombros |
| Muy cerca de la pantalla | El ancho de hombros en píxeles crece respecto a la calibración |

Se compara contra **tu** postura correcta (calibración con `c`) y solo alerta si la mala postura dura 3 s seguidos.
Muestra los grados en vivo y un puntaje de sesión.

## Instalación (Windows)

La webcam no está disponible en WSL, así que se ejecuta con Python de Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Uso

```powershell
python postura_ergonomica.py
```

- `c`: calibrar (siéntate derecho, con cara y hombros visibles)
- `q`: salir

La primera vez se descarga el modelo (~7.5 MB).
