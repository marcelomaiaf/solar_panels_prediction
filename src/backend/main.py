from io import BytesIO
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from PIL import Image
from ultralytics import YOLO

from backend.energy import draw_energy_labels_pil, extract_panel_geometry
from backend.helpers import get_area, get_coordinates, get_image

app = FastAPI()

@app.post("/pipeline/")
def pipeline(
    adress: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    img: Optional[UploadFile] = File(None),
    zoom: int = 20,
):
    # 1.0 obter latitude e longitude
    if lat is None or lon is None:
        if not adress:
            raise HTTPException(status_code=400, detail="Informe lat/lon ou o parametro 'adress'")
        if not get_coordinates(adress=adress):
            raise HTTPException(status_code=400, detail="Detalhe mais o endereço")
        
        lat, lon = get_coordinates(adress=adress)
        
    if img:
        img.file.seek(0)
        base_img_pil = Image.open(img.file).convert("RGB")
        img_rgb_np = np.array(base_img_pil)
    else:

        # 2.0 obter imagem satélite da área
        base_img_pil = get_image(lat=lat, lon=lon, zoom=zoom)

        # garantir 3 canais
        base_img_pil = base_img_pil.convert("RGB")

        # converter pra numpy pra passar no YOLO
        img_rgb_np = np.array(base_img_pil)

    # 3.0 predição de painéis
    model = YOLO("backend/model.pt")
    results = model.predict(
        img_rgb_np,
        show_boxes=False,
        show_labels=False,
        imgsz=1024,
        conf=0.30,
        retina_masks=True,
        verbose=False
    )
    if not results[0].masks:
      buf = BytesIO()
      base_img_pil.save(buf, format="PNG")
      buf.seek(0)
      return Response(content=buf.getvalue(),media_type="image/png")

    # 4.0 área m² de cada máscara/grupo
    areas_m2 = get_area(results, lat, zoom)  # tensor [N]

    # 5.0 geometria, azimute e posição de label pra cada máscara
    img_rgb_for_draw, panels_geom = extract_panel_geometry(results)

    # 6.0 desenhar labels acima de cada bloco usando PIL
    annotated_pil = draw_energy_labels_pil(
        img_rgb=img_rgb_for_draw,
        panels_geom=panels_geom,
        areas_m2=areas_m2,
        lat=lat,
        lon=lon
    )

    # 7.0 retornar imagem final anotada como PNG
    buf = BytesIO()
    annotated_pil.save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")
