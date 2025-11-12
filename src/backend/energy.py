import math
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pvlib import iotools, irradiance, location

from backend.helpers import get_energy


def get_masks_and_img(results):
    """
    Extrai imagem original (BGR) e máscaras binárias [N, H, W] do resultado YOLO.
    """
    res = results[0]

    # imagem original em numpy (BGR ou RGB dependendo da lib; vamos garantir BGR pro OpenCV)
    img = res.orig_img
    if img.shape[2] == 3:
        # Ultralytics geralmente fornece RGB. OpenCV usa BGR.
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img  # fallback

    # máscaras: tensor [N, H, W] com valores float (0-1)
    masks = res.masks.data  # torch.Tensor
    # binarizar
    masks_bin = (masks > 0.5).float()

    return img_bgr, masks_bin


def min_area_rect_angle(mask_bin_np):
    """
    Recebe UMA máscara binária (np.uint8 0/255),
    retorna:
      - center (cx, cy)
      - size (w, h)
      - angle_cv2  (em graus, padrão OpenCV)
      - box_points (4 pontos do retângulo rotacionado)
    """
    # encontra contornos da máscara
    contours, _ = cv2.findContours(mask_bin_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    # pega maior contorno (caso a máscara tenha ruído fragmentado)
    cnt = max(contours, key=cv2.contourArea)

    rect = cv2.minAreaRect(cnt)
    # rect = ((cx, cy), (w, h), angle)
    # angle é em graus, range ~ [-90, 0) no OpenCV

    box = cv2.boxPoints(rect)  # 4 pontos do retângulo rotacionado
    box = np.intp(box)

    return rect, box


def rect_angle_to_azimuth(rect):
    """
    Converte o ângulo do retângulo (OpenCV) em um azimute aproximado.
    Definição que vamos usar:
      - 0° = apontando para o Norte (pra cima da imagem)
      - 90° = Leste (direita)
      - 180° = Sul
      - 270° = Oeste

    Estratégia:
    1. Pegamos o eixo "comprido" do painel (w vs h).
    2. Construímos um vetor diretor desse eixo.
    3. Calculamos o ângulo desse vetor em coordenadas imagem.
    4. Convertendo pro sistema Norte=0°, horário.
    """
    (cx, cy), (w, h), angle_cv2 = rect

    # OpenCV retorna angle de forma que:
    # - w é o lado "maior ou menor" dependendo desse angle.
    # Queremos identificar qual lado é o MAIS comprido fisicamente (painel é retangular alongado).
    if w >= h:
        long_len = w
        short_len = h
        long_axis_angle = angle_cv2  # graus
    else:
        long_len = h
        short_len = w
        long_axis_angle = angle_cv2 + 90  # se h > w, gira 90° pra pegar o eixo longo

    # Agora long_axis_angle está em graus, onde:
    # - 0° significa eixo alinhado com o eixo x da imagem (horizontal),
    # - ângulos crescem no sentido anti-horário no sistema OpenCV,
    # - mas no sistema de imagem: x cresce pra direita, y cresce pra baixo.

    # Vamos transformar isso num vetor diretor (dx, dy) no espaço da imagem
    theta_rad = math.radians(long_axis_angle)

    # Atenção: no espaço da imagem OpenCV:
    # x -> direita  (leste)
    # y -> baixo    (sul)
    dx = math.cos(theta_rad)
    dy = math.sin(theta_rad)

    # Esse (dx, dy) nos diz o "alongamento" do painel.
    # Só que um painel fotovoltaico inclinado normalmente "aponta" perpendicular à sua superfície.
    # PRA SIMPLIFICAR: vamos assumir que ele "aponta" na direção ortogonal ao lado longo.
    # Isso é uma aproximação: telhado duas águas, fileiras etc.
    # Vetor ortogonal (normal "cima/baixo" da fileira):
    # rotacionar (dx, dy) em +90° -> (-dy, dx)
    nx = dy
    ny = -dx

    # Agora nx/ny é um vetor que representa "pra onde o painel está encarando".
    # Precisamos converter isso em azimute:
    # Norte = 0°, Leste = 90°, Sul = 180°, Oeste = 270°
    # Lembrando: na imagem,
    #   eixo y pra BAIXO é SUL (+180°),
    #   eixo x pra DIREITA é LESTE (+90°)
    # Então:
    #   vetor (0, -1) -> Norte (0°)
    #   vetor (1, 0)  -> Leste (90°)
    #   vetor (0, 1)  -> Sul (180°)
    #   vetor (-1,0)  -> Oeste (270°)

    # Primeiro pega ângulo cartesiano imagem:
    az_rad = math.atan2(ny, nx)  # atan2(y, x)

    # atan2 retorna:
    #   0 rad -> (1,0) -> Leste
    #   pi/2 rad -> (0,1) -> Sul
    #   pi rad -> (-1,0) -> Oeste
    #   -pi/2 rad -> (0,-1) -> Norte

    az_deg_math = math.degrees(az_rad)
    # az_deg_math:
    #   0°   = Leste
    #   90°  = Sul
    #   180° = Oeste
    #   -90° = Norte

    # Queremos:
    #   0°   = Norte
    #   90°  = Leste
    #   180° = Sul
    #   270° = Oeste
    #
    # Mapeamento:
    #   Norte (-90°) -> 0°
    #   Leste (0°)   -> 90°
    #   Sul (90°)    -> 180°
    #   Oeste (180°) -> 270°
    #
    # Isso é basicamente rotacionar +90° e normalizar 0-360:
    az_deg_geo = (az_deg_math + 90) % 360

    # az_deg_geo agora é o azimute "bússola":
    #   0   = Norte
    #   90  = Leste
    #   180 = Sul
    #   270 = Oeste

    return az_deg_geo, (cx, cy), (nx, ny)

def get_h_mensal(lat,lon,azimute, tilt=20):
  # --- 1. Obter Dados Meteorológicos (GHI, DNI, DHI) ---
  try:
      weather_data, meta = iotools.get_pvgis_tmy(lat, lon, map_variables=True)
      altitude = meta['inputs']['location']['elevation']
      tz = weather_data.index.tz
  except Exception as e:
      print(f"Erro ao buscar dados do PVGIS: {e}")
      # Encerrar ou tratar o erro
      exit()

  # --- 2. Configurar a Localização ---
  site = location.Location(lat, lon, tz=tz, altitude=altitude, name='SeuPainel')

  # --- 3. Calcular a Posição do Sol ---
  solar_position = site.get_solarposition(times=weather_data.index)

  # --- 4. CORREÇÃO: Calcular a Irradiação Extraterrestre (dni_extra) ---
  # O modelo 'perez' precisa deste valor para funcionar.
  # Usamos o índice de datetime dos dados meteorológicos.
  dni_extra = irradiance.get_extra_radiation(weather_data.index)
  # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  # --- 5. Calcular a Irradiação no Plano do Painel (POA) ---
  # Esta é a etapa de TRANSPOSTIÇÃO, agora com o parâmetro 'dni_extra'
  poa_irradiance = irradiance.get_total_irradiance(
      surface_tilt=tilt,
      surface_azimuth=azimute,
      solar_zenith=solar_position['apparent_zenith'],
      solar_azimuth=solar_position['azimuth'],
      dni=weather_data['dni'],
      ghi=weather_data['ghi'],
      dhi=weather_data['dhi'],
      dni_extra=dni_extra,  # <-- PARÂMETRO ADICIONADO
      model='perez'
  )

  # --- 6. Calcular a Energia Diária (Hdiário) ---
  poa_df = pd.DataFrame(poa_irradiance)

  # Converte de W/m² (Potência) para Wh/m² (Energia)
  # Assumindo que os dados TMY são intervalos de 1 hora
  hourly_energy = poa_df['poa_global']

  # Agrupa por dia e soma as energias horárias
  daily_energy = hourly_energy.resample('D').sum() # Unidade: Wh/m²/dia

  # Converte para kWh/m²/dia
  daily_energy_kwh = daily_energy / 1000.0

  # --- 7. Calcular a Média Mensal do Hdiário ---
  H_diario_mensal = daily_energy_kwh.resample('ME').mean()

  H_diario_mensal.index = H_diario_mensal.index.strftime('%B')

  print("--- H diário médio (kWh/m²/dia) para cada mês (calculado com PVLIB) ---")
  print(H_diario_mensal[8:9])
  return 30.4 * H_diario_mensal.mean()


def extract_panel_geometry(results):
    """
    Lê o output do YOLO e retorna uma lista com a geometria de cada máscara/grupo.
    Cada item da lista = {
        "center": (cx, cy),
        "box": np.array([[x1,y1],[x2,y2],...]),  # 4 pontos do retângulo mínimo
        "top_y": menor_y_do_box
    }
    Também retorna a imagem base em RGB (pra PIL depois).
    """
    img_bgr, masks_bin = get_masks_and_img(results)  # você já tem essa função
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    masks_np = (masks_bin.cpu().numpy().astype(np.uint8)) * 255  # [N,H,W] 0/255

    panels_geom = []

    for i in range(masks_np.shape[0]):
        mask_i = masks_np[i]

        rect_info = min_area_rect_angle(mask_i)
        if rect_info is None:
            continue

        rect, box = rect_info
        az_deg, (cx, cy), _ = rect_angle_to_azimuth(rect)

        # top_y = ponto mais alto visualmente (menor y na imagem)
        top_y = int(np.min(box[:,1]))

        panels_geom.append({
            "mask_index": i,
            "center": (int(cx), int(cy)),
            "box": box.astype(int),
            "top_y": top_y,
            "azimuth": az_deg
        })

    return img_rgb, panels_geom


def draw_energy_labels_pil(img_rgb, panels_geom, areas_m2, lat, lon):
    """
    img_rgb: imagem base em RGB (np.array HxWx3)
    panels_geom: lista de dicts (um por máscara) com:
        - mask_index
        - center (cx, cy)
        - box (4 pontos)
        - top_y
        - azimuth
    areas_m2: tensor com área m² de cada máscara (mesmo índice mask_index)
    lat, lon: coordenadas do local

    return: pil_img anotada (PIL.Image)
    """

    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img, "RGBA")

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    total_kwh_mes = 0.0

    W, H = pil_img.size

    # Precompute bounding rects for each panel to avoid placing labels on top of panels
    def panel_bounds(panel):
        box = panel["box"]
        min_x = int(np.min(box[:, 0]))
        min_y = int(np.min(box[:, 1]))
        max_x = int(np.max(box[:, 0]))
        max_y = int(np.max(box[:, 1]))
        return (min_x, min_y, max_x, max_y)

    all_panel_rects = [panel_bounds(p) for p in panels_geom]

    # Track already placed label rectangles to prevent overlaps
    placed_rects = []

    def rect_intersects(a, b):
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        # Treat touching edges as non-overlap for easier packing
        return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)

    def clamp_label(x, y, w, h):
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x + w > W:
            x = W - w
        if y + h > H:
            y = H - h
        return x, y

    for panel in panels_geom:
        idx = panel["mask_index"]
        cx, cy = panel["center"]
        top_y = panel["top_y"]
        az  = panel["azimuth"]
        box = panel["box"]  # (4,2)

        # área estimada do grupo (m²)
        area_m2 = float(areas_m2[idx].item())

        # irradiação mensal média pro azimute desse grupo
        H_mensal = get_h_mensal(lat, lon, az)

        # energia mensal estimada do grupo (kWh/mês)
        energia_kwh_mes = get_energy(
            area=area_m2,
            H_mensal=H_mensal,
            r=0.19,
            pr=0.75
        )

        total_kwh_mes += energia_kwh_mes

        # texto que vamos colocar
        text_str = f"{energia_kwh_mes:.1f} kWh/mês"

        # ===== ALTERADO AQUI 👇 (antes era draw.textsize)
        bbox = draw.textbbox((0, 0), text_str, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        # ===== FIM ALTERAÇÃO

        # Try to place text avoiding overlap with panels and other labels
        offset_px = 15  # base distance from panel
        # Current panel bounds
        min_x, min_y, max_x, max_y = panel_bounds(panel)

        # Candidate placements: above, below, right, left
        candidates = []
        candidates.append((int(cx - text_w // 2), int(min_y - offset_px - text_h)))
        candidates.append((int(cx - text_w // 2), int(max_y + offset_px)))
        candidates.append((int(max_x + offset_px), int(cy - text_h // 2)))
        candidates.append((int(min_x - offset_px - text_w), int(cy - text_h // 2)))

        pad = 4  # background padding

        chosen_rect = None
        for tx, ty in candidates:
            tx, ty = clamp_label(tx, ty, text_w, text_h)
            bg_x0 = tx - pad
            bg_y0 = ty - pad
            bg_x1 = tx + text_w + pad
            bg_y1 = ty + text_h + pad
            rect = (bg_x0, bg_y0, bg_x1, bg_y1)
            collision = False
            # Avoid all panels and previously placed labels
            for pr in all_panel_rects:
                if rect_intersects(rect, pr):
                    collision = True
                    break
            if not collision:
                for lr in placed_rects:
                    if rect_intersects(rect, lr):
                        collision = True
                        break
            if not collision:
                chosen_rect = rect
                text_x, text_y = tx, ty
                break

        # If all preferred candidates collide, scan upwards then downwards quickly
        if chosen_rect is None:
            # Upward scan
            step = max(6, text_h // 2)
            tx = int(cx - text_w // 2)
            ty = int(min_y - offset_px - text_h)
            found = False
            for _ in range(20):
                tx, ty = clamp_label(tx, ty, text_w, text_h)
                rect = (tx - pad, ty - pad, tx + text_w + pad, ty + text_h + pad)
                if not any(rect_intersects(rect, r) for r in all_panel_rects) and not any(rect_intersects(rect, r) for r in placed_rects):
                    chosen_rect = rect
                    text_x, text_y = tx, ty
                    found = True
                    break
                ty -= step
            # Downward scan
            if not found:
                tx = int(cx - text_w // 2)
                ty = int(max_y + offset_px)
                for _ in range(20):
                    tx, ty = clamp_label(tx, ty, text_w, text_h)
                    rect = (tx - pad, ty - pad, tx + text_w + pad, ty + text_h + pad)
                    if not any(rect_intersects(rect, r) for r in all_panel_rects) and not any(rect_intersects(rect, r) for r in placed_rects):
                        chosen_rect = rect
                        text_x, text_y = tx, ty
                        break

        # Fallback: if still none, use above with clamping
        if chosen_rect is None:
            text_x, text_y = clamp_label(int(cx - text_w // 2), int(min_y - offset_px - text_h), text_w, text_h)
            chosen_rect = (text_x - pad, text_y - pad, text_x + text_w + pad, text_y + text_h + pad)

        # Draw background and text, record placed rect
        draw.rectangle(
            [(chosen_rect[0], chosen_rect[1]), (chosen_rect[2], chosen_rect[3])],
            fill=(0,0,0,160),
            outline=(255,255,255,220),
            width=1
        )

        draw.text(
            (text_x, text_y),
            text_str,
            font=font,
            fill=(255,255,255,255)
        )

        placed_rects.append(chosen_rect)

        # desenhar contorno do bloco detectado
        poly_pts = [(int(x), int(y)) for (x, y) in np.vstack([box, box[0]])]
        draw.line(
            poly_pts,
            fill=(0,255,0,255),
            width=2
        )

    # bloco do total da usina

    total_txt = f"Energia total ~ {total_kwh_mes:.1f} kWh/mês"

    # ===== ALTERADO AQUI 👇 (antes era draw.textsize)
    total_bbox = draw.textbbox((0, 0), total_txt, font=font)
    total_w = total_bbox[2] - total_bbox[0]
    total_h = total_bbox[3] - total_bbox[1]
    # ===== FIM ALTERAÇÃO

    pad_box = 6
    box_x0, box_y0 = 20, 20
    box_x1, box_y1 = box_x0 + total_w + 2*pad_box, box_y0 + total_h + 2*pad_box

    draw.rectangle(
        [(box_x0, box_y0), (box_x1, box_y1)],
        fill=(0,0,0,180),
        outline=(255,255,255,220),
        width=2
    )

    draw.text(
        (box_x0 + pad_box, box_y0 + pad_box),
        total_txt,
        font=font,
        fill=(255,255,255,255)
    )

    return pil_img
