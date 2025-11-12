import math
import os
from io import BytesIO

import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

api_key = os.getenv("gcp_key")

def get_image(lat: int, lon: int, zoom: int = 19):
  url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={zoom}&size=640x640&maptype=satellite&key={api_key}"
  response = requests.get(url)
  if  response.status_code == 200:
    return Image.open(BytesIO(response.content))
  else:
    return "Imagem Não encontrada"

def get_coordinates(adress: str):
  params = {
    "address": adress,
    "language": "pt-BR",
    "key": api_key,
  }
  url = f"https://maps.googleapis.com/maps/api/geocode/json"
  response = requests.get(url, params=params)
  if response.status_code == 200 and response.json()['results']:
      return response.json()['results'][0]['geometry']['location']['lat'],  response.json()['results'][0]['geometry']['location']['lng']
  else: None, None

def get_m_per_px(lat: int, zoom: int):
    r_terra = 6378137
    lat_rad = math.radians(lat)
    return (math.cos(lat_rad) * 2 * math.pi * r_terra) / (256 * (2 ** (zoom)))

def get_area(results, lat: int, zoom: int = 19):
  masks = results[0].masks.data
  # Ensure masks is a NumPy array [N, H, W]
  if hasattr(masks, 'cpu') and hasattr(masks, 'numpy'):
    masks_np = masks.cpu().numpy()
  elif hasattr(masks, 'numpy'):
    masks_np = masks.numpy()
  else:
    masks_np = np.array(masks)

  # Sum pixels per mask (assumes boolean/0-1 masks)
  areas_px = np.sum(masks_np, axis=(1, 2))
  m2 = get_m_per_px(lat, zoom) ** 2
  areas_m2 = areas_px * m2

  return areas_m2

def get_energy(area: float, H_mensal: float, r: float = 0.19 , pr: float = 0.75):
# r = eficiência
# pr = Performance de ratio
  return area * H_mensal * r * pr

