# Solar Panels Prediction 

Satellite-powered intelligence for the solar economy.

This project detects rooftop solar panels from satellite imagery and estimates how much energy they produce. It combines **YOLOv11**, **physics-based PV modeling**, and a **cloud-native architecture on Azure** to turn raw images into business-ready energy insights.

---

## Overview

Given either:

- a **written address**, or  
- a **latitude / longitude** pair,

the system:

1. Fetches the **satellite image** of the location  
2. Uses a **fine-tuned YOLOv11 model** to detect solar panels  
3. Applies **deterministic physical formulas**, **meteorological data**, and **solar irradiance** for that location  
4. Outputs an estimate of **energy production** (e.g. kWh for that rooftop / area)

The application is deployed on **Azure Container Apps** and can be used both interactively and in batch/data-pipeline scenarios.

---

## Why This Matters (Business Value)

This is not just an ML demo—it is designed as a **data product** for the solar ecosystem:

- **Investors & Funds**  
  - Estimate installed PV capacity in **cities or small countries**  
  - Approximate expected annual **energy generation** from rooftops  
  - Support **valuation, risk analysis, and infrastructure planning**

- **Solar Integrators & Commercial Teams**  
  - Identify neighborhoods with **high or low solar penetration**  
  - Prioritize **sales campaigns** in under-served areas  
  - Estimate potential generation and revenue across regions

- **Energy & Urban Planners**  
  - Feed **grid studies, decarbonization plans, and smart city dashboards**  
  - Understand where distributed generation is already strong or still emerging

---

## Architecture (Azure-Native)

The solution is fully cloud-native on Azure with two main usage flows:

```text
User (Address / Lat-Long)
          │
          ▼
 Azure Container Apps (API + YOLOv11 + PV Model)
          │
          ├─► Interactive response (JSON + metrics)
          │
          └─► Azure Blob Storage (when used via batch mode)

```
#### Interactive mode

- Front-end calls the **Azure Container App API**
- Returns:
  - Detected panels
  - Estimated energy production
  - Metadata (location, timestamps, etc.)

#### Batch mode (Blob + Azure Function)

- User uploads satellite images to an `input/` container in **Azure Blob Storage**
- An **Azure Function** triggers and calls the API hosted in **Azure Container Apps**
- The predictions (and optional artifacts) are saved into an `output/` container

**This design demonstrates:**

- Event-driven **serverless processing**
- **Containerized ML** inference
- Clean separation between **storage, compute, and serving** layers

---

## Tech Stack

### AI & Modeling

- **YOLOv11** fine-tuned for rooftop solar panel detection  
- **Python** for model serving, preprocessing, and PV estimation logic  
- **Physics-based PV calculations** using:
  - Panel area derived from detections  
  - Local solar irradiance and meteorological data  
  - Deterministic formulas for approximate energy output  

### Cloud & Infrastructure (Azure)

- **Azure Container Apps** – hosts the API + model inference  
- **Azure Functions** – serverless trigger for Blob-based batch processing  
- **Azure Blob Storage** – input/output containers for images and predictions  

**This project highlights experience in:**

- **AI engineering** (CV, geospatial data, inference APIs)  
- **Cloud architecture** (event-driven pipelines, container apps, serverless)  
- **MLOps thinking** (clean interfaces, scalable design, production deployment mindset)

---

## Usage Modes

### 1. Interactive API / Front-End

- **Input:** address or coordinates  

- **The backend:**
  - Fetches a satellite image  
  - Runs YOLOv11 detection  
  - Applies the PV model  

- **Output (JSON):**
  - Detected panels (bounding boxes, confidence)  
  - Estimated energy production (e.g. kWh/month or kWh/year)  

### 2. Batch via Azure Blob Storage

- Upload satellite images into `input/`  
- Azure Function automatically:
  - Calls the Container App inference endpoint  
  - Writes prediction JSON (and optional artifacts) into `output/`  

> Ideal for **city-level** or **country-level** analysis, where thousands of images need to be processed.

---

## Getting Started

This repository focuses on the backend logic and cloud architecture.  
Some configurations (keys, resource names, etc.) are specific to the author’s Azure environment.

**Basic steps:**

```bash
git clone https://github.com/marcelomaiaf/solar_panels_prediction.git
cd solar_panels_prediction

# Create & activate a virtual environment, then:
pip install -r requirements.txt

```

Configure environment variables for:

Satellite imagery provider

Solar / weather data provider

Azure Storage (when using Blob/Function flow)

Model weights path (YOLOv11 checkpoint)

Run the backend locally (or via Docker) and deploy to Azure Container Apps following your preferred CI/CD flow.


