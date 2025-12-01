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
