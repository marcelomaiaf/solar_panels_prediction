import io
import logging
import os

import azure.functions as func
import requests
from azure.storage.blob import BlobServiceClient

app = func.FunctionApp()


_conn = os.getenv("mldata_STORAGE")
_svc = BlobServiceClient.from_connection_string(_conn) if _conn else None

def upload_blob_stream(blob_service_client: BlobServiceClient, container_name: str, blob_name: str, data: bytes) -> None:
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_client.upload_blob(data, overwrite=True)

API_URL = "https://solar-predict.purplebeach-dbb8ddaa.swedencentral.azurecontainerapps.io/pipeline/"
DEFAULT_PARAMS = {
    "zoom": 20,
    "adress": "rua professora aldeci barbosa, 79, cambeba, Brasil",
}

def call_api(img: bytes, url: str = API_URL, params: dict = DEFAULT_PARAMS):
    verify = os.getenv("REQUESTS_VERIFY", "true").lower() not in ("0", "false", "no")
    files = {"img": ("blob.png", img, "image/png")}
    response = requests.post(
        url,
        params=params,
        files=files,
        timeout=(60, 300),  
        verify=verify,
        proxies={"http": None, "https": None},
    )
   
    if response.status_code >= 400:
        logging.error(f"API error: status={response.status_code} body={response.text[:500]}")
    response.raise_for_status()
    return response

@app.blob_trigger(arg_name="myblob", path="input/{file}.png", connection="mldata_STORAGE")
def blob_trigger(myblob: func.InputStream):
    logging.info(
        f"Python blob trigger processed: Name={myblob.name} Size={myblob.length} bytes"
    )

    data = myblob.read()
    logging.info(myblob.name)
    
    
    try:
        resp = call_api(data)
        logging.info(f"API response status: {resp.status_code}")
    except requests.HTTPError as e:
        
        status = e.response.status_code if e.response is not None else "unknown"
        body = e.response.text[:500] if e.response is not None else ""
        logging.error(f"API HTTPError: status={status} body={body}")
        return

    file_name = myblob.name.replace("input/","")
    file_name = file_name.replace(".png","")
    svc = _svc or BlobServiceClient.from_connection_string(os.getenv("mldata_STORAGE"))
    upload_blob_stream(blob_service_client=svc, container_name="output", blob_name=f"{file_name}-result.png", data=resp)
    


