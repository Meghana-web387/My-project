# mantra_bridge.py - DIAGNOSTIC VERSION 2

import requests
import xml.etree.ElementTree as ET
import base64
import logging

logger = logging.getLogger(__name__)

class MantraCaptureError(Exception):
    """Custom exception for Mantra device capture errors."""
    pass

def capture_fp_base64(quality=70, timeout_ms=10000):
    
    print("--- 1. Starting RD Service Info Check ---")
    
    RD_SERVICE_PORTS = [11100, 11101, 11102, 11103, 11104]
    capture_xml = f"""<?xml version="1.0"?>
    <PidOptions ver="1.0">
        <Opts fCount="1" fType="0" iCount="0" pCount="0" format="1" pidVer="2.0" timeout="{timeout_ms}" posh="UNKNOWN" env="P" wadh=""/>
        <Demo/>
        <CustOpts/>
    </PidOptions>"""
    headers = {
        "Content-Type": "application/xml; charset=UTF-8",
        "Accept": "application/xml"
    }

    found_service_url = None
    for port in RD_SERVICE_PORTS:
        try:
            response = requests.request("RDSERVICE", f"http://localhost:{port}/rd/info", headers=headers, timeout=5)
            if response.status_code == 200:
                found_service_url = f"http://localhost:{port}/rd/capture"
                print(f"Service found at http://localhost:{port}/rd/info")
                break
        except requests.exceptions.RequestException:
            continue
            
    if not found_service_url:
        raise MantraCaptureError("Could not connect to Mantra RD Service. Please ensure the device is connected and the service is running.")

    print("--- 2. Sending Capture Request to RD Service ---")
    try:
        response = requests.request("RDSERVICE", found_service_url, data=capture_xml.encode('utf-8'), headers=headers, timeout=(timeout_ms / 1000) + 5)
        response.raise_for_status()
        
        # --- THIS IS THE FINAL LOG WE NEED TO SEE ---
        print("--- 3. RECEIVED CAPTURE RESPONSE (Raw XML Below) ---")
        print(response.content.decode('utf-8'))
        print("-------------------------------------------------")
        
        root = ET.fromstring(response.content)
        data_element = root.find(".//Data")
        if data_element is None:
            resp_element = root.find(".//Resp")
            err_info = resp_element.get("errInfo") if resp_element is not None else "Unknown error"
            raise MantraCaptureError(f"Capture failed: {err_info}")
            
        img_b64 = data_element.text
        fmr_element = root.find(".//Fmr")
        if fmr_element is not None:
            w = int(fmr_element.get("w"))
            h = int(fmr_element.get("h"))
        else:
            w, h = None, None

        return img_b64, w, h
    
    except requests.exceptions.RequestException as e:
        raise MantraCaptureError(f"Network or request error: {e}")
    except ET.ParseError as e:
        raise MantraCaptureError(f"Failed to parse XML response: {e}")
    except Exception as e:
        raise MantraCaptureError(f"An unexpected error occurred during capture: {e}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    try:
        print("Attempting to capture fingerprint...")
        img_data, width, height = capture_fp_base64()
        if img_data:
            print(f"Capture successful! Image data (Base64) received. Width: {width}, Height: {height}")
        else:
            print("Capture returned no data.")
            
    except MantraCaptureError as e:
        print(f"Error: {e}")