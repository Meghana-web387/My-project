import ctypes
import os
from ctypes import *

# Path to Mantra SDK library (adjust according to your installation)
# On Windows (64-bit):
dll_path = r"C:\Windows\System32\MFS100.dll"

# Load DLL
mfs100 = ctypes.WinDLL(dll_path)

# Define return types
mfs100.Init.argtypes = [c_int, c_int]
mfs100.Init.restype = c_int

mfs100.StartCapture.argtypes = [c_int, c_void_p, c_void_p]
mfs100.StartCapture.restype = c_int

mfs100.StopCapture.argtypes = []
mfs100.StopCapture.restype = c_int

mfs100.GetDeviceInfo.restype = c_char_p
mfs100.GetCaptureStatus.restype = c_char_p

# Initialize the device
ret = mfs100.Init(0, 0)
if ret != 0:
    print(f"Init Failed, Error Code: {ret}")
    exit()

print("Scanner Initialized Successfully ✅")

# Define buffer for image
class FingerData(Structure):
    _fields_ = [
        ("FingerImage", POINTER(c_ubyte)),
        ("FingerImageLength", c_int),
        ("NFIQ", c_int),
        ("Quality", c_int),
        ("Width", c_int),
        ("Height", c_int),
        ("Resolution", c_int),
    ]

finger_data = FingerData()

# Capture fingerprint (timeout 10000 ms)
ret = mfs100.StartCapture(10000, byref(finger_data), None)
if ret != 0:
    print(f"Capture Failed, Error Code: {ret}")
    mfs100.StopCapture()
    exit()

print("Fingerprint Captured ✅")

# Save image as BMP
output_file = "fingerprint.bmp"
with open(output_file, "wb") as f:
    buffer = (c_ubyte * finger_data.FingerImageLength).from_address(addressof(finger_data.FingerImage.contents))
    f.write(bytearray(buffer))

print(f"Fingerprint image saved: {output_file}")

# Stop capture
mfs100.StopCapture()
