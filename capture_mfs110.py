import ctypes
import time
import os

# 📍 Load the Mantra MFS110 DLL
dll_path = os.path.abspath("MantraMFS110API.dll")  # Adjust if needed
mfs110 = ctypes.CDLL(dll_path)

# 👇 Define buffer and required structures
BUFFER_SIZE = 512 * 512  # 262144 bytes max for image

# Allocate memory buffers
image_buffer = ctypes.create_string_buffer(BUFFER_SIZE)
iso_template = ctypes.create_string_buffer(1024)
quality = ctypes.c_int()
nfiq = ctypes.c_int()

# 📦 Device Initialization
ret = mfs110.Init()
if ret != 0:
    print("✅ Device initialized successfully.")
else:
    print("❌ Device initialization failed.")
    exit()

# 🔄 Start Capture
print("👉 Place your finger on the scanner...")
ret = mfs110.CaptureRaw(0, image_buffer, BUFFER_SIZE, ctypes.byref(quality), ctypes.byref(nfiq))

if ret == 0:
    print("✅ Fingerprint captured successfully.")
    print("📈 Quality:", quality.value, "| NFIQ:", nfiq.value)

    # Save fingerprint image as raw
    with open("fingerprint.raw", "wb") as f:
        f.write(image_buffer.raw)
    print("💾 Saved raw fingerprint image to 'fingerprint.raw'.")

else:
    print("❌ Failed to capture fingerprint. Error code:", ret)

# 🔚 Uninitialize
mfs110.Uninit()
