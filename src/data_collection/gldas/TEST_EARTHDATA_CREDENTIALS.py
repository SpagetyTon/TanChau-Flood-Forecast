#!/usr/bin/env python3
import requests

# Credentials are read from EARTHDATA_USERNAME / EARTHDATA_PASSWORD or prompted at run time;
# nothing is stored in the code. This helper reuses the same credential function without displaying the password.
from run_gldas_to_daily import get_credentials, get_or_create_edl_token

username, password = get_credentials()
try:
    token = get_or_create_edl_token(username, password)
    print("\nPASS: Earthdata username/password are valid.")
    print("A Bearer token was obtained successfully.")
except Exception as e:
    print("\nFAIL:")
    print(e)
input("\nPress Enter to close...")
