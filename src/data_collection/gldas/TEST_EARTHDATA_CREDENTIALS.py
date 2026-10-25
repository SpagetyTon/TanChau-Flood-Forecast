#!/usr/bin/env python3
import requests

# Password is embedded in run_gldas_to_daily.py as requested.
# This helper imports the same credential function without displaying the password.
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
