import requests
import json # Optional: for pretty-printing the JSON output

# --- WARNING: Hardcoded API Key - Significant Security Risk! ---
# This key is exposed directly in the code. Avoid this in production
# or shared environments. Consider environment variables instead.
API_KEY = '7a99e9cdebfa2bb28f57a6f6da095b96f14651a44ab38fe92a56d9ff027c20ee'
# --- End Warning ---

URL = "https://api.scraperx.com/api/v1/country/industry"

headers = {
  'Accept': 'application/json',
  'x-api-key': API_KEY  # Using the hardcoded key
}

print(f"Attempting to fetch data from: {URL}")
print(f"Using API Key ending in: ...{API_KEY[-6:]}") # Print last few chars for confirmation, slightly safer than full key

try:
    # Send the GET request using the requests library
    response = requests.get(URL, headers=headers, timeout=30) # Added a timeout

    # Raise an exception if the request returned an error status code (4xx or 5xx)
    response.raise_for_status()

    # Try to parse the JSON response
    data = response.json()

    # Print the JSON data nicely formatted
    print("\n--- API Response ---")
    print(json.dumps(data, indent=2))
    print("--- End Response ---")

# Handle potential errors during the request (network issues, DNS errors, etc.)
except requests.exceptions.RequestException as e:
    print(f"\n--- ERROR: Request failed ---")
    print(f"An error occurred: {e}")
    # If an error occurred, the response object might exist but contain error details
    if 'response' in locals() and response is not None:
        print(f"Status Code: {response.status_code}")
        print(f"Response Body: {response.text}") # Print raw text if JSON parsing failed or wasn't attempted

# Handle potential errors if the response is not valid JSON
except json.JSONDecodeError:
    print(f"\n--- ERROR: Failed to decode JSON ---")
    print(f"Status Code: {response.status_code}")
    print("The response from the server was not valid JSON.")
    print(f"Raw Response Body:\n{response.text}")

except Exception as e:
    print(f"\n--- ERROR: An unexpected error occurred ---")
    print(f"{e}")
