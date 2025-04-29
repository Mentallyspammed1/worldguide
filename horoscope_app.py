import requests
import datetime
from flask import Flask, request, jsonify, Response

app = Flask(__name__)

# --- Configuration ---
ASTRO_API_URL = "https://aztro.sameerkumar.website/"
# List of valid signs for validation
VALID_SIGNS = [
    "aries", "taurus", "gemini", "cancer", "leo", "virgo",
    "libra", "scorpio", "sagittarius", "capricorn",
    "aquarius", "pisces"
]

# --- HTML, CSS, and JavaScript for the Front-End ---
# We embed this directly in the Python script for simplicity in this example.
# In a real app, these would be separate files.
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Neon Horoscope</title>
    <style>
        /* Basic Reset */
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            background-color: #0f0f1f; /* Darker blue/purple */
            color: #f0f; /* Bright magenta default */
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 30px;
            min-height: 100vh;
        }}

        h1 {{
            color: #0ff; /* Bright cyan */
            text-shadow: 0 0 5px #fff, 0 0 10px #fff, 0 0 15px #0ff, 0 0 20px #0ff, 0 0 25px #0ff;
            margin-bottom: 30px;
            text-align: center;
        }}

        .controls {{
            margin-bottom: 30px;
            display: flex;
            gap: 15px; /* Spacing between select and button */
            align-items: center;
            flex-wrap: wrap; /* Allow wrapping on smaller screens */
            justify-content: center;
        }}

        #sign-select, #fetch-button {{
            padding: 12px 18px;
            border: 2px solid #0ff;
            background-color: rgba(0, 0, 0, 0.4); /* Semi-transparent black */
            color: #0ff;
            font-size: 1rem;
            border-radius: 8px;
            box-shadow: 0 0 8px #0ff inset, 0 0 5px #0ff;
            cursor: pointer;
            transition: all 0.3s ease;
        }}

        #sign-select {{
             min-width: 150px; /* Ensure dropdown is wide enough */
        }}

         #sign-select option {{
             background-color: #1a1a2e; /* Dark background for options */
             color: #e0e0e0; /* Light text for options */
         }}

        #fetch-button:hover {{
             background-color: rgba(0, 255, 255, 0.2); /* Cyan tint on hover */
             box-shadow: 0 0 12px #0ff inset, 0 0 10px #0ff, 0 0 15px #fff;
             color: #fff;
        }}

        #horoscope-details {{
            margin-top: 20px;
            padding: 25px;
            border: 3px solid #f0f; /* Magenta border */
            border-radius: 15px;
            width: 90%;
            max-width: 600px; /* Increased max width */
            min-height: 150px; /* Increased min height */
            background-color: rgba(0, 0, 0, 0.5); /* More transparency */
            box-shadow: 0 0 12px #f0f inset, 0 0 15px #f0f;
            text-align: left; /* Align text left for better reading */
            color: #e0e0e0; /* Slightly off-white for readability */
            line-height: 1.7;
            font-size: 1.1rem;
            overflow-wrap: break-word; /* Wrap long words */
             transition: all 0.5s ease; /* Smooth transitions */
        }}

        #horoscope-details.loading {{
             color: #aaa;
             font-style: italic;
             text-align: center;
        }}

        #horoscope-details.error {{
            color: #ff4d4d; /* Red for errors */
            border-color: #ff4d4d;
            box-shadow: 0 0 12px #ff4d4d inset, 0 0 15px #ff4d4d;
            text-align: center;
            font-weight: bold;
        }}

        #horoscope-details p {{
            margin-bottom: 10px; /* Space between paragraphs */
        }}

        #horoscope-details p strong {{
            color: #0ff; /* Cyan for labels */
            font-weight: normal; /* Avoid double bolding */
            margin-right: 8px;
            text-shadow: 0 0 3px #0ff;
        }}

        /* Simple Responsive adjustments */
        @media (max-width: 600px) {{
            h1 {{
                font-size: 1.8rem;
            }}
            .controls {{
                flex-direction: column;
                 gap: 20px;
            }}
             #sign-select, #fetch-button {{
                 width: 80%;
                 text-align: center;
             }}
             #horoscope-details {{
                 width: 95%;
                 padding: 20px;
                 font-size: 1rem;
             }}
        }}
    </style>
</head>
<body>
    <h1>Daily Neon Horoscope</h1>

    <div class="controls">
        <select id="sign-select">
            <option value="" disabled selected>-- Select Your Sign --</option>
            <option value="aries">Aries</option>
            <option value="taurus">Taurus</option>
            <option value="gemini">Gemini</option>
            <option value="cancer">Cancer</option>
            <option value="leo">Leo</option>
            <option value="virgo">Virgo</option>
            <option value="libra">Libra</option>
            <option value="scorpio">Scorpio</option>
            <option value="sagittarius">Sagittarius</option>
            <option value="capricorn">Capricorn</option>
            <option value="aquarius">Aquarius</option>
            <option value="pisces">Pisces</option>
        </select>

        <button id="fetch-button">Get Horoscope</button>
    </div>

    <div id="horoscope-details">
        Select your sign and click the button to see your horoscope!
    </div>

    <script>
        const signSelect = document.getElementById('sign-select');
        const fetchButton = document.getElementById('fetch-button');
        const displayDiv = document.getElementById('horoscope-details');

        fetchButton.addEventListener('click', () => {
            const selectedSign = signSelect.value;

            if (!selectedSign) {
                displayDiv.textContent = 'Please select a zodiac sign first.';
                displayDiv.className = 'error'; // Apply error style
                return;
            }

            // Indicate loading state
            displayDiv.textContent = 'Generating cosmic insights...';
            displayDiv.className = 'loading'; // Apply loading style

            // Fetch from our own backend endpoint
            fetch(`/get_horoscope?sign=${selectedSign}`)
                .then(response => {
                    // Check if response status is OK (200-299)
                    if (!response.ok) {
                         // Try to parse error message from backend if available
                         return response.json().then(errData => {
                             throw new Error(errData.error || `Server error: ${response.status}`);
                         }).catch(() => {
                            // Fallback generic error if parsing backend error fails
                             throw new Error(`Server error: ${response.status}`);
                         });
                    }
                    // If response is OK, parse the JSON body
                    return response.json();
                })
                .then(data => {
                    // Reset class name
                    displayDiv.className = '';

                    // Check if backend returned an error property (even with 200 OK)
                    if (data.error) {
                         displayDiv.textContent = `Error: ${data.error}`;
                         displayDiv.className = 'error';
                    } else if (data.description) {
                        // Build the display string with HTML
                         let output = `<p>${data.description}</p>`;
                         if(data.mood) output += `<p><strong>Mood:</strong> ${data.mood}</p>`;
                         if(data.color) output += `<p><strong>Lucky Color:</strong> ${data.color}</p>`;
                         if(data.lucky_number) output += `<p><strong>Lucky Number:</strong> ${data.lucky_number}</p>`;
                         if(data.compatibility) output += `<p><strong>Compatibility:</strong> ${data.compatibility}</p>`;

                         displayDiv.innerHTML = output; // Use innerHTML to render the tags
                    } else {
                        // Handle case where description is missing but no error reported
                         displayDiv.textContent = 'Could not retrieve horoscope details from the provider.';
                          displayDiv.className = 'error';
                    }
                })
                .catch(error => {
                    console.error('Fetch error:', error);
                    // Display the caught error message
                    displayDiv.textContent = `Failed to fetch horoscope: ${error.message}. Please try again later.`;
                    displayDiv.className = 'error';
                });
        });
    </script>
</body>
</html>
"""

# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return Response(HTML_TEMPLATE, mimetype='text/html')

@app.route('/get_horoscope')
def get_horoscope_api():
    """API endpoint to fetch horoscope data from Aztro."""
    sign = request.args.get('sign')

    # --- Input Validation ---
    if not sign:
        return jsonify({"error": "Zodiac sign parameter ('sign') is required"}), 400 # Bad Request

    sign = sign.lower() # Convert to lowercase for comparison
    if sign not in VALID_SIGNS:
         return jsonify({"error": f"Invalid sign: '{sign}'. Please provide a valid sign."}), 400 # Bad Request

    # --- Fetch data from External API ---
    try:
        # Aztro API requires a POST request, passing sign and day
        api_params = {'sign': sign, 'day': 'today'}
        response = requests.post(ASTRO_API_URL, params=api_params, timeout=10) # Added timeout

        # Check if the external API call failed
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

        horoscope_data = response.json()

        # --- Return relevant data to our front-end ---
        # Select specific fields we want to send back
        return jsonify({
            "description": horoscope_data.get("description"),
            "mood": horoscope_data.get("mood"),
            "color": horoscope_data.get("color"),
            "lucky_number": horoscope_data.get("lucky_number"),
            "lucky_time": horoscope_data.get("lucky_time"),
            "compatibility": horoscope_data.get("compatibility"),
            "date_range": horoscope_data.get("date_range"),
            "current_date": horoscope_data.get("current_date")
            # Add/remove fields as needed
        })

    # --- Error Handling ---
    except requests.exceptions.Timeout:
        print(f"Error: Timeout connecting to Aztro API for sign: {sign}")
        return jsonify({"error": "The horoscope provider took too long to respond. Please try again later."}), 504 # Gateway Timeout
    except requests.exceptions.HTTPError as http_err:
         # Handle HTTP errors (like 404 Not Found, 500 Internal Server Error from Aztro)
         print(f"HTTP error occurred fetching from Aztro API for sign {sign}: {http_err}") # Server-side log
         # Try to give a more specific error if possible
         status_code = http_err.response.status_code
         if status_code == 404:
             error_msg = "Horoscope data not found for the specified sign or day from the provider."
         elif status_code >= 500:
              error_msg = "The horoscope provider is experiencing technical difficulties. Please try again later."
         else:
              error_msg = "An error occurred communicating with the horoscope provider."
         return jsonify({"error": error_msg}), 502 # Bad Gateway (indicates problem with upstream server)
    except requests.exceptions.RequestException as req_err:
        # Handle other network-related errors (DNS failure, connection refused, etc.)
        print(f"Network error fetching from Aztro API for sign {sign}: {req_err}") # Server-side log
        return jsonify({"error": "Could not connect to the horoscope provider. Check your network connection."}), 503 # Service Unavailable
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred on the server: {e}") # Server-side log
        return jsonify({"error": "An internal server error occurred."}), 500 # Internal Server Error


# --- Run the Flask App ---
if __name__ == '__main__':
    # Runs the development server.
    # For production, use a proper WSGI server like Gunicorn or Waitress.
    # host='0.0.0.0' makes it accessible on your network, not just localhost.
    # Use debug=False for production.
    # Use port 5000 by default, you can change it if needed.
    print("Starting Flask server...")
    print("Access the app at: http://127.0.0.1:5000/ (or your local IP)")
    app.run(host='0.0.0.0', port=5000, debug=True)