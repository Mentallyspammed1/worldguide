// src/services/apiService.js
import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL;

// Critical check: Ensure the API URL is defined in the environment.
if (!API_URL) {
    const errorMsg = "FATAL ERROR: REACT_APP_API_URL is not defined. Check your frontend '.env' file and ensure it's built correctly.";
    console.error(errorMsg);
    // Display this error prominently in the UI if possible,
    // otherwise the app will fail on the first API call.
    // Replace alert with a more integrated UI error message in a real app.
    alert(errorMsg); // Simple alert for immediate feedback during development
    throw new Error(errorMsg);
} else {
    console.info(`[apiService] Using API URL: ${API_URL}`);
}


// Create an Axios instance with base URL and default settings
const apiClient = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
    },
    timeout: 20000, // Set request timeout (20 seconds)
});

// --- Centralized API Error Handling ---
/**
 * Handles errors from API calls, logging details and extracting useful messages.
 * @param {Error} error - The error object caught from Axios.
 * @param {string} functionName - The name of the API function where the error occurred.
 * @throws {Error} Re-throws a structured error message.
 */
const handleApiError = (error, functionName) => {
    console.error(`API Error in ${functionName}:`, error);
    let errorMessage = `An unexpected error occurred in ${functionName}.`;
    let isNetworkError = false;

    if (error.response) {
        // Server responded with a status code outside the 2xx range
        const status = error.response.status;
        const responseData = error.response.data;
        console.error(`[${functionName}] Server Error: Status ${status}`, responseData);
        // Try to get a meaningful error message from the response body
        errorMessage = `Server Error (${status}): ${responseData?.error || responseData?.message || error.response.statusText || 'Unknown server error'}`;
        if (status === 401) errorMessage += " (Check API Key / Authentication)";
        if (status === 404) errorMessage += " (Resource not found)";
        if (status === 429) errorMessage += " (Rate Limit Exceeded)";
        if (status === 503) errorMessage += " (Service Unavailable / Maintenance)";

    } else if (error.request) {
        // Request was made but no response received (network error, backend down, CORS)
        console.error(`[${functionName}] Network Error: No response received. Request:`, error.request);
        errorMessage = 'Network Error: Cannot reach the backend server. Please check if the server is running and accessible.';
        isNetworkError = true;
        if (API_URL.startsWith('http://localhost') || API_URL.startsWith('http://127.0.0.1')) {
             errorMessage += ' Ensure the backend is running locally.';
        } else {
             errorMessage += ' Check your network connection and the server status.';
        }
         // Check for timeout specifically
         if (error.code === 'ECONNABORTED') {
            errorMessage = `Request timed out after ${apiClient.defaults.timeout / 1000} seconds. The server might be busy or unreachable.`;
        }

    } else {
        // Error occurred in setting up the request
        console.error(`[${functionName}] Request Setup Error:`, error.message);
        errorMessage = `Request setup error: ${error.message}`;
    }

    // Throw a new error with the processed message for the UI to catch
    const processedError = new Error(errorMessage);
    processedError.isNetworkError = isNetworkError; // Add flag for UI handling
    throw processedError;
};

// --- API Service Functions ---

export const getStatus = async () => {
    try {
        const response = await apiClient.get('/status');
        // Axios throws for non-2xx status, so if we reach here, it's likely successful.
        // Return the data part of the response.
        if (!response.data || !response.data.success) {
            // Handle cases where backend returns 200 OK but with success: false
            throw new Error(response.data?.error || response.data?.message || "Status request failed on backend.");
        }
        return response.data; // { success: true, data: {...} }
    } catch (error) {
        // Let the centralized handler process and re-throw
        handleApiError(error, 'getStatus');
    }
};

export const startTrading = async (config = {}) => {
     try {
        const response = await apiClient.post('/trade/start', config);
        if (!response.data || !response.data.success) {
             throw new Error(response.data?.error || response.data?.message || "Start trading request failed on backend.");
        }
        return response.data; // { success: true, message: "..." }
    } catch (error) {
        handleApiError(error, 'startTrading');
    }
};

export const stopTrading = async () => {
     try {
        const response = await apiClient.post('/trade/stop');
         if (!response.data || !response.data.success) {
             throw new Error(response.data?.error || response.data?.message || "Stop trading request failed on backend.");
        }
        return response.data; // { success: true, message: "..." }
    } catch (error) {
        handleApiError(error, 'stopTrading');
    }
};

export const updateConfig = async (config) => {
     // Add validation here if needed before sending
     if (!config || typeof config !== 'object') {
         throw new Error("Invalid config object provided to updateConfig.");
     }
     try {
        const response = await apiClient.post('/config', config);
         if (!response.data || !response.data.success) {
             throw new Error(response.data?.error || response.data?.message || "Update config request failed on backend.");
        }
        return response.data; // { success: true, message: "...", config: {...} }
    } catch (error) {
        handleApiError(error, 'updateConfig');
    }
};

export const getSymbols = async () => {
     try {
        const response = await apiClient.get('/symbols');
         if (!response.data || !response.data.success) {
             throw new Error(response.data?.error || response.data?.message || "Get symbols request failed on backend.");
        }
        return response.data; // { success: true, data: [...] }
    } catch (error) {
        handleApiError(error, 'getSymbols');
    }
};

export const getOhlcv = async (symbol, interval, limit) => {
    // Basic validation on inputs
    if (!symbol || !interval) {
        throw new Error("Symbol and interval are required for getOhlcv.");
    }
    try {
        const response = await apiClient.get('/ohlcv', {
            params: { symbol, interval, limit }
        });
        if (!response.data || !response.data.success) {
             throw new Error(response.data?.error || response.data?.message || "Get OHLCV request failed on backend.");
        }
        return response.data; // { success: true, data: [...] }
    } catch (error) {
        // Pass identifying info to the error handler
        handleApiError(error, `getOhlcv(symbol=${symbol}, interval=${interval})`);
    }
};

// Add other API functions as needed (e.g., fetchOrderHistory, fetchMarketDetails)
