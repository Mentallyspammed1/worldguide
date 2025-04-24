// src/App.jsx
import React, { useState, useEffect, useCallback, useRef } from 'react';
import ChartComponent from './components/ChartComponent';
import ControlsComponent from './components/ControlsComponent';
import StatusComponent from './components/StatusComponent';
import LogComponent from './components/LogComponent';
import { getStatus } from './services/apiService';
import { Loader2, AlertTriangle, WifiOff, Activity } from 'lucide-react'; // Icons

// Constants
const STATUS_POLL_INTERVAL_MS = 5000; // Poll every 5 seconds (adjust as needed)
const MAX_STATUS_FAILURES = 5; // Stop polling after this many consecutive failures

function App() {
    // State to hold the entire status object from the backend
    const [statusData, setStatusData] = useState(null); // e.g., { isTradingEnabled, config, logs, balance, position, error, lastUpdate }
    // State for critical errors preventing status updates (e.g., network, backend down)
    const [globalError, setGlobalError] = useState(null);
    // State for initial loading phase
    const [isLoadingInitial, setIsLoadingInitial] = useState(true);
    // Ref for the status polling interval timer
    const statusIntervalRef = useRef(null);
    // Ref to track component mount status
    const isMounted = useRef(true);
    // Ref to track consecutive status fetch failures
    const failureCountRef = useRef(0);

    // Callback function to fetch status from the backend
    const fetchStatus = useCallback(async (isInitial = false) => {
         // console.debug(`Fetching status... (Initial: ${isInitial})`);
         // Don't show loading indicator for background refreshes unless it's the initial load
         if (!isInitial && !isLoadingInitial) {
             // Optionally add a subtle refresh indicator somewhere if desired
         }

        try {
            const response = await getStatus(); // Fetches { success: true, data: {...} } or throws error

            if (!isMounted.current) return; // Don't update state if component unmounted

            if (response && response.success && response.data) {
                setStatusData(response.data);
                failureCountRef.current = 0; // Reset failure count on success
                // Clear global error if fetch succeeds after a failure
                if (globalError) setGlobalError(null);
            } else {
                 // This case might indicate an issue with apiService returning unexpected format
                 // or backend sending success: false without error being thrown by axios
                 // Should be caught by apiService now, but handle as defensive measure
                 throw new Error(response?.error || "Received invalid status response from backend.");
            }
        } catch (err) {
            console.error("Critical Error fetching status:", err);
            if (!isMounted.current) return; // Don't update state if component unmounted

            // Set global error message based on the type of error
            // Keep existing status data visible if possible, but show the error prominently
            setGlobalError(`Failed to fetch status: ${err.message}`);
            failureCountRef.current += 1; // Increment failure count

            // Stop polling if the error seems persistent
            if (failureCountRef.current >= MAX_STATUS_FAILURES) {
                 console.error(`Stopping status polling after ${failureCountRef.current} consecutive failures.`);
                 setGlobalError(`Status polling stopped after ${failureCountRef.current} consecutive failures: ${err.message}. Please check backend and refresh manually.`);
                 if (statusIntervalRef.current) {
                     clearInterval(statusIntervalRef.current);
                     statusIntervalRef.current = null;
                 }
            }

        } finally {
             if (isMounted.current && isInitial) {
                 setIsLoadingInitial(false); // Mark initial load as complete
             }
        }
    }, [globalError]); // Include globalError dependency to potentially clear it

    // Effect for initial load and setting up the polling interval
    useEffect(() => {
        isMounted.current = true; // Set mounted ref
        failureCountRef.current = 0; // Reset failure count on mount/remount

        // Fetch status immediately when component mounts
        fetchStatus(true); // Pass true for initial load

        // Clear any previous interval timer if component re-renders unexpectedly
        if (statusIntervalRef.current) {
            clearInterval(statusIntervalRef.current);
        }

        // Set up polling interval to fetch status periodically
        // console.log(`Setting up status polling interval: ${STATUS_POLL_INTERVAL_MS}ms`);
        statusIntervalRef.current = setInterval(() => fetchStatus(false), STATUS_POLL_INTERVAL_MS);

        // Cleanup function: clear interval when component unmounts
        return () => {
            // console.log("Clearing status polling interval.");
            isMounted.current = false; // Set unmounted ref
            if (statusIntervalRef.current) {
                clearInterval(statusIntervalRef.current);
                statusIntervalRef.current = null;
            }
        };
    }, [fetchStatus]); // Rerun effect if fetchStatus function identity changes (it shouldn't with useCallback [])

    // Handler to manually trigger a status refresh (e.g., via a button)
    const handleManualRefresh = () => {
         console.log("Manual status refresh triggered.");
         setGlobalError(null); // Clear previous error display on manual refresh
         failureCountRef.current = 0; // Reset failure count
         fetchStatus(false); // Fetch immediately
         // Restart polling if it was stopped due to errors
         if (!statusIntervalRef.current) {
             console.log("Restarting status polling...");
             statusIntervalRef.current = setInterval(() => fetchStatus(false), STATUS_POLL_INTERVAL_MS);
         }
    };

    // --- Render Logic ---
    return (
        // Main container with padding and max-width for larger screens
        <div className="container mx-auto p-3 sm:p-4 md:p-6 space-y-6 max-w-7xl min-h-screen flex flex-col bg-gray-900">
            {/* Header */}
            <header className="text-center my-4 flex-shrink-0">
                 <h1 className="text-2xl sm:text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">
                     Pyrmethus Trading Automaton
                 </h1>
                 <p className="text-sm text-gray-500">CCXT Bot Interface for Termux</p>
            </header>

            <main className="flex-grow space-y-6">
                {/* Global Error Display Area */}
                 {globalError && (
                     <div className="bg-red-900 border border-red-700 text-red-200 px-4 py-3 rounded text-center flex flex-col sm:flex-row items-center justify-center shadow-lg">
                         <WifiOff className="h-5 w-5 mr-3 flex-shrink-0 mb-2 sm:mb-0"/>
                         <span className="flex-grow">{globalError}</span>
                          {/* Add refresh button if polling stopped */}
                          {!statusIntervalRef.current && (
                              <button
                                  onClick={handleManualRefresh}
                                  className="ml-0 sm:ml-4 mt-2 sm:mt-0 px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm"
                              >
                                  Retry Connection
                              </button>
                          )}
                     </div>
                 )}

                 {/* Initial Loading State */}
                 {isLoadingInitial && (
                     <div className="flex flex-col justify-center items-center text-gray-400 py-16">
                         <Loader2 className="animate-spin h-12 w-12 mb-4 text-blue-500" />
                         <p className="text-lg">Initializing Automaton Interface...</p>
                         <p className="text-sm">Connecting to backend and fetching initial status...</p>
                     </div>
                 )}

                 {/* Main Content Area - Render only after initial load attempt */}
                 {!isLoadingInitial && (
                     <>
                        {/* Controls Component */}
                        {/* Pass status safely, providing defaults if statusData is null */}
                        <ControlsComponent
                             initialConfig={statusData?.config} // Pass config object or undefined
                             isTradingEnabled={statusData?.isTradingEnabled || false} // Default to false if status is null
                             // Pass fetchStatus callback so Controls can trigger an immediate refresh after actions
                             onStatusChange={() => fetchStatus(false)}
                         />

                        {/* Chart and Status Grid */}
                         <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                             {/* Chart takes more space on larger screens */}
                             <div className="lg:col-span-2">
                                 {/* Pass relevant parts of status safely */}
                                 <ChartComponent
                                     symbol={statusData?.config?.symbol}
                                     interval={statusData?.config?.interval}
                                     position={statusData?.position} // Pass position data for reference lines
                                 />
                             </div>
                             {/* Status component */}
                             <div className="lg:col-span-1">
                                  {/* Pass the whole status object or null */}
                                  <StatusComponent statusData={statusData} />
                             </div>
                         </div>

                         {/* Log Component */}
                         {/* Pass logs array safely, defaulting to empty array */}
                         <LogComponent logs={statusData?.logs || []} />
                    </>
                )}
            </main>

             {/* Footer */}
             <footer className="text-center text-xs text-gray-600 mt-8 pb-4 flex-shrink-0">
                 <p>Disclaimer: Trading involves substantial risk. This tool is for educational and experimental purposes.</p>
                 <p>Always test thoroughly in Sandbox/Testnet mode before considering live trading.</p>
                 <p>&copy; {new Date().getFullYear()} Pyrmethus. Use at your own risk.</p>
             </footer>
        </div>
    );
}

export default App;
