// src/components/ChartComponent.jsx
import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine
} from 'recharts';
import { getOhlcv } from '../services/apiService';
import { Loader2, AlertTriangle, BarChart } from 'lucide-react'; // Icons

// Helper to format timestamp for XAxis
const formatXAxis = (timestamp) => {
    try {
        // Show Date if the range is large, otherwise just time
        // This is a basic heuristic, could be improved based on actual time range
        // return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
         return new Date(timestamp).toLocaleTimeString(); // Default locale time format
    } catch (e) { return ''; }
};

// Helper to format price for YAxis and Tooltip
const formatPrice = (price) => {
    if (typeof price !== 'number' || isNaN(price)) return 'N/A';
    // Dynamic precision based on price magnitude (basic example)
    if (price >= 1000) return price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    if (price >= 10) return price.toLocaleString(undefined, { minimumFractionDigits: 3, maximumFractionDigits: 3 });
    if (price >= 0.1) return price.toLocaleString(undefined, { minimumFractionDigits: 4, maximumFractionDigits: 4 });
    return price.toLocaleString(undefined, { minimumFractionDigits: 5, maximumFractionDigits: 5 }); // Higher precision for small prices
};

// Custom Tooltip Component for better formatting
const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload; // Access the full data point for the tooltip
    return (
      <div className="bg-gray-800 bg-opacity-90 border border-gray-600 rounded p-2 text-xs shadow-lg">
        <p className="font-semibold text-gray-200 mb-1">{`Time: ${formatXAxis(data.timestamp)}`}</p>
        {payload.map((entry, index) => (
          <p key={`item-${index}`} style={{ color: entry.color }}>
            {`${entry.name}: ${formatPrice(entry.value)}`}
          </p>
        ))}
        {/* Optionally add Open, High, Low, Volume */}
        {/* <p className="text-gray-400 mt-1">{`O: ${formatPrice(data.open)} H: ${formatPrice(data.high)} L: ${formatPrice(data.low)}`}</p> */}
        {/* <p className="text-gray-400">{`Vol: ${data.volume?.toLocaleString()}`}</p> */}
      </div>
    );
  }
  return null;
};


const ChartComponent = ({ symbol, interval, position }) => {
    const [chartData, setChartData] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const intervalIdRef = useRef(null); // Ref for the fetch interval timer
    const isMounted = useRef(true); // Track mount status

    const fetchData = useCallback(async (isInitialLoad = false) => {
        // Don't fetch if symbol or interval is missing, or component unmounted
        if (!symbol || !interval || !isMounted.current) {
            setChartData([]); // Clear data if params are missing or unmounted
            setError(null);
            setIsLoading(false);
            return;
        }

        if (isInitialLoad) setIsLoading(true);
        // Don't clear the error immediately on refresh, only if fetch is successful
        // setError(null);

        try {
            // console.debug(`Fetching chart data for ${symbol}/${interval}...`);
            const response = await getOhlcv(symbol, interval, 200); // Fetch last 200 candles

            if (!isMounted.current) return; // Check again after await

            if (response && response.success && Array.isArray(response.data)) {
                 // Format data for recharts, ensuring numeric types
                 const formattedData = response.data.map(d => ({
                    timestamp: d.timestamp, // Keep original timestamp for calculations/tooltip
                    time: formatXAxis(d.timestamp), // Formatted time string for axis label (might be redundant if tickFormatter used)
                    open: Number(d.open),
                    high: Number(d.high),
                    low: Number(d.low),
                    close: Number(d.close),
                    volume: Number(d.volume),
                })).sort((a, b) => a.timestamp - b.timestamp); // Ensure data is sorted by time

                setChartData(formattedData);
                setError(null); // Clear error on successful fetch
            } else {
                 // Handle cases where backend returns success: false or invalid data structure
                 throw new Error(response?.error || "Invalid data structure received from API");
            }
        } catch (err) {
             if (!isMounted.current) return; // Check again after await in catch
            console.error(`Chart data fetch error for ${symbol}/${interval}:`, err);
            // Keep showing the last known data if available, but display the error
            setError(err.message || "Failed to load chart data. Check backend connection or symbol/interval validity.");
        } finally {
             if (!isMounted.current) return; // Check again after await in finally
            // Only stop the main loading indicator on the initial load attempt
            if (isInitialLoad) setIsLoading(false);
        }
    }, [symbol, interval]); // Dependencies: refetch if symbol or interval changes

    // Effect to handle fetching data on mount, on param change, and periodically
    useEffect(() => {
         isMounted.current = true; // Set mount status on effect run

        // Clear any existing interval timer when dependencies change
        if (intervalIdRef.current) {
            clearInterval(intervalIdRef.current);
            intervalIdRef.current = null;
        }

        // Fetch immediately when component mounts or symbol/interval changes
        fetchData(true); // Pass true for initial load to show loader

        // Set up polling interval for refreshing data (only if symbol/interval are set)
        if (symbol && interval) {
            const refreshIntervalMs = 30000; // Refresh every 30 seconds
            const newIntervalId = setInterval(() => fetchData(false), refreshIntervalMs);
            intervalIdRef.current = newIntervalId;
            // console.debug(`Chart polling started for ${symbol}/${interval} every ${refreshIntervalMs}ms`);
        }

        // Cleanup function: called when component unmounts or dependencies change
        return () => {
            isMounted.current = false; // Clear mount status
            if (intervalIdRef.current) {
                // console.debug(`Chart polling stopped for ${symbol}/${interval}`);
                clearInterval(intervalIdRef.current);
                intervalIdRef.current = null;
            }
        };
    }, [fetchData, symbol, interval]); // Rerun effect if fetchData function or params change

     // Memoize reference lines to prevent unnecessary re-renders if position object reference changes but values don't
     const referenceLines = useMemo(() => {
         const lines = [];
         const entryPrice = Number(position?.entryPrice);
         const liqPrice = Number(position?.liquidationPrice);

         if (position && !isNaN(entryPrice)) {
             lines.push(
                 <ReferenceLine key="entry"
                     yAxisId="left" y={entryPrice}
                     label={{ value: `Entry ${formatPrice(entryPrice)}`, position: 'insideRight', fill: '#a0aec0', fontSize: 9 }}
                     stroke={position.side === 'long' ? '#2dd4bf' : '#f87171'} // teal-400 for long, red-400 for short
                     strokeDasharray="4 2" strokeWidth={1} ifOverflow="extendDomain"
                 />
             );
         }
         if (position && !isNaN(liqPrice) && liqPrice > 0) {
             lines.push(
                 <ReferenceLine key="liq"
                     yAxisId="left" y={liqPrice}
                     label={{ value: `Liq ${formatPrice(liqPrice)}`, position: 'insideRight', fill: '#fb923c', fontSize: 9 }} // orange-400
                     stroke="#f97316" // orange-500
                     strokeDasharray="4 2" strokeWidth={1} ifOverflow="extendDomain"
                 />
             );
         }
          // Add lines for SL/TP if available and passed as props
         return lines;
     }, [position]); // Re-calculate only when position object changes


    // --- Render Logic ---

    // Display loader only on the very first load attempt
    if (isLoading && chartData.length === 0) {
        return (
            <div className="flex justify-center items-center h-64 md:h-96 text-gray-400 bg-gray-800 rounded-md shadow-lg p-4">
                <Loader2 className="animate-spin h-8 w-8 mr-3" /> Loading Chart Data...
            </div>
        );
    }

    // Display message if no symbol/interval selected
    if (!symbol || !interval) {
         return (
            <div className="flex flex-col justify-center items-center h-64 md:h-96 text-gray-500 bg-gray-800 rounded-md shadow-lg p-4">
                <BarChart className="h-12 w-12 mb-3" />
                Select Symbol and Interval to display the chart.
            </div>
        );
    }

    // Main chart rendering
    return (
        <div className="h-64 md:h-96 w-full bg-gray-800 p-3 rounded-md shadow-lg relative text-xs">
            {/* Error Overlay */}
            {error && (
                <div className="absolute top-2 left-2 right-2 z-20 bg-red-700 bg-opacity-90 text-white p-2 rounded text-xs flex items-center shadow-md">
                    <AlertTriangle className="h-4 w-4 mr-2 flex-shrink-0" />
                    <span>Chart Error: {error}</span>
                </div>
            )}

             {/* Loading Indicator during refresh (subtle) */}
             {isLoading && chartData.length > 0 && (
                 <div className="absolute top-2 right-2 z-10 p-1 bg-gray-700 bg-opacity-70 rounded-full" title="Refreshing chart data...">
                     <Loader2 className="animate-spin h-4 w-4 text-gray-300" />
                 </div>
             )}


            {/* Chart Area - Only render if data exists */}
            {chartData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 5, right: 15, left: -10, bottom: 5 }}>
                        {/* Background Grid */}
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" /> {/* gray-700 */}

                        {/* X Axis (Time) */}
                        <XAxis
                            dataKey="timestamp" // Use raw timestamp for correct scaling
                            fontSize={10}
                            stroke="#9ca3af" /* gray-400 */
                            tick={{ fill: '#9ca3af' }}
                            tickFormatter={formatXAxis} // Format the timestamp for display
                            interval="preserveStartEnd" // Adjust interval dynamically based on data? 'auto' might work
                            // Example: Tick every 10 candles: interval={Math.floor(chartData.length / 10)}
                            minTickGap={40} // Minimum gap between ticks in pixels
                        />

                        {/* Y Axis (Price) */}
                        <YAxis
                            yAxisId="left" // Assign an ID if using multiple Y axes
                            fontSize={10}
                            stroke="#9ca3af" /* gray-400 */
                            tick={{ fill: '#9ca3af' }}
                            domain={['auto', 'auto']} // Auto-scale domain
                            tickFormatter={formatPrice} // Use custom price formatter
                            orientation="left"
                            width={55} // Allocate space for labels
                            allowDataOverflow={false} // Prevent lines going outside axis boundaries
                            scale="linear" // Use linear scale for price
                        />

                        {/* Tooltip on Hover */}
                        <Tooltip content={<CustomTooltip />} cursor={{ stroke: '#6b7280', strokeWidth: 1 }} />

                        {/* Legend */}
                        <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />

                        {/* Price Line */}
                        <Line
                            yAxisId="left"
                            type="monotone" // Or "linear"
                            dataKey="close"
                            name="Price"
                            stroke="#3b82f6" /* blue-500 */
                            strokeWidth={2}
                            dot={false}
                            isAnimationActive={false} // Disable animation for performance on frequent updates
                        />

                         {/* Render memoized reference lines */}
                         {referenceLines}

                        {/* Add other indicator lines here if data is available */}
                        {/* Example: <Line yAxisId="left" type="monotone" dataKey="ema" name="EMA" stroke="#f59e0b" strokeWidth={1} dot={false} /> */}

                    </LineChart>
                </ResponsiveContainer>
            ) : (
                // Display message if no data is available after loading attempt (and no error)
                !isLoading && !error && (
                    <div className="flex flex-col justify-center items-center h-full text-gray-500">
                         <BarChart className="h-12 w-12 mb-3" />
                         No chart data available for {symbol} ({interval}).
                    </div>
                )
            )}
        </div>
    );
};

// Use React.memo to prevent re-renders if props haven't changed shallowly
// Useful if parent component re-renders frequently but chart props remain the same
export default React.memo(ChartComponent);
