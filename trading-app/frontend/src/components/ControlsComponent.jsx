// src/components/ControlsComponent.jsx
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { startTrading, stopTrading, updateConfig, getSymbols } from '../services/apiService';
import { Play, StopCircle, Settings, Save, Loader2, AlertCircle, Info, RefreshCw } from 'lucide-react';

const ControlsComponent = ({ initialConfig, isTradingEnabled, onStatusChange }) => {
    // State for configuration form inputs
    const [config, setConfig] = useState({});
    // State for available symbols from the backend
    const [symbols, setSymbols] = useState([]);
    // Loading states
    const [isActionLoading, setIsActionLoading] = useState(false); // For Start/Stop/Update buttons
    const [isSymbolsLoading, setIsSymbolsLoading] = useState(true);
    // Error/Success messages
    const [actionError, setActionError] = useState(null);
    const [actionSuccess, setActionSuccess] = useState(null);
    // Ref to track if component is mounted
    const isMounted = useRef(true);
    // Ref for timeout clearing messages
    const messageTimeoutRef = useRef(null);

    // Available timeframes (adjust based on exchange/strategy needs)
    const timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w'];

    // Function to process and set config, ensuring numeric types
    const processAndSetConfig = useCallback((newConfig) => {
        if (newConfig && Object.keys(newConfig).length > 0) {
            const numericFields = ['leverage', 'riskPerTrade', 'atrPeriod', 'atrSlMult', 'atrTpMult', 'indicatorPeriod', 'ehlersMaPeriod', 'stochRsiK', 'stochRsiD', 'stochRsiLength', 'stochRsiStochLength'];
            const processed = { ...newConfig };
            numericFields.forEach(field => {
                if (processed[field] !== undefined && processed[field] !== null && processed[field] !== '') {
                    processed[field] = Number(processed[field]);
                    // Handle potential NaN after conversion (e.g., if input was non-numeric)
                    if (isNaN(processed[field])) {
                         console.warn(`Invalid numeric value encountered for field ${field}: ${newConfig[field]}. Setting to empty string.`);
                         processed[field] = ''; // Or set to a default, or keep original? Empty string is safer for input binding.
                    }
                } else if (processed[field] === null || processed[field] === undefined) {
                     processed[field] = ''; // Ensure controlled components have empty string instead of null/undefined
                }
            });
            setConfig(processed);
        } else {
             setConfig({}); // Reset if initialConfig is null/empty
        }
    }, []); // No dependencies, it's a pure function


    // Set initial config state when component mounts or initialConfig prop updates
    useEffect(() => {
        // console.log("ControlsComponent received initialConfig update:", initialConfig);
        processAndSetConfig(initialConfig);
    }, [initialConfig, processAndSetConfig]);


    // Fetch symbols function
    const fetchSymbols = useCallback(async () => {
         if (!isMounted.current) return; // Check mount status
         setIsSymbolsLoading(true);
         setActionError(null); // Clear previous errors on refresh attempt

         try {
             const response = await getSymbols();
             if (isMounted.current && response && response.success && Array.isArray(response.data)) {
                 setSymbols(response.data);
                 // If current config doesn't have a symbol (or it's invalid) and symbols loaded, set a default
                 // Only do this if config.symbol is actually missing or not in the new list
                 setConfig(prev => {
                     if ((!prev.symbol || !response.data.includes(prev.symbol)) && response.data.length > 0) {
                         return { ...prev, symbol: response.data[0] };
                     }
                     return prev; // No change needed
                 });
             } else if (isMounted.current) {
                  throw new Error(response?.error || "Failed to load symbols: Invalid response format.");
             }
         } catch (err) {
             if (isMounted.current) {
                 console.error("Error fetching symbols:", err);
                 setActionError(`Could not load symbols: ${err.message}`);
                 setSymbols([]); // Ensure symbols list is empty on error
             }
         } finally {
             if (isMounted.current) {
                 setIsSymbolsLoading(false);
             }
         }
    }, []); // Empty dependency array means this function identity is stable


    // Fetch symbols when component mounts
    useEffect(() => {
        isMounted.current = true; // Mark as mounted
        fetchSymbols(); // Initial fetch

        // Cleanup function for when component unmounts
        return () => {
            isMounted.current = false; // Mark as unmounted
            // Clear message timeout on unmount
            if (messageTimeoutRef.current) {
                clearTimeout(messageTimeoutRef.current);
            }
        };
    }, [fetchSymbols]); // Depend on fetchSymbols


    // Handle changes in form inputs
    const handleInputChange = (e) => {
        const { name, value, type, checked } = e.target;
        let processedValue;

        if (type === 'checkbox') {
            processedValue = checked;
        } else if (type === 'number') {
            // Allow empty string for clearing, otherwise parse as float
            processedValue = value === '' ? '' : parseFloat(value);
        } else {
            processedValue = value;
        }

        setConfig(prev => ({ ...prev, [name]: processedValue }));
        // Clear action messages when user starts editing config
        setActionSuccess(null);
        setActionError(null);
        if (messageTimeoutRef.current) {
            clearTimeout(messageTimeoutRef.current);
        }
    };


    // Function to display and auto-clear feedback messages
    const showFeedback = (message, type = 'success') => {
         if (messageTimeoutRef.current) {
            clearTimeout(messageTimeoutRef.current);
        }
        if (type === 'success') {
            setActionSuccess(message);
            setActionError(null);
        } else {
            setActionError(message);
            setActionSuccess(null);
        }
        // Auto-clear after 5 seconds
        messageTimeoutRef.current = setTimeout(() => {
            if (isMounted.current) {
                setActionError(null);
                setActionSuccess(null);
            }
        }, 5000);
    };


    // Generic handler for API actions (Start, Stop, Update)
    const handleAction = useCallback(async (actionFn, actionName, payload = null) => {
        setIsActionLoading(true);
        setActionError(null); // Clear previous messages immediately
        setActionSuccess(null);
         if (messageTimeoutRef.current) {
            clearTimeout(messageTimeoutRef.current);
        }

        try {
            // Prepare payload, ensuring numeric types for config update
            let finalPayload = payload;
            if (actionName === 'updateConfig' && typeof payload === 'object') {
                 finalPayload = { ...payload }; // Clone to avoid mutating state directly
                const numericFields = ['leverage', 'riskPerTrade', 'atrPeriod', 'atrSlMult', 'atrTpMult', 'indicatorPeriod', 'ehlersMaPeriod', 'stochRsiK', 'stochRsiD', 'stochRsiLength', 'stochRsiStochLength'];
                let validationError = null;
                numericFields.forEach(field => {
                    if (finalPayload[field] !== undefined && finalPayload[field] !== null && finalPayload[field] !== '') {
                         const numVal = Number(finalPayload[field]);
                         if (isNaN(numVal)) {
                             validationError = `Invalid number format for field: ${field} ("${finalPayload[field]}")`;
                         } else {
                            finalPayload[field] = numVal; // Use the converted number
                         }
                    } else if (finalPayload[field] === '') {
                         // Decide how to handle empty strings - remove them or send as null/0?
                         // Let's remove them to let backend use defaults if applicable
                         delete finalPayload[field];
                    }
                });
                 if (validationError) {
                     throw new Error(validationError); // Throw validation error before API call
                 }
            }


            const result = await actionFn(finalPayload); // Pass payload if needed (e.g., config)

            if (isMounted.current) { // Check mount status after await
                 if (result && result.success) {
                     showFeedback(result.message || `${actionName} successful.`, 'success');
                     // Trigger status refresh in the parent component
                     if (onStatusChange) onStatusChange();
                     // If config was updated successfully, the parent will eventually pass down the new initialConfig
                 } else {
                     // Handle cases where backend returns success: false (already handled by apiService throwing an error)
                     // This case should ideally not be reached if apiService throws correctly
                     throw new Error(result?.error || result?.message || `${actionName} failed with no specific error message.`);
                 }
             }
        } catch (err) {
             console.error(`Action error (${actionName}):`, err);
             if (isMounted.current) { // Check mount status after await
                 showFeedback(err.message || `An unknown error occurred during ${actionName}.`, 'error');
             }
        } finally {
            // Check mount status before setting state in async callback
            if (isMounted.current) {
                 setIsActionLoading(false);
            }
        }
    }, [onStatusChange, showFeedback]); // Include dependencies

    // Determine if the form is valid for starting/updating
    // Check for non-empty string/number for key fields
    const isFormValid = config.symbol && config.interval &&
                        (typeof config.leverage === 'number' && config.leverage > 0) &&
                        (typeof config.riskPerTrade === 'number' && config.riskPerTrade > 0); // Add more checks as needed

    return (
        <div className="p-4 border border-gray-700 rounded-md space-y-4 bg-gray-800 shadow-lg">
            <div className="flex justify-between items-center">
                 <h3 className="text-lg font-semibold text-gray-200">Controls & Configuration</h3>
                 {/* Trading Status Indicator */}
                 <div className={`text-sm font-semibold flex items-center px-3 py-1 rounded ${isTradingEnabled ? 'bg-green-800 text-green-200' : 'bg-yellow-800 text-yellow-200'}`}>
                    <span className={`inline-block h-2.5 w-2.5 rounded-full mr-2 ${isTradingEnabled ? 'bg-green-400 animate-pulse' : 'bg-yellow-400'}`}></span>
                    {isTradingEnabled ? 'Trading: ACTIVE' : 'Trading: STOPPED'}
                 </div>
            </div>

            {/* Action Feedback Area */}
            {actionError && (
               <div className="bg-red-900 border border-red-700 text-red-200 px-3 py-2 rounded text-sm flex items-center">
                   <AlertCircle className="h-4 w-4 mr-2 flex-shrink-0"/> Error: {actionError}
               </div>
             )}
            {actionSuccess && (
                <div className="bg-green-900 border border-green-700 text-green-200 px-3 py-2 rounded text-sm flex items-center">
                    <Info className="h-4 w-4 mr-2 flex-shrink-0"/> Success: {actionSuccess}
                </div>
              )}

             {/* Configuration Form Grid */}
             {/* Wrap in a form element for semantics, prevent default submission */}
             <form onSubmit={(e) => e.preventDefault()} className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-x-4 gap-y-3">
                  {/* Helper function for input fields */}
                  {renderInputField("Symbol", "symbol", config.symbol || '', handleInputChange, { type: 'select', options: symbols, isLoading: isSymbolsLoading, disabled: isActionLoading || isTradingEnabled, required: true, addon: !isSymbolsLoading && <button onClick={fetchSymbols} disabled={isSymbolsLoading || isActionLoading || isTradingEnabled} title="Refresh Symbols" className="p-1 text-gray-400 hover:text-white disabled:opacity-50"><RefreshCw size={14} /></button> })}
                  {renderInputField("Interval", "interval", config.interval || '', handleInputChange, { type: 'select', options: timeframes, disabled: isActionLoading || isTradingEnabled, required: true })}
                  {renderInputField("Leverage", "leverage", config.leverage ?? '', handleInputChange, { type: 'number', min: 1, step: 1, placeholder: "e.g., 10", disabled: isActionLoading || isTradingEnabled, required: true })}
                  {renderInputField("Risk %", "riskPerTrade", config.riskPerTrade ?? '', handleInputChange, { type: 'number', min: 0.0001, max: 0.1, step: 0.001, placeholder: "0.005 (0.5%)", disabled: isActionLoading || isTradingEnabled, required: true })}
                  {renderInputField("ATR Period", "atrPeriod", config.atrPeriod ?? '', handleInputChange, { type: 'number', min: 1, step: 1, placeholder: "e.g., 14", disabled: isActionLoading || isTradingEnabled })}
                  {renderInputField("ATR SL Mult", "atrSlMult", config.atrSlMult ?? '', handleInputChange, { type: 'number', min: 0.1, step: 0.1, placeholder: "e.g., 1.5", disabled: isActionLoading || isTradingEnabled })}
                  {renderInputField("ATR TP Mult", "atrTpMult", config.atrTpMult ?? '', handleInputChange, { type: 'number', min: 0.1, step: 0.1, placeholder: "e.g., 1.5", disabled: isActionLoading || isTradingEnabled })}
                  {renderInputField("Indic. Period", "indicatorPeriod", config.indicatorPeriod ?? '', handleInputChange, { type: 'number', min: 1, step: 1, placeholder: "e.g., 14", disabled: isActionLoading || isTradingEnabled })}
                  {renderInputField("Ehlers MA Pd", "ehlersMaPeriod", config.ehlersMaPeriod ?? '', handleInputChange, { type: 'number', min: 1, step: 1, placeholder: "e.g., 10", disabled: isActionLoading || isTradingEnabled })}
                  {renderInputField("Stoch RSI K", "stochRsiK", config.stochRsiK ?? '', handleInputChange, { type: 'number', min: 1, step: 1, placeholder: "e.g., 3", disabled: isActionLoading || isTradingEnabled })}
                  {renderInputField("Stoch RSI D", "stochRsiD", config.stochRsiD ?? '', handleInputChange, { type: 'number', min: 1, step: 1, placeholder: "e.g., 3", disabled: isActionLoading || isTradingEnabled })}
                  {renderInputField("Stoch RSI Len", "stochRsiLength", config.stochRsiLength ?? '', handleInputChange, { type: 'number', min: 1, step: 1, placeholder: "e.g., 14", disabled: isActionLoading || isTradingEnabled })}
                  {/* Add Stoch RSI Stoch Length if needed */}
             </form>

            {/* Action Buttons Area */}
            <div className="flex flex-wrap gap-3 pt-4 items-center border-t border-gray-700 mt-4">
                 {/* Update Config Button */}
                 <button
                     type="button" // Important: prevent form submission
                     onClick={() => handleAction(updateConfig, 'updateConfig', config)}
                     disabled={isActionLoading || isTradingEnabled || !isFormValid} // Disable if trading, loading, or form invalid
                     className="inline-flex items-center px-3 py-2 border border-gray-600 shadow-sm text-sm font-medium rounded-md text-gray-300 bg-gray-700 hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-900 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                     title={isTradingEnabled ? "Cannot update config while trading is active" : !isFormValid ? "Fill all required fields (Symbol, Interval, Leverage, Risk)" : "Save current configuration"}
                 >
                    {isActionLoading ? <Loader2 className="animate-spin -ml-1 mr-2 h-5 w-5" /> : <Save className="-ml-1 mr-2 h-5 w-5" />}
                    Update Config
                </button>

                 {/* Start Trading Button */}
                 <button
                     type="button"
                     onClick={() => handleAction(startTrading, 'startTrading', config)} // Send current config state
                     disabled={isActionLoading || isTradingEnabled || !isFormValid || isSymbolsLoading}
                     className="inline-flex items-center px-3 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-900 focus:ring-green-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                     title={isTradingEnabled ? "Trading is already active" : !isFormValid ? "Fill all required fields" : isSymbolsLoading ? "Waiting for symbols to load..." : "Start trading with current config"}
                 >
                    {isActionLoading ? <Loader2 className="animate-spin -ml-1 mr-2 h-5 w-5" /> : <Play className="-ml-1 mr-2 h-5 w-5" />}
                     Start Trading
                </button>

                 {/* Stop Trading Button */}
                 <button
                     type="button"
                     onClick={() => handleAction(stopTrading, 'stopTrading')}
                     disabled={isActionLoading || !isTradingEnabled}
                     className="inline-flex items-center px-3 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-900 focus:ring-red-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                     title={!isTradingEnabled ? "Trading is not active" : "Stop the trading bot"}
                 >
                    {isActionLoading ? <Loader2 className="animate-spin -ml-1 mr-2 h-5 w-5" /> : <StopCircle className="-ml-1 mr-2 h-5 w-5" />}
                     Stop Trading
                </button>

            </div>
        </div>
    );
};


// Helper component for rendering form fields consistently
const renderInputField = (label, name, value, onChange, props = {}) => {
    const { type = 'text', options = [], isLoading = false, disabled = false, addon = null, ...rest } = props;
    const id = `config-${name}`;
    const commonClasses = "block w-full px-3 py-1.5 border border-gray-600 bg-gray-700 text-gray-200 rounded-md shadow-sm focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm disabled:opacity-60 disabled:cursor-not-allowed";

    return (
        <div className="space-y-1">
            <label htmlFor={id} className="block text-xs font-medium text-gray-400">{label}{rest.required && <span className="text-red-400 ml-1">*</span>}</label>
            <div className="relative flex items-center">
                 {type === 'select' ? (
                    <select
                        id={id} name={name} value={value} onChange={onChange} disabled={disabled || isLoading}
                        className={`${commonClasses} appearance-none ${addon ? 'pr-10' : 'pr-8'}`} // Adjust padding if addon exists
                        {...rest}
                    >
                        {isLoading && <option value="" disabled>Loading {label}...</option>}
                        {!isLoading && options.length === 0 && <option value="" disabled>No options available</option>}
                         {/* Add a selectable placeholder if value is empty */}
                         {!isLoading && value === '' && <option value="" disabled>Select {label}</option>}
                        {!isLoading && options.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                    </select>
                 ) : (
                    <input
                        id={id} name={name} type={type} value={value} onChange={onChange} disabled={disabled}
                        className={`${commonClasses} ${addon ? 'pr-8' : ''}`} // Add padding if addon exists
                        {...rest} // Pass other props like min, max, step, placeholder, required
                    />
                 )}
                 {/* Optional Addon Button (e.g., refresh) */}
                 {addon && (
                     <div className="absolute inset-y-0 right-0 pr-1.5 flex items-center">
                         {addon}
                     </div>
                 )}
            </div>
        </div>
    );
};

export default ControlsComponent;
