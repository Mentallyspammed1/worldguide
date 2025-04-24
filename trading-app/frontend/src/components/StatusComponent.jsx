// src/components/StatusComponent.jsx
import React from 'react';
import { DollarSign, TrendingUp, TrendingDown, AlertCircle, Activity, AlertOctagon, Info, ZapOff } from 'lucide-react';

// Helper to format numbers, handling null/undefined and precision
const formatNumber = (num, options = {}) => {
    const { digits = 2, currency = false, sign = false, defaultValue = 'N/A' } = options;
    // Allow overriding defaultValue
    if (num === null || num === undefined || num === '' || typeof num !== 'number' || isNaN(num)) {
        return defaultValue;
    }

    const formatterOptions = {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
    };
    if (currency) {
        formatterOptions.style = 'currency';
        // Assuming USDT ~ USD, adjust currency code if needed
        formatterOptions.currency = 'USD';
        // Remove currency symbol if sign is also requested to avoid clutter like "+$10.00"
        if (sign) formatterOptions.currencyDisplay = 'code'; // Use 'USD' instead of '$'
    }

    let formatted = num.toLocaleString(undefined, formatterOptions);

    // Add sign prefix if requested and number is positive
    if (sign && num > 0) {
        formatted = `+${formatted}`;
    }
    // Negative sign is usually handled by toLocaleString

    return formatted;
};

// Helper to guess precision based on symbol (very basic)
// In a real app, fetch precise market details from backend if needed for display
const guessPrecision = (symbol, type) => {
    if (!symbol) return 2;
    try {
        const base = symbol.split('/')[0];
        if (type === 'price') {
            if (['BTC', 'ETH'].includes(base)) return 2;
            if (['XRP', 'DOGE', 'SHIB', 'PEPE'].includes(base)) return 6; // Higher precision for smaller value coins
            if (['SOL', 'ADA', 'DOT', 'LINK'].includes(base)) return 4;
            return 3; // Default guess
        }
        if (type === 'amount') {
             if (['BTC', 'ETH'].includes(base)) return 5; // Higher precision for amount
             if (['SOL', 'ADA', 'XRP', 'LINK'].includes(base)) return 2;
             if (['DOGE', 'SHIB', 'PEPE'].includes(base)) return 0; // Often whole numbers for amount
             return 3; // Default guess
        }
    } catch (e) { /* Ignore errors parsing symbol */ }
    return 2; // Fallback precision
};


const StatusComponent = ({ statusData }) => {
    // Handle loading state or missing data
    if (!statusData) {
        return (
            <div className="p-4 border border-gray-700 rounded-md text-center text-gray-500 bg-gray-800 shadow-lg h-full flex items-center justify-center min-h-[200px]">
                <Activity className="animate-pulse h-6 w-6 mr-2" /> Awaiting Status Update...
            </div>
        );
    }

    const { balance, position, config, error: statusError, lastStrategyRun, isTradingEnabled } = statusData;
    const symbol = config?.symbol || 'N/A';

    // Determine position side and apply consistent coloring
    const positionSide = position?.side?.toLowerCase(); // Ensure lowercase ('long' or 'sell')
    const isLong = positionSide === 'long';
    const isShort = positionSide === 'sell' || positionSide === 'short'; // Handle both 'sell' and 'short'
    const positionColor = isLong ? 'text-green-400' : isShort ? 'text-red-400' : 'text-gray-400';
    const positionBgColor = isLong ? 'bg-green-900/50 border-green-700/50' : isShort ? 'bg-red-900/50 border-red-700/50' : 'bg-gray-700/50 border-gray-600/50';


    // Determine PNL color and icon
    const pnl = typeof position?.unrealizedPnl === 'string' ? parseFloat(position.unrealizedPnl) : position?.unrealizedPnl;
    const pnlColor = typeof pnl === 'number' ? (pnl > 0 ? 'text-green-400' : pnl < 0 ? 'text-red-400' : 'text-gray-400') : 'text-gray-400';
    const PnlIcon = typeof pnl === 'number' ? (pnl > 0 ? TrendingUp : pnl < 0 ? TrendingDown : null) : null;

    // Get market precision (using helper for now)
    const pricePrecision = guessPrecision(symbol, 'price');
    const amountPrecision = guessPrecision(symbol, 'amount');

    // Format last run time
    const lastRunTime = lastStrategyRun ? new Date(lastStrategyRun).toLocaleString() : 'Never';


    return (
        <div className="p-4 border border-gray-700 rounded-md space-y-3 bg-gray-800 shadow-lg h-full text-sm min-h-[200px] flex flex-col">
            <h3 className="text-lg font-semibold text-gray-200 border-b border-gray-600 pb-2 mb-3 flex-shrink-0">Account & Position Status</h3>

            {/* Display Status Fetch Errors */}
            {statusError && (
                 <div className={`border px-3 py-2 rounded text-sm flex items-center mb-3 bg-yellow-900 border-yellow-700 text-yellow-200 flex-shrink-0`}>
                    <AlertOctagon className="h-4 w-4 mr-2 flex-shrink-0"/> Status Warning: {statusError}
                </div>
            )}

            {/* Balance Section */}
            <div className="flex items-center justify-between space-x-2 text-gray-300 bg-gray-700 px-3 py-2 rounded flex-shrink-0">
                <div className="flex items-center">
                   <DollarSign className="h-5 w-5 mr-2 text-blue-400 flex-shrink-0" />
                   <span>Balance (USDT):</span>
                </div>
                <span className="font-mono font-semibold text-lg text-gray-100">{formatNumber(balance, { digits: 2, defaultValue: '...' })}</span>
            </div>

             {/* Last Strategy Run Time */}
             <div className="text-xs text-gray-500 text-right flex-shrink-0">
                 Last Strategy Check: {lastRunTime}
             </div>


            {/* Position Details Section */}
            <div className="flex-grow"> {/* Allow this section to grow */}
                <h4 className="font-medium text-gray-400 mb-2">Position ({symbol})</h4>
                {position ? (
                    <div className={`space-y-1.5 text-xs p-3 rounded border ${positionBgColor}`}>
                        {/* Row Helper */}
                        {renderStatusRow("Side:", <span className={`font-semibold ${positionColor} uppercase`}>{positionSide}</span>)}
                        {renderStatusRow(`Size (${marketBase(symbol)}):`, formatNumber(position.contractsFormatted ?? position.contracts, { digits: amountPrecision }))}
                        {renderStatusRow("Entry Price:", formatNumber(position.entryPriceFormatted ?? position.entryPrice, { digits: pricePrecision }))}
                        {renderStatusRow("Mark Price:", formatNumber(position.markPriceFormatted ?? position.markPrice, { digits: pricePrecision }))}
                        {renderStatusRow("Unrealized PNL:",
                            <span className={`font-mono font-semibold ${pnlColor} flex items-center`}>
                                {PnlIcon && <PnlIcon className="h-3.5 w-3.5 mr-1"/>}
                                {formatNumber(pnl, { digits: 2, sign: true })}
                            </span>
                        )}
                        {renderStatusRow("Leverage:", position.leverage ? `${formatNumber(position.leverage, {digits: 0})}x` : 'N/A')}
                        {renderStatusRow("Liq. Price:",
                            <span className="font-mono text-orange-400 font-semibold">
                                {formatNumber(position.liquidationPriceFormatted ?? position.liquidationPrice, { digits: pricePrecision })}
                            </span>
                        )}
                        {/* Add Margin if available */}
                         {position.initialMargin && renderStatusRow("Margin (Initial):", formatNumber(position.initialMargin, { digits: 2 }))}
                         {position.maintMargin && renderStatusRow("Margin (Maint):", formatNumber(position.maintMargin, { digits: 2 }))}
                    </div>
                ) : (
                     // Show different message based on trading status
                     isTradingEnabled ? (
                        <div className="text-gray-500 pl-3 text-sm italic border-l-2 border-gray-600 py-1 flex items-center">
                            <Activity size={14} className="mr-2"/> Waiting for entry signal...
                        </div>
                     ) : (
                        <div className="text-gray-600 pl-3 text-sm italic border-l-2 border-gray-700 py-1 flex items-center">
                            <ZapOff size={14} className="mr-2"/> Trading stopped. No position open.
                        </div>
                     )
                )}
            </div>
        </div>
    );
};

// Helper component for consistent status row rendering
const renderStatusRow = (label, value) => (
    <p className="flex justify-between items-center gap-2">
        <span className="text-gray-400 mr-2 flex-shrink-0">{label}</span>
        <span className="font-mono text-gray-200 text-right break-words">{value}</span>
    </p>
);

// Helper to get base currency from symbol string
const marketBase = (symbol) => {
    if (!symbol || typeof symbol !== 'string') return 'Units';
     try {
        // Handles formats like BTC/USDT or BTC/USDT:USDT
        return symbol.split(':')[0].split('/')[0];
    } catch (e) { return 'Units'; }
};


export default React.memo(StatusComponent);
