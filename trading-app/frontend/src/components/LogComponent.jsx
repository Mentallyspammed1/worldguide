// src/components/LogComponent.jsx
import React, { useRef, useEffect } from 'react';
import { Terminal } from 'lucide-react';

const LogComponent = ({ logs }) => {
    const logEndRef = useRef(null); // Ref to the end of the log container
    const logContainerRef = useRef(null); // Ref to the scrollable container

    // Auto-scroll to the bottom when logs array updates, but only if user isn't scrolled up
    useEffect(() => {
        const container = logContainerRef.current;
        if (container) {
            // Check if user is scrolled near the bottom before auto-scrolling
            const isScrolledToBottom = container.scrollHeight - container.clientHeight <= container.scrollTop + 50; // 50px threshold
            if (isScrolledToBottom) {
                logEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
            }
        }
    }, [logs]); // Dependency: run effect when 'logs' prop changes

    // Function to determine Tailwind CSS class based on log level keywords
    const getLogColorClass = (logLine) => {
        const lowerCaseLog = logLine.toLowerCase();
        // More specific matching for levels at the start or within brackets
        if (lowerCaseLog.includes('[error]') || lowerCaseLog.includes('[fatal]')) return 'text-red-400';
        if (lowerCaseLog.includes(' error:') || lowerCaseLog.startsWith('error:')) return 'text-red-400';
        if (lowerCaseLog.includes('[warn]') || lowerCaseLog.includes(' warning') || lowerCaseLog.includes(' failed')) return 'text-yellow-400';
        if (lowerCaseLog.includes('[success]')) return 'text-green-400';
        if (lowerCaseLog.includes('[info]')) return 'text-blue-400';
        if (lowerCaseLog.includes('[debug]')) return 'text-gray-500';
        // Default color for lines without a recognized level
        return 'text-gray-300';
    };

    return (
        <div className="border border-gray-700 rounded-md h-64 md:h-80 bg-gray-900 shadow-inner flex flex-col">
             {/* Header */}
             <h3 className="text-base font-semibold text-gray-200 p-3 border-b border-gray-700 sticky top-0 bg-gray-900 z-10 flex items-center flex-shrink-0">
                 <Terminal className="h-5 w-5 mr-2 text-cyan-400 flex-shrink-0" />
                 Strategy & System Logs
             </h3>
             {/* Log Content Area */}
             <div ref={logContainerRef} className="overflow-y-auto flex-grow p-3 scroll-smooth">
                <pre className="text-xs font-mono whitespace-pre-wrap break-words leading-relaxed">
                    {(logs && logs.length > 0)
                        ? logs.map((log, index) => (
                            // Render each log line with appropriate color
                            <div key={index} className={getLogColorClass(log)}>
                                {log}
                            </div>
                          ))
                        : (
                            // Message when no logs are available
                            <span className="text-gray-500 italic flex items-center justify-center h-full">
                                Logs will appear here once trading starts or actions occur...
                            </span>
                        )
                    }
                    {/* Empty div at the end to act as a target for scrolling */}
                    <div ref={logEndRef} style={{ height: '1px' }} />
                </pre>
            </div>
        </div>
    );
};

export default React.memo(LogComponent); // Memoize as logs can update frequently
