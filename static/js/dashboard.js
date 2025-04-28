// Dashboard.js - Client-side JavaScript for the Trading Bot Dashboard

// DOM elements
let performanceChart = null;

// Initialize dashboard components when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    setupEventListeners();
    refreshData();
});

// Set up event listeners
function setupEventListeners() {
    // Refresh button
    const refreshBtn = document.getElementById('refreshBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', refreshData);
    }
    
    // Save configuration button
    const saveConfigBtn = document.getElementById('saveConfigBtn');
    if (saveConfigBtn) {
        saveConfigBtn.addEventListener('click', saveConfiguration);
    }
}

// Initialize charts
function initializeCharts() {
    // Performance chart
    const ctx = document.getElementById('performanceChart');
    if (ctx) {
        performanceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Cumulative PnL',
                    data: [],
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderWidth: 2,
                    tension: 0.1,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: {
                            display: false
                        }
                    },
                    y: {
                        beginAtZero: false
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                }
            }
        });
    }
}

// Refresh dashboard data
function refreshData() {
    // Get performance data
    fetch('/api/performance')
        .then(response => response.json())
        .then(data => {
            updatePerformanceStats(data);
        })
        .catch(error => {
            console.error('Error fetching performance data:', error);
        });
        
    // Get state data to update charts
    fetch('/api/state')
        .then(response => response.json())
        .then(data => {
            updateCharts(data);
        })
        .catch(error => {
            console.error('Error fetching state data:', error);
        });
}

// Update performance statistics
function updatePerformanceStats(data) {
    // If we had elements with IDs matching these stats, we'd update them
    console.log('Performance data:', data);
    
    // For now, just log the data to console
    // In a full implementation, we'd update DOM elements with these values
}

// Update charts with new data
function updateCharts(data) {
    // If no trade data, don't update charts
    if (!data.trades || data.trades.length === 0) {
        return;
    }
    
    // Sort trades by timestamp
    const sortedTrades = data.trades.sort((a, b) => a.timestamp - b.timestamp);
    
    // Update performance chart
    if (performanceChart) {
        // Calculate cumulative PnL
        let cumulativePnL = 0;
        const chartData = sortedTrades.map(trade => {
            cumulativePnL += trade.pnl || 0;
            return cumulativePnL;
        });
        
        // Format dates for labels
        const labels = sortedTrades.map(trade => {
            const date = new Date(trade.timestamp);
            return date.toLocaleDateString();
        });
        
        // Update chart data
        performanceChart.data.labels = labels;
        performanceChart.data.datasets[0].data = chartData;
        performanceChart.update();
    }
}

// Save configuration changes
function saveConfiguration() {
    // Get values from form fields
    const config = {
        symbol: document.getElementById('symbol').value,
        exchange: document.getElementById('exchange').value,
        timeframe: document.getElementById('timeframe').value,
        risk_management: {
            position_size_pct: parseFloat(document.getElementById('position_size').value),
            max_risk_per_trade_pct: parseFloat(document.getElementById('max_risk').value),
            use_atr_position_sizing: document.getElementById('use_atr_sizing').checked
        }
    };
    
    // Send to API
    fetch('/api/config', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(config)
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            alert('Configuration saved successfully');
        } else {
            alert('Error saving configuration: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error saving configuration:', error);
        alert('Error saving configuration');
    });
}

// Utility function to format currency values
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(value);
}

// Utility function to format percentage values
function formatPercentage(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value / 100);
}
