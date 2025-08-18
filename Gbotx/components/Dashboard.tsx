import React from 'react';
import { BotConfig, BotStatus, ChartDataPoint, TradeState } from '../types';


interface DashboardProps {
    botStatus: BotStatus;
    config: BotConfig;
    tradeState: TradeState;
    chartData: ChartDataPoint[];
}

const Dashboard: React.FC<DashboardProps> = ({ botStatus, config, tradeState, chartData }) => {
    return (
        <div>
            <h1>Bot Status: {botStatus}</h1>
            <h2>Config</h2>
            <pre>{JSON.stringify(config, null, 2)}</pre>
            <h2>Trade State</h2>
            <pre>{JSON.stringify(tradeState, null, 2)}</pre>
            <h2>Chart Data</h2>
            <pre>{JSON.stringify(chartData, null, 2)}</pre>
        </div>
    );
};

export default Dashboard;