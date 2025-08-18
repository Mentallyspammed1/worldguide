import { jsxs as _jsxs, jsx as _jsx } from "react/jsx-runtime";
const Dashboard = ({ botStatus, config, tradeState, chartData }) => {
    return (_jsxs("div", { children: [_jsxs("h1", { children: ["Bot Status: ", botStatus] }), _jsx("h2", { children: "Config" }), _jsx("pre", { children: JSON.stringify(config, null, 2) }), _jsx("h2", { children: "Trade State" }), _jsx("pre", { children: JSON.stringify(tradeState, null, 2) }), _jsx("h2", { children: "Chart Data" }), _jsx("pre", { children: JSON.stringify(chartData, null, 2) })] }));
};
export default Dashboard;
