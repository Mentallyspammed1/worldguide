import { jsx as _jsx } from "react/jsx-runtime";
import { Component } from "react";
class ErrorBoundary extends Component {
    constructor() {
        super(...arguments);
        this.state = {
            hasError: false
        };
    }
    static getDerivedStateFromError(_) {
        // Update state so the next render will show the fallback UI.
        return { hasError: true };
    }
    componentDidCatch(error, errorInfo) {
        console.error("Uncaught error:", error, errorInfo);
    }
    render() {
        if (this.state.hasError) {
            return _jsx("h1", { children: "Something went wrong." });
        }
        return this.props.children;
    }
}
export default ErrorBoundary;
