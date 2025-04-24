import React from 'react';
import ReactDOM from 'react-dom/client';
// If you create src/index.css for global styles, uncomment below:
// import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';

const rootElement = document.getElementById('root');
const root = ReactDOM.createRoot(rootElement);

root.render(
  // StrictMode removed for now, as it can sometimes cause double renders/effects
  // during development which might complicate debugging async operations or intervals.
  // Re-enable if needed for identifying potential problems.
  // <React.StrictMode>
    <App />
  // </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
