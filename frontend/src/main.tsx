import React from "react";
import ReactDOM from "react-dom/client";

import "./g3-base.css";
import "./perf.css";
import "./app.css";
import App from "./App";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
