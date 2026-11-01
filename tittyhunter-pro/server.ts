import "dotenv/config";
import helmet from "helmet";
import rateLimit from "express-rate-limit";
import cors from "cors";
import express from "express";
const app = express();

// Security headers
app.use(
  helmet({
    hidePoweredBy: true,
    contentSecurityPolicy: {
      directives: {
        defaultSrc: ["'self'"],
        scriptSrc: ["'self'", "https://cdn.jsdelivr.net"],
        imgSrc: ["'self'", "data:"],
        connectSrc: ["'self'", "ws://localhost:3000"],
      },
    },
    referrerPolicy: { policy: "no-referrer" },
  })
);

// CORS – allow the PWA origin
app.use(
  cors({
    origin: process.env.ALLOWED_ORIGIN || "*",
    credentials: true,
  })
);

// Rate limiting – stricter for public API endpoints
const apiLimiter = rateLimit({
  windowMs: 60_000, // 1 minute
  max: 30, // limit each IP to 30 requests per window
  message: "Too many requests, please try again later.",
});
app.use("/api", apiLimiter);

// Simple health‑check – returns uptime, port, and environment
app.get("/health", (_req, res) => {
  res.json({
    status: "ok",
    uptime: process.uptime(),
    port: process.env.PORT || 3000,
    node: process.version,
    env: process.env.NODE_ENV,
  });
});

// ... other routes would be added here

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`🔞 tittyhunter-pro listening on http://localhost:${PORT}`);
});

export default app;