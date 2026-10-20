
require("dotenv").config();

const express = require("express");
const axios = require("axios");
const path = require("path");
const cors = require("cors");

const app = express();
const PORT = process.env.PORT || 3000;
const AI_ENGINE_URL =
    process.env.AI_ENGINE_URL || "http://127.0.0.1:8000";

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

// initial health check api
app.get("/health", (req, res) => {
    res.json({
        status: "ok",
        service: "backend"
    });
});

app.get("/api/health-ai", async (req, res) => {
    try {
        const response = await axios.get(
            `${AI_ENGINE_URL}/health`,
            { timeout: 3000 }
        );

        res.json({
            status: "ok",
            aiEngine: response.data
        });
    } catch (error) {
        res.status(503).json({
            status: "error",
            message: "AI engine is unavailable"
        });
    }
});

const READY_QUESTIONS = [
    { id: "hotels-udaipur", label: "Hotels in Udaipur", prompt: "What hotels are available in Udaipur on JusBid?" },
    { id: "create-account", label: "Create Account", prompt: "How do I create an account or sign up on JusBid?" },
    { id: "forgot-password", label: "Forgot Password", prompt: "I forgot my password. What should I do?" },
    { id: "post-requirement", label: "Post a Requirement", prompt: "How can I post a requirement?" },
    { id: "submit-bid", label: "Submit a Bid", prompt: "How do I submit a bid?" },
    { id: "track-bid", label: "Track Bid Status", prompt: "Where can I check my bid status?" },
    { id: "payment-failed", label: "Payment Issue & Refund", prompt: "My payment failed but money was deducted. What should I do?" },
    { id: "platform-fees", label: "Platform Fees & Charges", prompt: "What are JusBid's fees?" },
    { id: "cancel-bid", label: "Cancel a Bid", prompt: "Can you cancel my bid for me?" },
    { id: "security-otp", label: "Security & OTP Issue", prompt: "I received a suspicious message asking for my OTP." },
    { id: "human-support", label: "Talk to Human Support", prompt: "Can I speak to a human support representative?" }
];

// quick ready questions for chat widget suggestion pills
app.get("/api/quick-questions", (req, res) => {
    res.json({
        company: "JusBid",
        greeting: "Welcome to JusBid! I'm your AI assistant. What can I help you with today?",
        questions: READY_QUESTIONS
    });
});

// chat api
app.post("/api/chat", async (req, res) => {
    try {
        const { message, history = [] } = req.body || {};

        if (typeof message !== "string" || !message.trim()) {
            return res.status(400).json({
                error: "A non-empty message is required"
            });
        }

        const response = await axios.post(
            `${AI_ENGINE_URL}/chat`,
            {
                message: message.trim(),
                history
            },
            { timeout: 45000 }
        );

        res.json(response.data);
    } catch (error) {
        res.status(503).json({
            error: "The support service is temporarily unavailable"
        });
    }
});


app.listen(PORT, "127.0.0.1", () => {
    console.log(`Backend running at http://127.0.0.1:${PORT}`);
});
