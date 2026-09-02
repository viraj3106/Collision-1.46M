# COLLISION Public Launch & Developer Onboarding Guide

This document outlines the developer onboarding flows, system positioning, and local developer verification steps.

## 1. Onboarding Funnel Workflow

```
Developer Landing Page
       ↓
Account Registration (website/src/App.tsx)
       ↓
Onboarding Checklist Dashboard (Checklist UI)
       ↓
Open Developer Portal (playground/app.py)
       ↓
Generate API Authorization Token
       ↓
Execute cURL / Python / JS API Completions Requests
```

## 2. Platform Positioning
- **轻量级 API / Lightweight API**: Emphasize that COLLISION is a compact 10M parameter causal completions base model designed for prototyping and research, rather than comparing it to frontier chatbot tools.
- **CPU Inference**: Clearly document that COLLISION is fully optimized for CPU execution.

## 3. Local Verification Tests
1. Verify database mappings:
   ```bash
   python -m unittest tests/test_production_flow.py
   ```
2. Verify security boundaries:
   ```bash
   python -m unittest tests/test_production_deployment.py
   ```
