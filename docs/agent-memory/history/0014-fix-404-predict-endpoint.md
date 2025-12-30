---
type: task
title: Troubleshooting 404 error on /predict endpoint
date: 2025-12-30
status: In Progress
tags:
  - Bug
  - FastAPI
  - NextJS
  - 404
project: solar-panel
---

# Task: Troubleshooting 404 error on /predict endpoint

## Objective
Investigate and fix the 404 error occurring when the frontend attempts to call `http://localhost:8000/predict`.

## Context
- **Error**: 404 Not Found.
- **Frontend**: `ImageUpload.tsx` calls `http://localhost:8000/predict`.
- **Backend (Edge)**: Recently implemented FastAPI service.

## Visual Flow (Mermaid)
```mermaid
sequenceDiagram
    participant User
    participant Frontend as Dashboard (Next.js)
    participant Backend as Edge API (FastAPI)
    User->>Frontend: Click "Analyse Image"
    Frontend->>Backend: POST /predict
    Backend-->>Frontend: 404 Not Found
    Note over Backend: Endpoint might be named /detect?
```

## Plan
- [x] Create bug report task.
- [ ] Verify endpoint name in `backend_edge/app/main.py`.
- [ ] Verify fetch URL in `frontend/src/components/ImageUpload.tsx`.
- [ ] Align naming conventions between frontend and edge backend.
- [ ] Test the fix with synthetic data.

## Execution Log
- [x] Task created.
