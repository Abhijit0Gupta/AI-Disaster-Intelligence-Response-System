# API Contract

## 1. Purpose

This document defines the communication structure between the frontend and backend of the AI Disaster Intelligence & Response System.

All team members must follow the field names, data types, endpoints, and response formats defined in this document.

---

## 2. Base API

During local development:

http://127.0.0.1:8000

---

## 3. Health Check Endpoint

### Endpoint

GET /

### Purpose

Checks whether the backend server is running.

### Expected Response

```json
{
  "status": "success",
  "message": "AI Disaster Intelligence & Response System is running"
}