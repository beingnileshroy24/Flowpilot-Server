# Nexucon FlowPilot Manager Backend

This repository contains the foundation and issue tracking engine for **Nexucon FlowPilot Manager**, built as a high-performance, asynchronous REST API using Python, FastAPI, and MongoDB (via Beanie ODM).

It serves as the backend supporting issue registration, user permissions (Admin, Manager, Developer, Client), and state syncing for visual Kanban boards.

---

## Technical Stack
- **Framework**: FastAPI (Asynchronous Python)
- **Database ODM**: Beanie (Object Document Mapper)
- **Database Driver**: Motor (Asynchronous MongoDB Client)
- **Data Validation**: Pydantic v2
- **Authentication**: JSON Web Tokens (PyJWT) and bcrypt (direct verification)

---

## Project Structure
```
Flowpilot-Server/
├── app/
│   ├── main.py                 # FastAPI app initialization & lifespans
│   ├── config.py               # App settings & Env variables
│   ├── database.py             # MongoDB connection logic
│   ├── auth/                   # Security, bcrypt and JWT dependencies
│   ├── models/                 # Beanie ODM schemas (User, Task)
│   ├── schemas/                # Request/Response shapes (Pydantic)
│   └── routers/                # Endpoint routers (Auth, Users, Tasks)
├── tests/                      # Automated test suite
├── .env                        # Local credentials configuration
├── requirements.txt            # Dependency definitions
└── README.md                   # This instructions guide
```

---

## Local Setup

### 1. Prerequisites
- **Python**: 3.11+
- **MongoDB**: Make sure a MongoDB instance is running locally at `mongodb://localhost:27017` or prepare a remote connection URI.

### 2. Configure Environment Settings
Create your `.env` file in the root directory (based on the default configuration):
```ini
PROJECT_NAME="Nexucon FlowPilot Manager"
MONGODB_URL="mongodb://localhost:27017"
DATABASE_NAME="flowpilot"
JWT_SECRET="super-secret-key-change-in-production-123456789"
ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=1440
```

### 3. Initialize Virtual Environment
Create a Python virtual environment and install the required dependencies:
```bash
# Initialize venv
python3 -m venv .venv

# Activate venv (macOS/Linux)
source .venv/bin/activate

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Running the Server
Start the development server using Uvicorn with auto-reload enabled:
```bash
uvicorn app.main:app --reload
```
Once started, you can access:
- **API Base**: `http://127.0.0.1:8000/`
- **Interactive Documentation (Swagger UI)**: `http://127.0.0.1:8000/docs`
- **Alternative Documentation (ReDoc)**: `http://127.0.0.1:8000/redoc`

---

## API Endpoints Summary

### 🔑 Authentication (`/api/v1/auth`)
* `POST /api/v1/auth/signup` - Register a new User with role permissions (`ADMIN`, `MANAGER`, `DEVELOPER`, `CLIENT`).
* `POST /api/v1/auth/login` - Submit credentials via form-data (OAuth2 flow compatibility) to obtain a secure JWT token.

### 👤 Users (`/api/v1/users`)
* `GET /api/v1/users/me` - Retrieve the current authenticated user's profile info.
* `GET /api/v1/users/` - List all registered users (used for selecting task assignees).

### 📋 Tasks (`/api/v1/tasks`)
* `POST /api/v1/tasks/` - Create a new issue/ticket (Task, Subtask, Epic, Bug).
* `GET /api/v1/tasks/` - List tasks/bugs belonging to a specific `project_id`.
* `GET /api/v1/tasks/{task_id}` - Retrieve detailed task information.
* `PATCH /api/v1/tasks/{task_id}` - Perform general updates (e.g. description, logged hours, assignees).
* `PATCH /api/v1/tasks/{task_id}/status` - Move a task through Kanban columns (`TODO`, `IN_PROGRESS`, `IN_REVIEW`, `DONE`).

---

## Running Automated Tests
Run tests inside your virtual environment using Pytest:
```bash
pytest
```
