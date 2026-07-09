# Flowpilot Server Backend

Flowpilot Server is a high-performance, asynchronous REST API built using **FastAPI** and **MongoDB (via Beanie ODM)**. It functions as the issue tracking and workflow sync engine for the Flowpilot management system, powering Kanban boards, secure user role permissions, and project tracking.

---

## 🛠 Tech Stack

*   **Web Framework:** [FastAPI](https://fastapi.tiangolo.com/) (Asynchronous Python REST framework)
*   **Database ODM:** [Beanie ODM](https://roman-right.github.io/beanie/) (Asynchronous Object Document Mapper for MongoDB built on Pydantic and Motor)
*   **Database Client:** [Motor](https://motor.readthedocs.io/) (Asynchronous driver for MongoDB)
*   **Data Validation & Serialization:** [Pydantic v2](https://docs.pydantic.dev/)
*   **Security & Auth:** [PyJWT](https://pyjwt.readthedocs.io/) (JSON Web Tokens) & [bcrypt](https://github.com/pyca/bcrypt/) (Secure hashing)
*   **Testing:** [Pytest](https://docs.pytest.org/) & [HTTPX](https://www.python-httpx.org/) (Async HTTP test client)

---

## 📁 Repository Structure

```text
Flowpilot-Server/
├── app/
│   ├── main.py                 # FastAPI application entrypoint & lifespans
│   ├── config.py               # Application configurations & env settings
│   ├── database.py             # MongoDB client initialization & Beanie registration
│   ├── auth/                   # JWT creation, extraction, and utility dependencies
│   │   ├── __init__.py
│   │   ├── dependencies.py     # OAuth2 password bearer flow & current user lookup
│   │   └── utils.py            # Password hashing (bcrypt) & JWT helpers
│   ├── models/                 # Beanie ODM models (MongoDB documents)
│   │   ├── __init__.py
│   │   ├── user.py             # User Document (ADMIN, MANAGER, DEVELOPER, CLIENT)
│   │   └── task.py             # Task/Issue Document (EPIC, TASK, SUBTASK, BUG)
│   ├── schemas/                # Pydantic schemas for request/response validation
│   └── routers/                # API router endpoints
│       ├── auth_router.py      # /api/v1/auth - SignUp, Login, JWT tokens
│       ├── user_router.py      # /api/v1/users - Profile info, assignee lookup
│       └── task_router.py      # /api/v1/tasks - CRUD operations & Kanban column movement
├── tests/                      # Automated API endpoint integration tests
│   ├── conftest.py             # Test setups, overrides, and DB isolation mocks
│   └── test_endpoints.py       # Integration tests for auth, users, and tasks
├── .env                        # Local & Cloud configuration settings
├── requirements.txt            # Python environment dependencies
└── README.md                   # This overview guide
```

---

## 🚀 Getting Started

### 1. Prerequisites
*   Python **3.11+** installed.
*   A running **MongoDB instance** (either locally or on the cloud via **MongoDB Atlas**).

### 2. Configure Environment Settings (`.env`)
Create a `.env` file in the root directory. You can use the following configuration:

```ini
PROJECT_NAME="Nexucon FlowPilot Manager"

# Cloud Atlas Connection or local: mongodb://localhost:27017
MONGODB_URL="mongodb+srv://Flowpilot01:Nexu2002@flowpilot01.iobzpqy.mongodb.net/"
DATABASE_NAME="flowpilot"

# JWT Auth security settings
JWT_SECRET="super-secret-key-change-in-production-123456789"
ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=1440
```

### 3. Initialize & Setup Virtual Environment
Run the following commands inside the repository root to create a virtual environment and install the required dependencies:

```bash
# Create the virtual environment
python3 -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Upgrade pip and install all required modules
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Running the API Server
Start the Uvicorn ASGI server with automatic reload for development:

```bash
uvicorn app.main:app --reload
```

Once running, the server is available at:
*   **API Base URL:** `http://127.0.0.1:8000/`
*   **Swagger API Docs:** `http://127.0.0.1:8000/docs`
*   **ReDoc API Docs:** `http://127.0.0.1:8000/redoc`

---

## 🔒 Roles & Access Control

Flowpilot uses role-based fields to categorize users:
*   **`ADMIN`**: Full permissions across the application.
*   **`MANAGER`**: Oversees project tasks, updates estimates, assigns tickets.
*   **`DEVELOPER`**: Updates task progress, logs hours, views assignees.
*   **`CLIENT`**: Views progress, reports issues, interacts with task status.

---

## 🔌 API Endpoints Summary

### 🔑 Authentication (`/api/v1/auth`)
*   `POST /signup` - Registers a new user. Expects user details and returns the profile without passwords.
*   `POST /login` - Accepts OAuth2 compatible form-data (`username`, `password`) and returns a secure JWT bearer token.

### 👤 User Profiles (`/api/v1/users`)
*   `GET /me` - Returns the logged-in user's profile details.
*   `GET /` - Lists all registered users (useful for populating dropdown lists in tasks).

### 📋 Task Manager (`/api/v1/tasks`)
*   `POST /` - Creates a new task or bug.
*   `GET /` - Queries all tasks associated with a given `project_id`.
*   `GET /{task_id}` - Fetches a specific task's details.
*   `PATCH /{task_id}` - Modifies fields such as description, assignees, or logged/estimated hours.
*   `PATCH /{task_id}/status` - Quick update of task columns through Kanban stages (`TODO`, `IN_PROGRESS`, `IN_REVIEW`, `DONE`).

---

## 🧪 Running Integration Tests

Integration and unit tests run against a mock database using `pytest`. Activate the virtual environment and execute:

```bash
pytest
```

The test runner will run endpoint validations for registration, login, profile checks, task CRUD, and full lifecycle Kanban movements.
