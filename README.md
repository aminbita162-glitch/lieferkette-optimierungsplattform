# AI-Powered Supply Chain Optimization Platform 2

An end-to-end logistics decision-support platform for demand forecasting, inventory planning, warehouse allocation, and route optimization.

---

## 🔗 Live Access

| Resource | URL |
|---|---|
| 🌐 Live API | [lieferkette-optimierungsplattform.onrender.com](https://lieferkette-optimierungsplattform.onrender.com) |
| 🎬 Project Video Demo | [Watch on YouTube](https://youtu.be/hUALYUOO_sI) |
| 📊 Operations Dashboard | [Open Dashboard](https://lieferkette-optimierungsplattform.onrender.com/dashboard/) |
| 🧭 Swagger UI | [Open API Explorer](https://lieferkette-optimierungsplattform.onrender.com/docs) |
| 📘 ReDoc | [Open API Reference](https://lieferkette-optimierungsplattform.onrender.com/redoc) |
| 🧩 OpenAPI Schema | [View `openapi.json`](https://lieferkette-optimierungsplattform.onrender.com/openapi.json) |
| ❤️ Health Endpoint | [Check Service Health](https://lieferkette-optimierungsplattform.onrender.com/health) |

## 🎯 Executive Overview

The **AI-Powered Supply Chain Optimization Platform** is a working logistics application that connects operational data management with forecasting and optimization services through a documented FastAPI backend and an interactive browser-based dashboard.

The current system supports:

- 🔐 Account registration, password hashing, JWT authentication, and authenticated profiles
- 🏭 User-scoped warehouse creation, retrieval, listing, and deletion
- 📦 User-scoped order creation, retrieval, listing, and deletion
- 🧠 Automatic warehouse selection using proximity-based allocation
- 📈 Short-horizon demand forecasting using a moving-average baseline with stochastic variation
- 🧮 Inventory recommendations based on service level, lead time, demand uncertainty, and cost inputs
- 🗺️ Multi-stop route sequencing using a nearest-neighbor heuristic
- 🧪 Synthetic demand and calendar-data generation
- 📊 A responsive dashboard with KPIs, tables, charts, an interactive map, and module-status reporting
- 📚 OpenAPI 3.1 documentation through Swagger UI and ReDoc
- ✅ Automated Python testing on pushes and pull requests to `main`

This repository represents an actively evolving engineering platform. It is not presented as a completed enterprise product: implemented behavior, known technical boundaries, and planned capabilities are documented separately.

## 🧭 Problem and Engineering Objective

Supply-chain teams frequently operate across disconnected workflows: demand estimation, stock planning, warehouse assignment, order processing, and route planning. Fragmentation makes decisions slower, harder to reproduce, and more difficult to audit.

This project consolidates those workflows behind one API and one operational interface:

1. 📥 Capture warehouse, order, demand, location, and cost inputs.
2. 🧠 Apply forecasting, allocation, inventory, and routing logic.
3. 📤 Return structured recommendations through typed API contracts.
4. 📊 Expose results through a live dashboard and interactive API documentation.
5. 🔄 Establish a foundation for progressively more reliable, explainable, and autonomous supply-chain intelligence.

## 🏗️ Current System Architecture

### System context

```mermaid
flowchart TB
    USER["Authenticated Platform User"]
    CLIENT["External API Consumer"]

    subgraph PLATFORM["AI Logistics Platform"]
        WEB["Responsive Operations Dashboard"]
        DOCS["Swagger UI · ReDoc · OpenAPI"]
        CORE["FastAPI Application"]
        DECISION["Logistics Decision Services"]
        DATA["PostgreSQL Persistence"]
    end

    MAP["OpenStreetMap Tile Service"]
    DELIVERY["GitHub Actions · Render Deployment"]

    USER --> WEB
    CLIENT --> DOCS
    CLIENT --> CORE
    WEB --> CORE
    DOCS --> CORE
    CORE --> DECISION
    CORE --> DATA
    WEB --> MAP
    DELIVERY --> CORE
```

The platform exposes the same application capabilities through an operational browser interface and a typed HTTP API. FastAPI coordinates identity, domain routing, decision services, and persistence; the dashboard consumes those APIs and adds interactive operational visualization.

### Layered application architecture

```mermaid
flowchart TB
    subgraph EXPERIENCE["1 · Experience and Integration Layer"]
        DASH["Dashboard<br/>KPI · Tables · Forms"]
        GEO["Leaflet Map<br/>Markers · Allocation Lines · Routes"]
        CHART["Chart.js<br/>Capacity vs Demand"]
        CONTRACT["OpenAPI 3.1<br/>Swagger · ReDoc"]
        CONSUMER["HTTP API Clients"]
    end

    subgraph EDGE["2 · Application Edge"]
        STATIC["Static Dashboard Mount<br/>/dashboard"]
        FASTAPI["FastAPI Application<br/>v0.8.1"]
        CORS["CORS Middleware"]
        SCHEMA["Pydantic Contracts<br/>Validation · Serialization"]
    end

    subgraph ACCESS["3 · Identity and Access"]
        REGISTER["Registration"]
        LOGIN["OAuth2 Password Flow"]
        TOKEN["JWT Issue and Validation"]
        OWNER["User-Scoped Data Access"]
    end

    subgraph DOMAIN["4 · Domain API Layer"]
        AI["AI Logistics Router<br/>Forecast · Allocation · Plan"]
        OPT["Optimization Router<br/>Inventory Policy"]
        SIM["Simulation Router<br/>Synthetic Demand"]
        WH["Warehouse Router<br/>Create · Read · Delete"]
        ORD["Order Router<br/>Create · Read · Delete"]
        ROUTE["Route Router<br/>Group · Sequence · Summarize"]
    end

    subgraph INTELLIGENCE["5 · Decision and Simulation Layer"]
        FORECAST["Demand Forecast<br/>Moving-Average Baseline"]
        INVENTORY["Inventory Recommendation<br/>Safety Stock · Reorder Point"]
        ALLOCATION["Warehouse Allocation<br/>Nearest Coordinate"]
        ROUTING["Route Optimization<br/>Nearest-Neighbor Heuristic"]
        GENERATOR["Synthetic Data Generation<br/>Gaussian · Negative Binomial"]
    end

    subgraph PERSISTENCE["6 · Persistence Layer"]
        SESSION["SQLAlchemy Engine<br/>Session Lifecycle"]
        MODELS["ORM Models<br/>User · Warehouse · Order"]
        POSTGRES[("PostgreSQL")]
    end

    DASH --> STATIC
    GEO --> STATIC
    CHART --> STATIC
    CONTRACT --> FASTAPI
    CONSUMER --> FASTAPI
    STATIC --> FASTAPI
    CORS --> FASTAPI
    FASTAPI --> SCHEMA
    FASTAPI --> REGISTER
    FASTAPI --> LOGIN
    LOGIN --> TOKEN
    TOKEN --> OWNER
    FASTAPI --> AI
    FASTAPI --> OPT
    FASTAPI --> SIM
    OWNER --> WH
    OWNER --> ORD
    OWNER --> ROUTE
    AI --> FORECAST
    AI --> ALLOCATION
    AI --> ROUTING
    OPT --> INVENTORY
    SIM --> GENERATOR
    ORD --> ALLOCATION
    ROUTE --> ROUTING
    REGISTER --> SESSION
    OWNER --> SESSION
    WH --> SESSION
    ORD --> SESSION
    SESSION --> MODELS
    MODELS --> POSTGRES
```

### Runtime responsibility map

| Layer | Current responsibility |
|---|---|
| 🖥️ Presentation | Single-page responsive dashboard, KPI cards, operational tables, Chart.js visualization, Leaflet map, and API-driven forms |
| 🚪 API | FastAPI application, Pydantic request/response validation, router composition, OpenAPI generation, and CORS middleware |
| 🔐 Identity | Registration, bcrypt password hashing, OAuth2 password flow, JWT issuance, and authenticated profile lookup |
| 🧠 Decision services | Demand forecast, safety-stock and reorder policy, warehouse allocation, and route sequencing |
| 🗃️ Persistence | SQLAlchemy models and sessions backed by PostgreSQL |
| 🧪 Simulation | API-level daily-demand simulation plus an offline synthetic demand generator with seasonality, promotion, price, and dispersion controls |
| ✅ Delivery assurance | GitHub Actions workflow using Python 3.11 and pytest |

### Authenticated transaction path

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Dashboard
    participant API as FastAPI
    participant Auth as JWT Security
    participant Router as Domain Router
    participant Service as Decision Service
    participant DB as PostgreSQL

    User->>Dashboard: Submit credentials or operation
    Dashboard->>API: HTTPS request
    API->>Auth: Validate credentials or bearer token
    Auth->>DB: Resolve user identity
    DB-->>Auth: User record
    Auth-->>API: Authenticated user context
    API->>Router: Validate and dispatch request

    alt Data operation
        Router->>DB: Execute owner-scoped query
        DB-->>Router: Persisted domain records
    else Decision operation
        Router->>Service: Execute forecast or optimization
        Service-->>Router: Structured recommendation
    end

    Router-->>API: Typed response
    API-->>Dashboard: JSON result
    Dashboard-->>User: Tables, KPIs, chart, or map
```

### Delivery topology

```mermaid
flowchart TB
    DEV["Repository Change"]
    MAIN["GitHub Main Branch"]

    subgraph CI["Verification Path"]
        ACTION["GitHub Actions"]
        ENV["Python 3.11 Environment"]
        TEST["pytest Calendar Test Suite"]
    end

    subgraph RUNTIME["Deployment Path"]
        BUILD["Dependency Build<br/>anforderungen.txt"]
        START["Uvicorn Process<br/>FastAPI Application"]
        LIVE["Public HTTPS Service<br/>Frankfurt Region"]
    end

    DEV --> MAIN
    MAIN --> ACTION
    ACTION --> ENV
    ENV --> TEST
    MAIN --> BUILD
    BUILD --> START
    START --> LIVE
```

The diagrams above document the current implementation and deployment path. They do not imply distributed microservices, asynchronous processing, model serving infrastructure, a vector database, or autonomous agents; those capabilities remain roadmap items.

## 📖 Executable and Living Documentation Strategy

The repository is designed to evolve from a conventional README into a version-controlled, executable documentation system. The following documentation capabilities define the intended engineering direction and must be treated as planned work unless the corresponding artifacts exist in the repository.

```mermaid
flowchart LR
    CODE["Application and Model Code"]
    TESTS["Tests and BDD Scenarios"]
    CONTRACTS["OpenAPI · AsyncAPI · Schemas"]
    NOTEBOOKS["Quarto · Jupyter Book"]
    VISUALS["Maps · Metrics · Reports"]
    DOCS["Versioned Living Documentation"]

    CODE --> TESTS
    CODE --> CONTRACTS
    CODE --> NOTEBOOKS
    TESTS --> DOCS
    CONTRACTS --> DOCS
    NOTEBOOKS --> VISUALS
    VISUALS --> DOCS
```

### Documentation-as-code target

| Engineering concern | Planned artifact | Purpose |
|---|---|---|
| Executable analysis | `docs/` with Quarto or Jupyter Book | Run forecasting, inventory, allocation, and routing examples alongside explanations and real outputs |
| Architecture as code | Mermaid diagrams and C4 views | Keep system context, containers, components, decision flows, and deployment topology reviewable with code changes |
| Spatial-temporal logistics | Folium or Deck.gl HTML reports | Visualize warehouses, optimized routes, fleet positions, demand evolution, and scenario comparisons |
| REST contracts | Versioned OpenAPI specification | Provide a machine-readable source of truth for synchronous interfaces |
| Event contracts | `asyncapi.yaml` | Define future logistics, IoT, GPS, order, and optimization events |
| Data governance | `schemas/` using JSON Schema or Protobuf | Version logistics entities, units, constraints, compatibility rules, and ownership metadata |
| Behavioral specifications | `features/*.feature` | Describe expected logistics behavior using executable Gherkin scenarios |
| Security | `SECURITY.md` | Define vulnerability reporting, secret handling, authentication controls, and location-data protection |
| Compliance | `COMPLIANCE.md` | Record geographic-data privacy, retention, consent, access, deletion, and audit requirements |
| Product delivery | `ROADMAP.md` | Track engineering outcomes, ownership, acceptance criteria, dependencies, and delivery status |
| UX system | `docs/ux/` | Document dashboard interaction patterns, map behavior, accessibility, error states, and responsive design |
| Observability | `monitoring/` | Version Prometheus rules, Grafana dashboards, SLOs, alerts, and operational runbooks |
| MLOps lineage | MLflow-compatible tracking | Record datasets, parameters, metrics, artifacts, models, and decision provenance |
| Runtime orchestration | `compose.yaml` | Start the API, database, observability, and supporting services through a reproducible command |
| AI-assisted context | Controlled documentation-update workflow | Analyze repository changes and propose bounded documentation updates for human review |

### Documentation quality rules

- Generated content must be reproducible, reviewable, and clearly marked.
- AI-generated documentation changes must be submitted as diffs and approved by a human.
- Generated sections must never silently overwrite manually maintained architectural decisions.
- Executable examples must use deterministic seeds where reproducibility is required.
- Documentation builds must fail when referenced code, schemas, contracts, examples, or links become invalid.
- Current capabilities, verified behavior, experiments, and roadmap items must remain explicitly separated.
- Architecture diagrams must describe the implementation present in the corresponding revision.
- Public examples must use synthetic data and must not expose secrets, credentials, personal data, or real fleet locations.

### Contract-first integration model

```mermaid
flowchart TB
    CLIENT["Dashboard and API Clients"]
    OPENAPI["OpenAPI REST Contract"]
    API["FastAPI Services"]
    ASYNCAPI["AsyncAPI Event Contract"]
    EVENTS["Future Logistics and IoT Events"]
    SCHEMAS["JSON Schema or Protobuf"]
    GOVERNANCE["Validation · Compatibility · Audit"]

    CLIENT --> OPENAPI
    OPENAPI --> API
    API --> ASYNCAPI
    ASYNCAPI --> EVENTS
    OPENAPI --> SCHEMAS
    ASYNCAPI --> SCHEMAS
    SCHEMAS --> GOVERNANCE
```

The current FastAPI-generated OpenAPI 3.1 schema is the authoritative live REST contract. AsyncAPI, independently versioned contract files, event-driven integrations, and formal compatibility gates are planned extensions.

### Example BDD scenario

```gherkin
Feature: Route recovery after telemetry loss

  Scenario: Re-plan a delivery when vehicle GPS becomes unavailable
    Given an authenticated planner has an active delivery route
    And the assigned vehicle has stopped reporting valid GPS coordinates
    When the configured telemetry timeout is exceeded
    Then the platform should mark the vehicle position as stale
    And it should preserve the last verified location
    And it should propose a fallback planning strategy
    And it should require human approval before operational execution
```

This scenario documents a target resilience behavior; it is not a claim that GPS telemetry, automated re-routing, or approval orchestration is currently implemented.

### Spatial-temporal visualization target

Future spatial reports should provide:

- Warehouse, order, supplier, vehicle, and service-region layers
- Time-filtered demand, capacity, delay, and fulfillment views
- Road-network routes with distance, duration, traffic, and vehicle constraints
- Scenario overlays comparing baseline and optimized plans
- Exportable, self-contained HTML reports for audit and review
- Explicit coordinate reference systems, units, timestamps, data freshness, and privacy classification

### AI-generated living context

A future controlled documentation agent may inspect code changes, tests, contracts, schemas, and migration files to propose updates to designated generated sections. The workflow must:

1. Detect relevant changes from a version-controlled diff.
2. Map changed components to owned documentation sections.
3. Generate a proposed patch rather than directly rewriting documentation.
4. Cite the code, contract, test, or schema supporting each material claim.
5. Run documentation, link, diagram, and executable-example checks.
6. Require human review before merge.
7. Preserve an audit trail of the model, prompt, inputs, and approved output.

## ✨ Implemented Capabilities

### 🔐 Authentication and user isolation

- Registers users with validated email addresses and passwords of at least eight characters.
- Stores bcrypt password hashes rather than plaintext passwords.
- Issues HS256 JWT access tokens with issued-at and expiration claims.
- Protects warehouse, order, route, and profile workflows with bearer-token authentication.
- Filters warehouse and order records by the authenticated owner.

### 🏭 Warehouse operations

- Creates warehouses with name, coordinates, capacity, and owner association.
- Lists and retrieves warehouses owned by the current user.
- Deletes user-owned warehouses.
- Selects the nearest available warehouse for an unassigned order using the current coordinate-distance policy.

### 📦 Order operations

- Creates orders containing description, coordinates, demand, status, and ownership.
- Supports explicit warehouse assignment.
- Automatically allocates an available warehouse when an order is created without a warehouse ID.
- Tracks pending and planned states.
- Lists, retrieves, and deletes records within the authenticated user boundary.

### 📈 Demand forecasting

The online forecasting service:

1. Converts demand history into a numeric series.
2. Uses the series mean for short histories.
3. Uses the latest three-observation moving average when sufficient history exists.
4. Produces a seven-day forecast by default.
5. Adds normally distributed variation and clips negative predictions to zero.

This is a lightweight forecasting baseline, not a trained probabilistic forecasting model. Repeated calls may produce different values because the online function does not currently fix a random seed.

### 🧮 Inventory optimization

The inventory service calculates a service-level-driven replenishment recommendation for every demand item.

For forecast demand $D$, lead time $L$, demand deviation $\sigma$, current inventory $I$, and service-level safety factor $z$:

```math
\text{Safety Stock} = z \cdot \sigma \cdot \sqrt{\max(1, L)}
```

```math
\text{Reorder Point} = (D \cdot L) + \text{Safety Stock}
```

```math
\text{Recommended Order Quantity}
= \max\left(0,\ \text{Reorder Point} - I\right)
```

The service also reports expected shortage and a simplified cost proxy using holding and shortage costs. Inputs are validated at schema and domain levels, and invalid business values produce structured HTTP errors.

### 🗺️ Route optimization

- Groups authenticated orders by their assigned warehouse.
- Builds a distance matrix from warehouse and order coordinates.
- Starts from the first stop and iteratively selects the nearest unvisited stop.
- Returns ordered stops, route indexes, grouped route metadata, and accumulated coordinate distance.
- Renders optimized sequences on the dashboard map.

The current implementation is a nearest-neighbor heuristic operating on Euclidean latitude/longitude differences. It does not yet model road networks, travel time, vehicle capacity, traffic, or distance in kilometers.

### 🧪 Synthetic data generation

The repository contains two simulation paths:

- **Online demand simulation** — generates non-negative daily product demand using Gaussian sampling.
- **Offline research generator** — creates higher-volume SKU/warehouse datasets with configurable random seeds, negative-binomial demand, weekly and annual seasonality, trend, payday effects, promotions, price variation, and elasticity.

The calendar generator produces daily temporal features including year, month, day, weekday, weekend status, and an example payday indicator for day 25 of each month.

### 📊 Operational dashboard

The live dashboard integrates:

- Authentication and profile loading
- Warehouse and order tables
- Warehouse and order creation
- KPI summaries
- Capacity-versus-demand visualization
- Interactive OpenStreetMap/Leaflet logistics mapping
- Warehouse-to-order allocation lines
- Optimized route overlays
- Inventory optimization forms and results
- Combined logistics planning
- Per-module status and dashboard error history

## 🧠 Decision Flow

```mermaid
flowchart TB
    subgraph INPUT["Validated Logistics Inputs"]
        HISTORY["Demand History"]
        LOCATION["Order Coordinates"]
        WAREHOUSES["Warehouse Candidates"]
        MATRIX["Distance Matrix"]
        LABELS["Stop Labels"]
    end

    subgraph EXECUTION["Combined Planning Execution"]
        FORECAST["7-Day Demand Forecast"]
        ALLOCATE["Nearest-Warehouse Selection"]
        ROUTE["Nearest-Neighbor Route"]
        COMPOSE["Plan Composition"]
    end

    subgraph OUTPUT["Structured API Response"]
        PREDICTION["Forecast Values"]
        SELECTION["Selected Warehouse<br/>Coordinate Distance"]
        SEQUENCE["Route Indexes<br/>Route Labels"]
    end

    subgraph EXPERIENCE["Dashboard Projection"]
        TABLE["Planning Result Table"]
        STATUS["Module Status"]
        MAP["Operational Map Context"]
    end

    HISTORY --> FORECAST
    LOCATION --> ALLOCATE
    WAREHOUSES --> ALLOCATE
    MATRIX --> ROUTE
    LABELS --> ROUTE
    FORECAST --> COMPOSE
    ALLOCATE --> COMPOSE
    ROUTE --> COMPOSE
    COMPOSE --> PREDICTION
    COMPOSE --> SELECTION
    COMPOSE --> SEQUENCE
    PREDICTION --> TABLE
    SELECTION --> TABLE
    SEQUENCE --> TABLE
    COMPOSE --> STATUS
    SEQUENCE -. "Related route view" .-> MAP
```

The combined `/ai/logistics-plan` operation executes the forecast, warehouse allocation, and route-sequencing services in one request and returns a consolidated plan.

## 🔌 API Surface

### Platform and identity

| Method | Endpoint | Purpose | Authentication |
|---|---|---|---|
| `GET` | `/` | API availability message | No |
| `GET` | `/health` | Lightweight health response | No |
| `POST` | `/register` | Create a user account | No |
| `POST` | `/login` | Exchange credentials for a bearer token | No |
| `GET` | `/me` | Return the authenticated profile | Bearer token |

### Decision services

| Method | Endpoint | Purpose | Authentication |
|---|---|---|---|
| `POST` | `/optimize` | Calculate inventory recommendations | No |
| `GET` | `/simulation/demand` | Generate daily synthetic demand | No |
| `POST` | `/ai/forecast` | Produce a short-horizon demand forecast | No |
| `POST` | `/ai/route-optimize` | Sequence stops from a distance matrix | No |
| `POST` | `/ai/warehouse-allocate` | Select the closest warehouse | No |
| `POST` | `/ai/logistics-plan` | Run the combined planning workflow | No |

### Authenticated operations

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/warehouses/` | Create a warehouse |
| `GET` | `/warehouses/` | List owned warehouses |
| `GET` | `/warehouses/{warehouse_id}` | Retrieve an owned warehouse |
| `DELETE` | `/warehouses/{warehouse_id}` | Delete an owned warehouse |
| `POST` | `/orders/` | Create and optionally allocate an order |
| `GET` | `/orders/` | List owned orders |
| `GET` | `/orders/{order_id}` | Retrieve an owned order |
| `DELETE` | `/orders/{order_id}` | Delete an owned order |
| `POST` | `/routes/optimize` | Build grouped routes for owned orders |

The live OpenAPI schema remains the authoritative machine-readable contract: [`/openapi.json`](https://lieferkette-optimierungsplattform.onrender.com/openapi.json).

## 🗃️ Data Model

```mermaid
erDiagram
    USER ||--o{ WAREHOUSE : owns
    USER ||--o{ ORDER : creates
    WAREHOUSE o|--o{ ORDER : receives

    USER {
        int id PK
        string email UK
        string full_name
        string hashed_password
        boolean disabled
    }

    WAREHOUSE {
        int id PK
        string name
        float latitude
        float longitude
        int capacity
        int owner_id FK
    }

    ORDER {
        int id PK
        string description
        float latitude
        float longitude
        int demand
        string status
        int warehouse_id FK
        int owner_id FK
        datetime created_at
    }
```

## 📐 Data Governance and Contract Rules

The current Pydantic and ORM models provide application-level validation and persistence mapping. The planned governance layer will add versioned JSON Schema or Protobuf definitions covering:

- Entity identifiers, ownership, tenancy, and lifecycle state
- Coordinate systems, geographic precision, and location-data classification
- Timestamps, time zones, freshness, and event ordering
- Demand, capacity, inventory, distance, duration, cost, currency, and fuel units
- Required fields, value ranges, nullability, enumerations, and compatibility policy
- Data lineage, retention, deletion, consent, access, and audit metadata

Schema changes should be backward-compatible by default and validated in CI against API contracts, event contracts, fixtures, migrations, and consumer expectations.

## 📊 KPI, Analytics, MLOps, and FinOps Direction

Optimization modules should document the business metric they influence without presenting synthetic results as actual business performance.

| Dimension | Candidate metrics |
|---|---|
| Forecasting | MAE, RMSE, MAPE/WAPE, bias, interval coverage |
| Inventory | Service level, stockout rate, safety stock, carrying cost, inventory turns |
| Allocation | Capacity utilization, assignment cost, SLA compliance, unallocated demand |
| Routing | Distance, duration, fuel estimate, emissions estimate, on-time-delivery rate |
| Platform | Request rate, latency, error rate, availability, queue time |
| MLOps | Data drift, model drift, experiment version, approval state, rollback status |
| FinOps | Compute cost per forecast, optimization run, tenant, route, and fulfilled order |

Future MLflow-compatible decision tracking should capture input references, dataset and feature versions, algorithm or model version, parameters, metrics, artifacts, output, timestamp, actor, and approval state. Sensitive payloads and precise location data must not be copied into experiment tracking without an explicit governance policy.

## 🧰 Technology Stack

| Area | Technologies |
|---|---|
| 🐍 Runtime | Python |
| 🚀 API | FastAPI, Uvicorn |
| 🧾 Contracts | Pydantic, OpenAPI 3.1 |
| 🔐 Security | OAuth2 password flow, JWT, Passlib, bcrypt |
| 🗃️ Data | PostgreSQL, SQLAlchemy, Psycopg 3 |
| 🧠 Analytics | NumPy, pandas |
| 🖥️ Dashboard | HTML, CSS, JavaScript |
| 📊 Visualization | Chart.js |
| 🗺️ Mapping | Leaflet, OpenStreetMap |
| ✅ Quality | pytest, GitHub Actions |
| ☁️ Delivery | GitHub-connected continuous deployment |

## 📁 Repository Structure

```text
.
├── .github/
│   └── workflows/
│       └── python-tests.yml
├── app/
│   ├── app/
│   │   └── simulation/
│   ├── dashboard/
│   ├── models/
│   ├── routers/
│   ├── services/
│   └── database.py
├── dashboard/
│   └── index.html
├── quellcode/
│   └── datensimulation/
├── tests/
├── anforderungen.txt
├── main.py
└── README.md
```

### Planned documentation and operations structure

The following structure is a target, not a representation of files currently present:

```text
.
├── docs/
│   ├── architecture/
│   ├── executable/
│   ├── logistics-maps/
│   └── ux/
├── features/
├── monitoring/
│   ├── grafana/
│   └── prometheus/
├── schemas/
├── scripts/
│   └── documentation/
├── asyncapi.yaml
├── COMPLIANCE.md
├── compose.yaml
├── ROADMAP.md
└── SECURITY.md
```

## ⚙️ Local Setup

### Prerequisites

- Python 3.11 recommended
- A reachable PostgreSQL database
- Git

### Installation

```bash
git clone https://github.com/aminbita162-glitch/lieferkette-optimierungsplattform.git
cd lieferkette-optimierungsplattform

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r anforderungen.txt
```

### Environment configuration

The application requires `DATABASE_URL`.

```bash
export DATABASE_URL="postgresql://USER:PASSWORD@HOST:PORT/DATABASE"
```

The database adapter automatically converts a standard `postgresql://` URL to the Psycopg 3 SQLAlchemy dialect.

### Start the application

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

- Dashboard: <http://localhost:8000/dashboard/>
- Swagger UI: <http://localhost:8000/docs>
- ReDoc: <http://localhost:8000/redoc>
- Health: <http://localhost:8000/health>

### Target one-command orchestration

A future `compose.yaml` should make the complete local ecosystem reproducible through one explicit command:

```bash
docker compose up --build
```

The target composition may include the API, PostgreSQL, migrations, documentation builder, Prometheus, Grafana, and optional model-tracking services. This command is not available until the corresponding Compose definition is added and verified.

## 🔐 Authentication Example

### Register

```bash
curl -X POST "http://localhost:8000/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "replace-with-a-secure-password",
    "full_name": "Example User"
  }'
```

### Login

```bash
curl -X POST "http://localhost:8000/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=replace-with-a-secure-password"
```

Use the returned token:

```bash
curl "http://localhost:8000/me" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## ✅ Testing and Continuous Integration

Run the current test suite:

```bash
pytest -q
```

The GitHub Actions workflow:

- Runs on pushes and pull requests targeting `main`
- Uses Python 3.11
- Installs dependencies from `anforderungen.txt`
- Sets the repository root as `PYTHONPATH`
- Executes `pytest -q`

The current automated suite verifies calendar generation: expected columns, inclusive date ranges, binary weekend flags, and the example payday rule. API, authentication, database, optimization, routing, and dashboard tests are planned but are not part of the current suite.

### Target quality gates

The planned verification pipeline should add:

- Unit, integration, contract, ownership, security, and negative-path tests
- Gherkin scenarios connected to executable BDD steps
- OpenAPI and AsyncAPI validation
- JSON Schema or Protobuf compatibility checks
- Type checking, linting, formatting, dependency review, and secret scanning
- Executable Quarto or Jupyter Book builds
- Mermaid rendering and internal-link verification
- Docker Compose smoke tests
- Deterministic reference scenarios for decision services
- Performance, resilience, and recovery tests with published acceptance thresholds

## 🔬 Verified Live Behavior

The following workflows have been exercised against the deployed application:

| Workflow | Observed result |
|---|---|
| ❤️ Service health | `/health` returned `{"ok": true}` |
| 🔐 Authentication | Login succeeded and the authenticated profile loaded |
| 🏭 Warehouse operations | Authenticated warehouse data loaded in the dashboard |
| 📦 Order operations | Authenticated planned orders loaded with warehouse assignments |
| 🧮 Inventory optimization | Returned recommendation, safety stock, reorder point, shortage, and cost fields |
| 🧠 Combined logistics plan | Returned a seven-day forecast, warehouse selection, distance score, route indexes, and route labels |
| 🗺️ Route visualization | Optimized route loaded and rendered on the interactive map |
| 📚 API contract | Swagger UI and OpenAPI 3.1 schema loaded successfully |
| ✅ Continuous integration | The inspected Python Tests workflow completed successfully |

These checks demonstrate functional integration; they are not a performance benchmark, production certification, or claim of model accuracy.

## 🧾 Data and Evidence Boundary

> [!IMPORTANT]
> All warehouse records, orders, coordinates, SKUs, demand histories, costs, capacities, credentials, forecasts, optimization outputs, and dashboard values shown in this project are synthetic, illustrative, or created specifically for testing. No result in this repository should be interpreted as an operational business KPI, customer record, production forecast, or decision derived from real enterprise data.

The simulation and live-demo evidence establishes software behavior under test inputs. Real-world adoption would require governed data onboarding, domain-specific validation, calibrated models, operational acceptance criteria, and continuous monitoring.

## 🔒 Security, Privacy, and Compliance Direction

The platform processes data types that may become operationally sensitive, especially geographic positions, routes, account identities, costs, and decision histories. Production hardening should include:

- Encryption in transit and at rest
- Managed secrets and signing-key rotation
- Least-privilege access and organization-aware authorization
- Location precision minimization and purpose limitation
- Configurable retention, deletion, export, and legal-hold workflows
- Tamper-evident security and decision audit trails
- Dependency, container, code, and secret scanning
- Incident reporting, response, recovery, and disclosure procedures
- Jurisdiction-specific assessments for geographic, employee, driver, supplier, and customer data

A future `SECURITY.md` must define the security-reporting process and supported versions. A future `COMPLIANCE.md` must map applicable obligations to technical controls and evidence. Neither document should claim certification without independent, current evidence.

## 🛡️ Resilience and Fallback Strategy

Future operational behavior should define safe degradation for:

| Failure | Planned fallback behavior |
|---|---|
| GPS or telemetry loss | Mark location as stale, retain the last verified position, reduce automation, and request operator review |
| Mapping provider outage | Preserve cached operational context where permitted and disable unsupported route claims |
| Forecast-service failure | Return a traceable baseline or last approved forecast with explicit freshness and confidence warnings |
| Optimization timeout | Return no recommendation or a validated fallback heuristic with a clear degraded-state indicator |
| Database interruption | Reject unsafe writes, use bounded retries, expose dependency health, and preserve idempotency |
| Model-quality breach | Stop promotion, revert to the last approved version, and require investigation |
| External event duplication | Apply event identity, deduplication, ordering, and idempotent handling |
| Partial orchestration failure | Compensate or roll back where safe and preserve an auditable recovery record |

Fallbacks must never silently present degraded results as fully optimized or current.

## ⚠️ Current Engineering Boundaries

| Area | Current boundary |
|---|---|
| 📈 Forecasting | Moving-average baseline with unseeded Gaussian variation; no training, backtesting, confidence intervals, or accuracy registry |
| 🧮 Inventory policy | Deterministic single-policy calculation and simplified cost proxy |
| 🗺️ Routing | Nearest-neighbor heuristic over coordinate differences; no road graph, fleet constraints, traffic, or route optimality guarantee |
| 🏭 Allocation | Selects by coordinate proximity; does not yet enforce capacity, stock availability, cost, SLA, or service-region constraints |
| 🔐 Security | Authentication exists, but deployment secrets, signing-key management, CORS restrictions, authorization design, and token lifecycle require production hardening |
| 🗃️ Database lifecycle | Tables are created at application startup; versioned schema migrations are not yet implemented |
| ✅ Verification | Current automated coverage is limited to calendar generation |
| 📦 Packaging | The nested application package should be consolidated into one canonical package structure |
| 📊 Observability | Dashboard status reporting is client-side; centralized logs, metrics, traces, alerting, and SLOs are not yet implemented |

## 🗺️ Four-Phase Engineering Roadmap

Everything in this section is planned work. Roadmap items must not be interpreted as currently implemented functionality.

### Phase 1 — Reliability, Security, and Reproducibility

**Objective:** Convert the working prototype into a deterministic, testable, and operationally safer engineering baseline.

1. **Repository and configuration consolidation**
   - Consolidate the nested package structure.
   - Centralize configuration and authentication dependencies.
   - Move signing keys and environment-specific settings to managed configuration.
   - Replace startup table creation with version-controlled database migrations.
   - Establish Google- or NumPy-style docstrings with executable examples for logistics functions.

2. **Correctness and validation**
   - Add deterministic random seeds where reproducibility is required.
   - Introduce strict validation for matrices, coordinates, capacities, demand, and cost units.
   - Replace coordinate-distance ambiguity with explicit geospatial units.
   - Standardize error contracts across every router.
   - Introduce versioned JSON Schema definitions for core logistics entities.

3. **Verification expansion**
   - Add unit tests for all decision services.
   - Add API, authentication, ownership, database, and negative-path integration tests.
   - Add deterministic reference scenarios for forecasting and optimization.
   - Publish reproducible verification results and model assumptions.
   - Add BDD scenarios for delay, GPS loss, capacity exhaustion, re-routing, and service degradation.

4. **Operational hardening**
   - Add structured logging, request correlation, readiness checks, and dependency health checks.
   - Define CI quality gates for tests, formatting, typing, security checks, dependency review, and documentation builds.
   - Establish backup, recovery, and rollback procedures.
   - Create `SECURITY.md`, `COMPLIANCE.md`, and operational runbooks.

5. **Executable documentation**
   - Introduce a Quarto or Jupyter Book documentation project.
   - Execute forecasting, inventory, allocation, and routing examples during CI.
   - Generate reproducible Folium or Deck.gl logistics-map artifacts from synthetic scenarios.
   - Validate Mermaid diagrams, links, OpenAPI contracts, and code examples.

### Phase 2 — Enterprise and SaaS Platform Foundation

**Objective:** Evolve the hardened system into a scalable, governed, multi-organization supply-chain platform.

1. **Platform architecture**
   - Introduce organization-aware tenancy, role-based access control, and auditable permissions.
   - Separate API, domain, data, and decision-model boundaries.
   - Add asynchronous workloads for long-running simulations and optimization jobs.
   - Version APIs and support backward-compatible client evolution.
   - Define AsyncAPI contracts for real-time logistics and IoT events.
   - Document load-balancing, queueing, backpressure, partitioning, caching, and horizontal-scaling policies.

2. **Supply-chain intelligence**
   - Add calibrated time-series forecasting with backtesting and uncertainty intervals.
   - Extend inventory optimization to multi-SKU and multi-echelon planning.
   - Add capacity-, inventory-, cost-, and SLA-aware warehouse allocation.
   - Integrate road-network routing, vehicle constraints, travel time, and scenario comparison.
   - Expand toward supplier risk, order fulfillment, production scheduling, and network-design workflows.
   - Generate spatial-temporal HTML reports for routes, warehouses, demand, capacity, and fleet scenarios.

3. **MLOps and governance**
   - Add dataset, feature, experiment, and model versioning.
   - Establish model evaluation, registry, approval, deployment, and rollback workflows.
   - Introduce explainability artifacts, decision lineage, audit logs, and policy controls.
   - Monitor service quality, data quality, model performance, latency, and cost.
   - Integrate MLflow-compatible experiment and decision tracking.

4. **SaaS operations**
   - Add tenant onboarding, quotas, usage metering, rate limiting, and lifecycle controls.
   - Introduce scalable deployment environments and isolated background processing.
   - Define SLOs, incident procedures, disaster recovery, and compliance evidence.
   - Version Prometheus metrics, Grafana dashboards, alerts, and runbooks.
   - Track cloud processing, storage, model-inference, and logistics fuel-cost metadata.

5. **Experience design**
   - Create a documented dashboard design system under `docs/ux/`.
   - Define responsive behavior, accessibility requirements, loading states, empty states, errors, maps, tables, charts, and approval interactions.
   - Validate critical workflows with task-level usability criteria.

### Phase 3 — Retrieval-Augmented Supply-Chain Intelligence

**Objective:** Combine structured operational data with verified enterprise knowledge for evidence-backed decisions.

1. **Retrieval foundation**
   - Introduce a vector database, semantic retrieval, document ingestion, and versioned knowledge artifacts.
   - Orchestrate relational data, vector search, and language-model workflows.
   - Add citation integrity, source verification, access controls, and retrieval evaluation.

2. **Domain RAG services**
   - Supplier contract and compliance analysis
   - Warehouse maintenance-manual retrieval
   - Market-intelligence retrieval
   - Policy-aware executive reporting

3. **Decision experience**
   - Provide answers with evidence, assumptions, confidence, and alternatives.
   - Connect retrieved knowledge to forecasting, sourcing, risk, fulfillment, and maintenance workflows.
   - Add human approval gates for material operational recommendations.

4. **Living context automation**
   - Add a controlled LLM-based documentation assistant.
   - Analyze repository diffs and propose traceable documentation updates.
   - Require source-linked claims, automated validation, and human approval.
   - Record the model, prompt, input revision, generated patch, reviewer, and final decision.

### Phase 4 — Autonomous, Adaptive, and Self-Healing Operations

**Objective:** Build governed autonomy that can detect, explain, coordinate, and recover while preserving human control.

1. **Adaptive intelligence**
   - Time-series anomaly detection and preventive forecasting
   - Dynamic reinforcement-learning policies in controlled simulation environments
   - Automated data- and model-drift detection with governed retraining
   - Explainable decisions, auditability, and approval workflows

2. **Multi-agent orchestration**
   - Specialized planning, sourcing, risk, inventory, and logistics agents
   - Negotiation and coordination protocols
   - Shared memory, policy constraints, action budgets, and escalation paths
   - Simulation-first validation before real-world execution

3. **Root-cause and recovery systems**
   - Automated incident correlation and root-cause analysis
   - Service recovery, failover, redeployment, and traffic rerouting
   - Vector-index integrity checks and retrieval-quality safeguards
   - Context-integrity and hallucination controls

4. **Resilience engineering**
   - Controlled fault injection and chaos experiments
   - Recovery-time and recovery-point objectives
   - Continuous resilience scoring
   - Evidence-backed rollback and post-incident learning

5. **Future-proof model boundaries**
   - Keep forecasting, routing, allocation, and inventory engines behind stable interfaces.
   - Support model replacement without rewriting platform-level documentation or consumer contracts.
   - Separate domain contracts from framework-, vendor-, and model-specific implementations.
   - Validate replacement models against identical reference scenarios and acceptance gates.

## 🧩 Delivery Governance

A structured `ROADMAP.md` should eventually track each roadmap item using:

- Status
- Owner
- Business outcome
- Engineering scope
- Dependencies
- Risks
- Acceptance criteria
- Verification evidence
- Target release
- Actual completion revision

A roadmap checkbox must indicate delivery status only after the capability, tests, documentation, and operational evidence are present.

## 🤝 Engineering Principles

- **Evidence before claims** — implemented, verified, and planned capabilities remain clearly separated.
- **Business problem first** — technical choices must trace back to a measurable operational need.
- **Reproducibility by design** — inputs, assumptions, versions, and outputs should be independently verifiable.
- **Explainability and auditability** — decision support must expose why a recommendation was produced.
- **Security and isolation** — identity, authorization, data boundaries, and secret management are platform concerns.
- **Human-governed automation** — higher autonomy requires stronger controls, evaluation, and recovery mechanisms.
- **Incremental evolution** — enterprise capability is built through verified phases rather than asserted in advance.
- **Contracts before coupling** — services, events, and data structures evolve through explicit versioned interfaces.
- **Documentation as a tested artifact** — examples, diagrams, schemas, and reports must remain synchronized with implementation.
- **Safe degradation** — failures and fallbacks must be visible, bounded, traceable, and reversible.
- **Model replaceability** — domain workflows must not depend permanently on one algorithm, framework, or provider.
- **Cost-aware engineering** — service quality, cloud cost, fuel use, and operational value must be evaluated together.

## 👤 Author

**Amin Azimi**  
AI Architect · AI Development Product Engineer

- GitHub: [@aminbita162-glitch](https://github.com/aminbita162-glitch)

> From fragmented logistics signals to structured, explainable, and progressively autonomous decisions.