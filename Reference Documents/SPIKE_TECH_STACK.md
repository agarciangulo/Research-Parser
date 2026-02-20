# Spike Tech Stack – Three-Process Architecture

This document catalogs the technologies, libraries, and services required for the spike prototype implementing the three-process architecture. It complements `TECH_STACK.md` (current production system) by focusing specifically on the spike's needs.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SPIKE PLATFORM                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │   Process 1     │  │   Process 2     │  │   Process 3     │             │
│  │  Task Intake    │  │  Prioritization │  │  Query System   │             │
│  │  & Processing   │  │  & Insights     │  │                 │             │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘             │
│           │                    │                    │                       │
│           └────────────────────┼────────────────────┘                       │
│                                │                                            │
│                    ┌───────────▼───────────┐                               │
│                    │   LangGraph Engine    │                               │
│                    │   (Orchestration)     │                               │
│                    └───────────┬───────────┘                               │
│                                │                                            │
│      ┌─────────────────────────┼─────────────────────────┐                 │
│      │                         │                         │                 │
│  ┌───▼───┐  ┌───▼───┐  ┌───▼───┐  ┌───▼───┐  ┌───▼───┐ │
│  │Task DB│  │UB DB  │  │BP DB  │  │Ins DB │  │Gemini │ │
│  └───────┘  └───────┘  └───────┘  └───────┘  └───────┘ │
│                                                         │
│                    PostgreSQL              Vertex AI    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Runtime & Language

| Component | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.11+ | Primary language for all spike components |
| **Poetry** or **Pip** | Latest | Dependency management (`requirements.txt` or `pyproject.toml`) |
| **Docker** | Latest | Containerization for Cloud Run deployment |

---

## 2. Orchestration Framework

| Technology | Version | Purpose |
|------------|---------|---------|
| **LangGraph** | 0.2.x+ | Graph-based agent orchestration |
| **LangChain** | 0.2.x+ | LLM abstractions and tools (LangGraph dependency) |

### Why LangGraph?

- **State Management:** Persists conversation and task state across nodes
- **Conditional Routing:** Routes based on intent (activity vs. correction vs. context)
- **Checkpointing:** Saves state to PostgreSQL for durability
- **Observability:** Built-in tracing and debugging

### LangGraph Nodes by Process (Optimized)

| Process | Nodes | LLM Calls |
|---------|-------|-----------|
| **Process 1** | `EmailIntakeNode`, `UnifiedExtractionNode`, `BehaviorAnalyzerNode`, `BehaviorPersistNode`, `TaskComparisonNode`, `TaskPersistNode`, `PresenterNode` | 2 |
| **Process 2** | `TaskFetchNode`, `PrioritizerNode`, `BestPracticesFetchNode`, `UserBehaviourFetchNode`, `InsightGeneratorNode`, `InsightPersistNode`, `EmailComposerNode` | 2 |
| **Process 3** | `QueryParserNode`, `AccessControlNode`, `CoordinatorNode`, `DataFetchNode`, `ResponseGeneratorNode` | 2 |

**Total Nodes:** 19 (down from 23)
**Total LLM Calls per full cycle:** 6 (down from 11)

---

## 3. AI / LLM Layer

| Technology | Model | Purpose |
|------------|-------|---------|
| **Google Vertex AI** | Gemini 3.0 | Primary LLM for all AI tasks |
| **google-generativeai** | Latest SDK | Python client for Gemini API |

### Why Gemini 3.0?

| Feature | Benefit for Spike |
|---------|-------------------|
| **Enhanced reasoning** | Better task extraction and intent classification accuracy |
| **1.5M token context** | Handle long email threads and full task history |
| **Improved multimodal** | Future support for email attachments/images |
| **Faster inference** | Lower latency for real-time email processing |

### Model Selection

| Use Case | Recommended Model |
|----------|-------------------|
| **Production (MVP)** | Gemini 3.0 Flash - Best balance of speed, cost, and quality |
| **Complex reasoning** | Gemini 3.0 Pro - For difficult query decomposition or insight generation |
| **Development/Testing** | Gemini 3.0 Flash - Lower cost during iteration |

### LLM Usage by Node (Optimized)

**Consolidated calls reduce LLM usage from 12 to 6:**

| Node | LLM Call? | Purpose | Replaces |
|------|-----------|---------|----------|
| `UnifiedExtractionNode` | ✅ Yes | Intent + tasks + JSON + context + entities | IntentClassifier, TaskExtractor, TaskTransformer, ContextChecker |
| `BehaviorAnalyzerNode` | ✅ Yes | Detect meta-patterns | - |
| `PrioritizerNode` | ✅ Yes | Generate priority list | - |
| `InsightGeneratorNode` | ✅ Yes | Generate personalized advice (uses Best Practices + User Behaviour) | - |
| `QueryParserNode` | ✅ Yes | Parse + decompose query | QueryIntake, Breakdown |
| `ResponseGeneratorNode` | ✅ Yes | Summarize + synthesize response | Summarizer, Synthesizer |

**LLM Call Reduction:**
| Process | Before | After | Savings |
|---------|--------|-------|---------|
| Process 1 | 5 calls | 2 calls | 60% |
| Process 2 | 2 calls | 2 calls | 0% |
| Process 3 | 4 calls | 2 calls | 50% |
| **Total** | **11 calls** | **6 calls** | **45%** |

### Cost Optimization Notes

- Use **Gemini 3.0 Flash** for most nodes (speed + cost efficiency)
- Consider **Gemini 3.0 Pro** for `BehaviorAnalyzerNode` and `InsightGeneratorNode` where deeper reasoning adds value
- Batch similar operations where possible
- Cache common classifications/transformations
- Monitor token usage per process
- Leverage the 1.5M context window to reduce redundant context-building calls

---

## 4. Database Layer

### PostgreSQL (Cloud SQL)

All four databases are implemented as schemas/tables in a single PostgreSQL instance for MVP simplicity.

| Database | Table Name | Purpose |
|----------|------------|---------|
| **Task DB** | `tasks` | User tasks with status, priority, due dates |
| **User Behaviour DB** | `user_behaviours` | AI-generated behavior observations |
| **Best Practices DB** | `best_practices` | Organizational standards for comparison |
| **Insights DB** | `insights` | Historical insights generated for users |

### Task DB Schema

```sql
CREATE TABLE tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    due_date DATE,
    priority VARCHAR(20) DEFAULT 'medium',  -- high, medium, low
    status VARCHAR(20) DEFAULT 'pending',   -- pending, in_progress, completed
    -- Classification hierarchy: Client > Project > Category
    category VARCHAR(100),                   -- Task type: Admin, Meetings, Development, etc.
    project VARCHAR(255),                    -- Project name: Q4 Budget Review, Website Redesign
    client VARCHAR(255),                     -- Client name: Acme Corp, Beta Industries
    classification_source VARCHAR(20),       -- header, explicit, inferred, unknown
    estimated_hours DECIMAL(5,2),
    source_email_id VARCHAR(255),
    raw_text TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX idx_tasks_user_client ON tasks(user_id, client);
CREATE INDEX idx_tasks_user_project ON tasks(user_id, project);
CREATE INDEX idx_tasks_user_status ON tasks(user_id, status);
```

### User Behaviour DB Schema

```sql
CREATE TABLE user_behaviours (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    observation TEXT NOT NULL,           -- AI-generated observation
    pattern_type VARCHAR(100),           -- e.g., "missing_context", "date_corrections"
    frequency INTEGER DEFAULT 1,
    first_detected_at TIMESTAMP DEFAULT NOW(),
    last_detected_at TIMESTAMP DEFAULT NOW()
);
```

### Best Practices DB Schema

```sql
CREATE TABLE best_practices (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    category VARCHAR(100) NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    priority INTEGER DEFAULT 0,          -- For ordering
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Insights DB Schema

```sql
CREATE TABLE insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    insight_type VARCHAR(50),            -- "priority", "improvement", "reminder"
    content TEXT NOT NULL,
    source_practice_id UUID REFERENCES best_practices(id),
    delivered_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Recurring Patterns Schema

```sql
CREATE TABLE recurring_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    title VARCHAR(500) NOT NULL,
    expected_frequency VARCHAR(50),        -- "daily", "weekly:mon,wed,fri", "monthly"
    detection_method VARCHAR(50),          -- "language", "pattern_analysis", "user_explicit"
    status VARCHAR(20) DEFAULT 'assumed',  -- "assumed", "confirmed", "rejected"
    announced_at TIMESTAMP,                -- When user was notified
    last_logged_at TIMESTAMP,
    streak_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Users Table (Simple Auth for MVP)

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    role VARCHAR(50) DEFAULT 'user',      -- 'user' or 'manager'
    timezone VARCHAR(50) DEFAULT 'UTC',
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Database Access

| Technology | Purpose |
|------------|---------|
| **SQLAlchemy** | ORM for database operations |
| **asyncpg** | Async PostgreSQL driver (optional for performance) |
| **Alembic** | Database migrations |

---

## 5. Email Integration

| Technology | Purpose |
|------------|---------|
| **imaplib** (stdlib) | Gmail IMAP polling for incoming emails |
| **smtplib** (stdlib) | SMTP for sending email responses |
| **email** (stdlib) | Email parsing and construction |

### Email Flow

```
Gmail IMAP → EmailIntakeNode → [Process 1/2/3] → smtplib → Gmail SMTP
```

### Configuration

| Variable | Purpose |
|----------|---------|
| `GMAIL_ADDRESS` | Email address for IMAP/SMTP |
| `GMAIL_APP_PASSWORD` | App-specific password |
| `IMAP_SERVER` | `imap.gmail.com` |
| `SMTP_SERVER` | `smtp.gmail.com` |

---

## 6. Scheduling & Triggers

| Technology | Purpose |
|------------|---------|
| **Google Cloud Scheduler** | Triggers Process 2 daily (e.g., 6:00 PM); also triggered on-demand via email |
| **Cloud Run Jobs** | Executes scheduled processes |

### Trigger Configuration

| Process | Trigger | Frequency |
|---------|---------|-----------|
| Process 1 | Email arrival (IMAP poll) | Every 5 minutes |
| Process 2 | Cloud Scheduler + Email Router | Daily at configured time OR on-demand via email |
| Process 3 | Email arrival (query detected) | Every 5 minutes |

---

## 7. Deployment Infrastructure

| Service | Purpose |
|---------|---------|
| **Google Cloud Run** | Hosts spike services (stateless, auto-scaling) |
| **Google Cloud SQL** | Managed PostgreSQL instance |
| **Google Secret Manager** | Secure storage for API keys and credentials |
| **Google Cloud Build** | CI/CD for container images |
| **Google Artifact Registry** | Docker image storage |

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Google Cloud                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐     ┌─────────────┐     ┌──────────┐ │
│  │ Cloud       │     │ Cloud Run   │     │ Cloud    │ │
│  │ Scheduler   │────▶│ Services    │────▶│ SQL      │ │
│  └─────────────┘     └──────┬──────┘     │(Postgres)│ │
│                             │            └──────────┘ │
│                             │                         │
│  ┌─────────────┐            │            ┌──────────┐ │
│  │ Gmail       │◀───────────┘            │ Vertex   │ │
│  │ (IMAP/SMTP) │                         │ AI       │ │
│  └─────────────┘                         └──────────┘ │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 8. Observability

| Technology | Purpose |
|------------|---------|
| **Google Cloud Logging** | Centralized log collection |
| **Google Cloud Monitoring** | Metrics and alerting |
| **LangSmith** (optional) | LangGraph tracing and debugging |

### Key Metrics to Track

| Metric | Process | Purpose |
|--------|---------|---------|
| End-to-end latency | All | Performance monitoring |
| LLM token usage | All | Cost optimization |
| Task extraction accuracy | Process 1 | Quality monitoring |
| Intent classification accuracy | Process 1 | Quality monitoring |
| Email delivery rate | All | Reliability |
| Query response time | Process 3 | User experience |

---

## 9. Development Tools

| Tool | Purpose |
|------|---------|
| **pytest** | Unit and integration testing |
| **black** | Code formatting |
| **ruff** | Linting |
| **mypy** | Type checking |
| **pre-commit** | Git hooks for quality gates |

---

## 10. Environment Variables

### Required Variables

| Variable | Category | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | AI | Vertex AI / Gemini API key |
| `DATABASE_URL` | Database | PostgreSQL connection string |
| `GMAIL_ADDRESS` | Email | Gmail address for IMAP/SMTP |
| `GMAIL_APP_PASSWORD` | Email | App-specific password |
| `GCP_PROJECT_ID` | Infrastructure | Google Cloud project |

### Optional Variables

| Variable | Category | Description |
|----------|----------|-------------|
| `LANGSMITH_API_KEY` | Observability | LangSmith tracing (optional) |
| `LOG_LEVEL` | Logging | DEBUG, INFO, WARNING, ERROR |
| `PROCESS_2_SCHEDULE` | Scheduling | Cron expression for Process 2 |

---

## 11. Dependencies (requirements.txt)

```
# Core
python-dotenv>=1.0.0
pydantic>=2.0.0

# LLM & Orchestration
langchain>=0.2.0
langgraph>=0.2.0
google-generativeai>=0.5.0

# Database
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
alembic>=1.13.0

# Email
# (using stdlib: imaplib, smtplib, email)

# Web (for health checks / admin)
flask>=3.0.0
flask-cors>=4.0.0

# Testing
pytest>=8.0.0
pytest-asyncio>=0.23.0

# Development
black>=24.0.0
ruff>=0.3.0
mypy>=1.9.0
pre-commit>=3.6.0
```

---

## 12. Comparison: Current System vs. Spike

| Aspect | Current System | Spike |
|--------|----------------|-------|
| **Orchestration** | Custom Python + Celery | LangGraph |
| **Task Storage** | Notion API | PostgreSQL |
| **State Management** | File-based (JSON) | LangGraph checkpoints (Postgres) |
| **LLM Provider** | Gemini 1.5 (direct) | Gemini 3.0 via LangChain |
| **Scheduling** | Celery Beat | Cloud Scheduler |
| **Email** | Custom processing | Same (IMAP/SMTP) |
| **Architecture** | Single pipeline | Three distinct processes |
| **Behavior Analysis** | None | AI-powered Behavior Analyzer |
| **Query System** | None | Full Query Router with access control |

---

## Next Steps

1. Set up development environment with these dependencies
2. Create PostgreSQL schemas for all four databases
3. Implement LangGraph flows for each process
4. Configure Cloud Scheduler for Process 2
5. Deploy to Cloud Run for testing

See `SPIKE_EXECUTION_PLAN.md` for the day-by-day implementation guide.

