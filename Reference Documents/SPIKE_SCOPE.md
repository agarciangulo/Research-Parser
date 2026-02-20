# Spike Scope – AI Task Management Agent (Parallel Prototype)

## Objective

Validate a next-generation AI task management platform by implementing three core processes end-to-end:

1. **Task Intake & Processing** - Intelligent email processing with activity logging, corrections, and context handling
2. **Prioritization & Insights** - Daily task priorities combined with personalized improvement advice
3. **Query & Analytics** - Natural language data access with role-based privacy controls

The spike results will inform the decision to migrate, integrate, or evolve the new architecture alongside the existing system.

---

## Core Processes to Demonstrate

### Process 1: Task Intake & Processing Pipeline

**Trigger:** User sends an email containing activities, corrections, or context replies.

**Capabilities:**
- Classify email intent (new activities vs. corrections vs. context)
- Extract tasks from free-form text or lists
- **Transform to structured JSON:** Use AI to convert extracted tasks into uniform schema (title, due_date, priority, status, etc.)
- Request additional context when information is missing
- Compare new tasks against existing database (AI-powered match classification: status change, progress update, recurring activity, correction, or new task)
- **AI-powered Behavior Analyzer:** Detect meta-patterns in user behavior and generate observations (e.g., "user frequently omits due dates - consider prompting for details")
- Send confirmation/summary emails back to user

**Output:**
- Tasks added/updated in Task DB
- **AI-generated behavior observations** logged to User Behaviour DB
- Email response with processed tasks and correction confirmations

---

### Process 2: Prioritization & Insights Generation

**Triggers:**
- Scheduled daily job (e.g., Cloud Scheduler at 6:00 PM)
- On-demand via email request (e.g., "Send me my status report")

**Capabilities:**
- Read user's task backlog and generate prioritized list for next day
- Analyze tasks against organizational best practices
- Generate personalized improvement advice and recommendations
- Store all insights for historical reference

**Output:**
- Single email containing:
  - Suggested priorities for tomorrow (separate section)
  - Insights and recommendations (separate section)
  - Full list of outstanding tasks (separate section)
- All insights persisted to Insights Database

---

### Process 3: Query & Analytics System

**Trigger:** User or Manager submits a natural language query.

**Interface:**
| Phase | Channel |
|-------|---------|
| **MVP** | Email (e.g., "What tasks do I have due this week?") |
| **Future** | Web UI with chat-like experience |

**Capabilities:**
- Parse and decompose complex queries
- Route to appropriate data sources (Task DB, User Behaviour DB, Insights DB, Best Practices DB)
- Apply access controls based on requester role
- Enforce strict access control based on role

**Access Control:**
| Requester | Own Data | Any User | Firm Total |
|-----------|----------|----------|------------|
| **User** | ✅ Full | ❌ No | ❌ No |
| **Manager** | ✅ Full | ✅ Full | ✅ Full |

**Output:**
- Natural language response to the query
- Appropriate level of detail based on access rights

---

## Success Criteria

| Category | Criteria |
|----------|----------|
| **Process 1** | Email correctly classified; tasks extracted and stored; corrections applied; context requested when needed; user behavior logged |
| **Process 2** | Priorities generated based on due dates and importance; insights compare against best practices and user behavior patterns; combined email sent successfully |
| **Process 3** | Queries parsed correctly; access controls enforced; users restricted to own data only |
| **Data Model** | All four databases (Task, User Behaviour, Best Practices, Insights) implemented and functional |
| **Observability** | Each step logs with run IDs; latency tracked; errors captured with context |
| **Architecture** | Clear separation of three processes; documented data flows; extension points identified |

---

## Databases Required

| Database | Purpose | MVP Scope |
|----------|---------|-----------|
| **Task DB** | Store all user tasks with status, priority, due dates | Full implementation |
| **User Behaviour DB** | Track user patterns and habits | Basic pattern logging |
| **Best Practices DB** | Organizational standards for comparison | Seed with sample practices |
| **Insights DB** | Historical record of generated insights | Full implementation |

---

## Constraints & Assumptions

- **Scope:** Focus on the three core processes; defer complex UI, multi-tenant auth, and non-essential features
- **Stability:** Spike runs in isolated environment; no production traffic
- **Timebox:** 5-7 day effort for MVP of all three processes
- **Team:** Primary owner with optional reviewers
- **Data:** Use synthetic or anonymized data; demo-ready but not production-grade
- **AI/LLM:** Use Google Gemini 3.0 (via Vertex AI) for all AI tasks
- **Process 3 Note:** Query system is foundational MVP; full access control and analytics are future enhancements

---

## Deliverables

### 1. Working Prototype
- [ ] Process 1: Email intake → extraction → comparison → storage → response
- [ ] Process 2: Task prioritization + insight generation → combined email
- [ ] Process 3: Query parsing → routing → access control → response (MVP)
- [ ] Four databases implemented with schemas
- [ ] Email integration (receive and send)

### 2. Architecture Documentation
- [ ] Updated SPIKE_ARCHITECTURE.md with all three processes
- [ ] Data flow diagrams for each process
- [ ] Database schema documentation
- [ ] Component responsibility descriptions

### 3. Evaluation Report
- [ ] Latency measurements for each process
- [ ] AI accuracy observations (task extraction, intent classification)
- [ ] Comparison against current system architecture
- [ ] Recommendations for next phase

---

## Out of Scope (Deferred)

- Production-grade authentication and authorization
- Complex team hierarchy management
- Real-time collaboration features
- Mobile app or dedicated UI (beyond email)
- Notion bidirectional sync (post-MVP)
- Calendar integration (future)

---

## Next Steps

1. Review and approve this scope document
2. Implement Process 1 as the core MVP
3. Add Process 2 for daily value delivery
4. Scaffold Process 3 as foundation for future analytics
5. Document learnings and prepare evaluation report

This scope ensures the spike demonstrates real business value (task management + insights) while establishing the foundation for future analytics capabilities.
