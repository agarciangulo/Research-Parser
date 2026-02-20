# Spike Roadmap – Progression from MVP to Enterprise Platform

This roadmap outlines the planned evolution of the spike prototype into a full-featured, multi-tenant agent platform. The spike implements **three core processes**; this roadmap shows how each process evolves across phases.

---

## Three Core Processes

| Process | MVP Focus | Long-Term Vision |
|---------|-----------|------------------|
| **Process 1: Task Intake & Processing** | Email intake, JSON transformation, behavior analysis | Multi-channel intake, advanced NLP |
| **Process 2: Prioritization & Insights** | Daily + on-demand priorities, insights, outstanding tasks | Real-time coaching, predictive analytics |
| **Process 3: Query & Analytics** | Email-based queries, strict access control | Full UI, dashboards, reporting |

---

## Phase 0 – Spike MVP (Current Focus)

**Objective:** Validate all three processes end-to-end with working prototypes.

### Process 1: Task Intake & Processing
- [ ] Email intake (activities, corrections, context replies)
- [ ] Intent classification (Gemini)
- [ ] Task extraction and JSON transformation
- [ ] Task comparison (match classification: status change, progress update, recurring, correction, new)
- [ ] AI-powered Behavior Analyzer (meta-pattern detection)
- [ ] Task Presenter (processed tasks, correction confirmations)

### Process 2: Prioritization & Insights
- [ ] Scheduled daily trigger (Cloud Scheduler)
- [ ] On-demand trigger via email request
- [ ] Task Prioritizer (reads Task DB, generates priority list)
- [ ] Insight Generator (compares tasks to Best Practices DB + User Behaviour DB)
- [ ] Insights stored to Insights DB
- [ ] Combined email output (priorities, insights, outstanding tasks)

### Process 3: Query & Analytics (MVP)
- [ ] Email-based query interface
- [ ] Query Router (Breakdown → Coordinator → Synthesizer)
- [ ] Access control enforcement (users: own data only, managers: full access)

### Infrastructure
- [ ] Four PostgreSQL databases (Task, User Behaviour, Best Practices, Insights)
- [ ] LangGraph orchestration with state persistence
- [ ] Vertex AI (Gemini 3.0) for all LLM calls
- [ ] Basic observability (logs, latency metrics)
- [ ] Single-tenant operation (no user segregation)

**Exit criteria:** All three processes work reliably, latency/cost acceptable, architecture documentation complete.

---

## Phase 1 – Multi-User Foundations

**Objective:** Introduce user management and secure access across all processes.

### Authentication & Authorization
- [ ] Upgrade from simple `users` table lookup to full JWT authentication
- [ ] Implement OAuth integration (Google Workspace, Microsoft 365)
- [ ] Add roles (User, Manager, Admin) with permission enforcement
- [ ] Scope all data by `user_id` / `tenant_id`
- [ ] API key intake for programmatic access
- [ ] Session management and token refresh

### Process 1 Enhancements
- [ ] Per-user Behavior profiles
- [ ] User preferences for task categories and priorities
- [ ] Personalized context checking based on history

### Process 2 Enhancements
- [ ] Per-user scheduling preferences (time of day, frequency)
- [ ] User-specific Best Practices subsets

### Process 3 Enhancements
- [ ] Full access control matrix enforcement
- [ ] Manager access to any user's data
- [ ] Audit logging for all queries

**Dependencies:** Spike MVP completed, decision to proceed with migration.

---

## Phase 2 – Multi-Channel & Automation + AI Optimization

**Objective:** Move beyond email to production-grade intake/output channels, and optimize AI usage.

### Fast Layer (Pre-LLM Processing)
- [ ] Implement embedding-based intent classification (use ChromaDB)
- [ ] Add regex-based entity extraction (dates, times, numbers)
- [ ] Create rules engine for simple pattern matching
- [ ] Route simple cases to Fast Layer, complex to Gemini
- [ ] Expected result: 50% of emails skip LLM entirely

### Intent Classification Cache
- [ ] Cache intent decisions by email embedding similarity
- [ ] If new email is 95%+ similar to cached → reuse intent
- [ ] Reduces repeat LLM calls for similar emails

### Query Pattern Cache (Process 3)
- [ ] Cache common query patterns ("What's due tomorrow?")
- [ ] Template-based responses for frequent queries

### Multi-Channel Intake (Process 1)
- [ ] Gmail webhook or Cloud Function (real-time)
- [ ] Slack bot integration for task capture
- [ ] Microsoft Teams integration
- [ ] Mobile app intake (future)

### Automation & Scheduling
- [ ] n8n or Temporal workflows for reminders and escalations
- [ ] Configurable Process 2 schedules per user
- [ ] Smart notifications (email, Slack, push)

### Query Interface (Process 3)
- [ ] Web UI for query submission (chat-like experience)
- [ ] Query history and saved queries
- [ ] Basic dashboards for personal stats

### Operator Tools
- [ ] Admin dashboard for system oversight
- [ ] User onboarding workflows
- [ ] Feature flags for gradual rollouts

**Dependencies:** Phase 1 auth in place; stable spike infrastructure.

---

## Phase 3 – Enterprise Readiness

**Objective:** Harden for production scale and compliance.

### Observability & Monitoring
- [ ] BigQuery exports for analytics
- [ ] Grafana/Data Studio dashboards
- [ ] Per-process latency and error rate monitoring
- [ ] LLM cost tracking and optimization

### Security & Compliance
- [ ] RBAC with fine-grained permissions
- [ ] Tenant isolation and data segregation
- [ ] Full audit logging with retention policies
- [ ] GDPR/SOC2 compliance features

### Integrations
- [ ] Notion bidirectional sync
- [ ] Jira, ServiceNow, Asana connectors
- [ ] Calendar integration (Google Calendar, Outlook)
- [ ] Custom webhook destinations

### Resilience
- [ ] Disaster recovery plan
- [ ] Automated backups
- [ ] Multi-region deployment (optional)

**Dependencies:** Multi-user adoption; automation flows operating reliably.

---

## Phase 4 – Intelligence & Insights Expansion

**Objective:** Turn the platform into a proactive advisor and analytics powerhouse.

### Advanced Analytics (Process 3)
- [ ] Team and firm-wide dashboards
- [ ] Trend detection and anomaly alerts
- [ ] Productivity metrics and benchmarking
- [ ] Custom report generation

### Proactive Coaching (Process 2)
- [ ] Recommendation engine powered by historical data
- [ ] Predictive insights ("You may miss this deadline based on patterns")
- [ ] Goal tracking and progress visualization

### Platform Extensions
- [ ] Plugin marketplace for third-party adapters
- [ ] Custom AI model fine-tuning
- [ ] Evaluation harness for agent performance (Vertex Eval)

### Behavior Intelligence (Process 1)
- [ ] Cross-user pattern analysis (anonymized)
- [ ] Team behavior insights for managers
- [ ] Automated coaching suggestions

**Dependencies:** Enterprise rollout stable; metrics demonstrating value.

---

## Phase 5 – Non-LLM AI & Advanced ML

**Objective:** Reduce LLM dependency with specialized AI models for speed and cost.

### Small Specialized Models
- [ ] Fine-tuned intent classifier (replace embedding fallback)
- [ ] NER model for entity extraction (dates, people, projects)
- [ ] Sentiment analysis model for user tone detection
- [ ] Priority scoring model based on historical patterns

### Time Series & Predictions
- [ ] Predict task completion likelihood
- [ ] Anomaly detection (user disengagement)
- [ ] Smart scheduling recommendations
- [ ] Workload forecasting

### Advanced Embeddings
- [ ] Task clustering for automatic project grouping
- [ ] User behavior clustering for cohort analysis
- [ ] Semantic search across all databases

### Recommendation Systems
- [ ] "Next task" recommendations based on context
- [ ] Collaborative filtering for best practices
- [ ] Personalized insight ranking

**Dependencies:** Phase 4 analytics providing training data; ML infrastructure in place.

---

## AI Strategy Summary

| Phase | AI Focus | Primary Tech |
|-------|----------|--------------|
| **MVP (Phase 0)** | Consolidated LLM calls, existing ChromaDB | Gemini 3.0 |
| **Phase 1** | Per-user AI profiles | Gemini 3.0 |
| **Phase 2** | Fast Layer + caching | Embeddings + Gemini fallback |
| **Phase 3** | Cost optimization, monitoring | LLM cost tracking |
| **Phase 4** | Proactive AI coaching | Gemini 3.0 + analytics |
| **Phase 5** | Specialized ML models | Small models + Gemini |

**Guiding Principle:** Use Gemini (enterprise LLM) as the primary AI, but layer in faster/cheaper alternatives for well-defined tasks as the system matures.

---

## Roadmap Governance

- Review and adjust phases after Spike MVP evaluation.
- Maintain this roadmap in `docs/` and update quarterly or after major milestones.
- Align each phase with resource planning (team members, budget, infra commitments).
- Each phase should have clear exit criteria before proceeding.

---

## Phase Timeline (Estimated)

| Phase | Duration | Key Milestone |
|-------|----------|---------------|
| Phase 0 (MVP) | 1-2 weeks | Working three-process prototype |
| Phase 1 | 2-4 weeks | Multi-user auth, per-user data |
| Phase 2 | 4-6 weeks | Multi-channel intake, Web UI |
| Phase 3 | 6-8 weeks | Enterprise features, compliance |
| Phase 4 | Ongoing | Advanced analytics, AI coaching |

This staged path lets you validate quickly with the core three processes while building a clear bridge to a robust, multi-tenant, enterprise-grade platform.
