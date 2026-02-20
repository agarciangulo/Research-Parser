# Spike Architecture – AI Task Management Agent (Parallel Prototype)

This document outlines the proposed architecture for the spike described in `docs/SPIKE_SCOPE.md`. It defines three core processes, their components, data flows, and the databases that support them.

---

## 1. System Overview

The AI Task Management Agent consists of **three core processes** that work together to help users manage tasks, receive insights, and query their data:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AI TASK MANAGEMENT AGENT                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │   PROCESS 1     │  │   PROCESS 2     │  │   PROCESS 3     │             │
│  │   Task Intake   │  │  Prioritization │  │  Query System   │             │
│  │   & Processing  │  │   & Insights    │  │  & Analytics    │             │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘             │
│           │                    │                    │                       │
│           ▼                    ▼                    ▼                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        DATA LAYER                                    │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐        │   │
│  │  │ Task DB  │ │ User     │ │ Insights │ │ Best Practices   │        │   │
│  │  │          │ │ Behaviour│ │    DB    │ │       DB         │        │   │
│  │  │          │ │    DB    │ │          │ │                  │        │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1.5 Email Router (Pre-Process)

Before routing to Process 1, 2, or 3, incoming emails pass through the Email Router:

```
┌─────────────┐      ┌─────────────────┐      ┌───────────────────┐
│   EMAIL     │      │  Email Router   │      │  Process 1, 2, or 3│
│   INBOX     │ ──▶  │                 │ ──▶  │                   │
│             │      │ 1. Validate     │      │                   │
└─────────────┘      │ 2. Deduplicate  │      └───────────────────┘
                     │ 3. Classify     │
                     └─────────────────┘
```

### Router Responsibilities

| Step | Action | Details |
|------|--------|---------|
| **1. Validate Sender** | Check `users` table | Reject unknown senders with "please register" reply |
| **2. Deduplicate** | Check Gmail Message-ID | Skip if already processed |
| **3. Classify Intent** | AI classification | Task/Activity → Process 1, Status Request → Process 2, Query → Process 3 |

### Intent Classification Examples

| Email Content | Classification | Route To |
|---------------|----------------|----------|
| "I completed the report today" | Task/Activity | Process 1 |
| "Actually the due date is Thursday" | Correction | Process 1 |
| "Send me my status report" | Status Request | Process 2 |
| "What are my priorities?" | Status Request | Process 2 |
| "Give me a summary of my tasks" | Status Request | Process 2 |
| "What tasks do I have due tomorrow?" | Query | Process 3 |
| "The report is done, what's next?" | Task + Query | Process 1 (primary), note query for follow-up |

---

## 2. Process 1: Task Intake & Processing Pipeline

### 2.1 Overview

This process handles all incoming emails from users, extracts tasks, compares with existing data, and updates the database accordingly.

### 2.2 Flow Diagram

```
┌─────────────┐      ┌─────────────────┐      ┌───────────────────┐         ┌──────────┐
│   EMAIL     │      │  Task Extractor │      │  Task Processor   │  ──▶    │  Task    │
│             │      │                 │      │                   │ writes  │    DB    │
│ Contains:   │ ──▶  │ - Classifies    │ ──▶  │ - Compares new    │         │          │
│ • Activities│      │   intent        │      │   vs existing     │  ◀──    │          │
│ • Corrections      │ - Checks context│      │ - Decides: ADD or │ reads   │          │
│ • Context   │ ◀──  │ - Asks for more │      │   UPDATE          │         │          │
│   replies   │      │   if needed     │      │ - Updates DB      │         │          │
│             │      │ - Extracts to   │      │                   │         │          │
│             │      │   standard JSON │      │                   │         │          │
└─────────────┘      └────────┬────────┘      └───────────────────┘         └────┬─────┘
       ▲                      │                                                  │
       │                      │ analyzes patterns (AI)                           │
       │                      ▼                                                  │
       │             ┌─────────────────┐                              ┌──────────▼──────┐
       │             │ Behavior        │                              │ Task Presenter  │
       │             │ Analyzer (AI)   │                              │                 │
       │             │                 │                              │ Outputs:        │
       │             │ - Detects meta  │                              │ • Tasks processed│
       │             │   patterns      │                              │   from email    │
       │             │ - Generates     │                              │ • Corrections   │
       │             │   observations  │                              │   confirmed     │
       │             └────────┬────────┘                              └─────────────────┘
       │                      │ stores                                        │
       │                      ▼                                               │
       │             ┌─────────────────┐                                      │
       │             │ User Behaviour  │                                      │
       │             │      DB         │                                      │
       │             └─────────────────┘                                      │
       │                                                                      │
       └──────────────────────────────────────────────────────────────────────┘
                              (email response)
```

### 2.3 Email Input Types

| Content Type | Example | Purpose |
|--------------|---------|---------|
| **New Activities** | "Today I completed X, Y, Z" | Log completed tasks |
| **Corrections** | "Task was due Wednesday not Thursday" | Update existing tasks |
| **Context Replies** | Response to system's clarification request | Fill in missing information |

### 2.4 Component Details

#### Task Extractor
| Responsibility | Description |
|----------------|-------------|
| **Classify Intent** | Determine if email contains new activities, corrections, or context replies |
| **Extract Tasks** | Identify individual tasks from free-form text or lists |
| **Transform to JSON** | Use AI to convert extracted tasks into uniform JSON schema (see below) |
| **Check Completeness** | Verify tasks have required info (due date, priority, etc.) |
| **Request Context** | Send email back to user if information is missing |
| **Trigger Behavior Analysis** | Pass interaction data to Behavior Analyzer for pattern detection |

**Task JSON Schema:**
```json
{
  "task_id": "uuid",
  "title": "string (required)",
  "description": "string (optional)",
  "due_date": "ISO 8601 date (optional)",
  "priority": "high | medium | low (default: medium)",
  "status": "pending | in_progress | completed",
  "classification": {
    "category": "string (optional) - e.g., Admin, Meetings, Development",
    "project": "string (optional) - e.g., Q4 Budget Review, Website Redesign",
    "client": "string (optional) - e.g., Acme Corp, Beta Industries"
  },
  "classification_source": "header | explicit | inferred | unknown",
  "estimated_hours": "number (optional)",
  "source_email_id": "string",
  "extracted_at": "ISO 8601 timestamp",
  "raw_text": "original text from email"
}
```

**Classification Detection:**
Users typically specify classification as headers before listing tasks:

```
Email example:
"Acme Corp:
- Had standup with team
- Reviewed contract proposal

Beta Industries:  
- Sent follow-up emails
- Scheduled demo"
```

The system should:
1. **Detect headers** - Look for patterns like "Client:", "Project:", or standalone names followed by colon/tasks
2. **Apply to tasks below** - All tasks under a header inherit that classification
3. **Ask if unclear** - If tasks have no classification and can't be inferred, add to `missing_context`

**Example Transformation:**

*User writes:*
> "Tomorrow I need to finish the quarterly report (urgent!) and also send follow-up emails to the 3 clients from last week's meeting"

*AI transforms to:*
```json
[
  {
    "title": "Finish quarterly report",
    "due_date": "2025-12-24",
    "priority": "high",
    "status": "pending",
    "raw_text": "finish the quarterly report (urgent!)"
  },
  {
    "title": "Send follow-up emails to clients",
    "description": "3 clients from last week's meeting",
    "due_date": "2025-12-24",
    "priority": "medium",
    "status": "pending",
    "raw_text": "send follow-up emails to the 3 clients from last week's meeting"
  }
]
```

#### Behavior Analyzer (AI-Powered)
| Responsibility | Description |
|----------------|-------------|
| **Analyze Patterns** | Use AI to detect meta-patterns in user behavior over time |
| **Generate Observations** | Create human-readable observations (e.g., "User tends to submit tasks without due dates") |
| **Track Frequency** | Monitor how often patterns occur |
| **Store to DB** | Persist observations to User Behaviour DB |

**Example Meta-Behavior Observations:**
- "User frequently submits tasks without context - consider prompting for details"
- "User tends to forget to log completed tasks - send reminders"
- "User often corrects due dates within 24 hours - verify dates at intake"
- "User is highly responsive to context requests - can ask follow-ups"

#### Task Processor
| Responsibility | Description |
|----------------|-------------|
| **Read Existing Tasks** | Fetch current tasks from Task DB for comparison |
| **Hybrid Comparison** | Use embeddings + AI to find similar tasks (existing implementation) |
| **Classify Match Type** | AI determines: status change, progress update, recurring activity, correction, or new task |
| **Apply Status Changes** | Update task status (e.g., in_progress → completed) |
| **Log Progress Updates** | Add notes or update progress on multi-session tasks |
| **Log Recurring** | Create new occurrence linked to recurring pattern |
| **Apply Corrections** | Process correction requests from user (fix wrong values) |
| **Update Database** | Write final state to Task DB |

**Hybrid Comparison Logic (Existing):**
The spike will leverage the existing `src/core/task_similarity.py` which implements:
1. Chroma embeddings for fast top-k candidate retrieval
2. AI (Gemini) for semantic analysis of candidates
3. Returns confidence score and matched task

**Expanding ChromaDB Usage:**
The existing `SimpleChromaEmbeddingManager` can be extended for:
- **Intent classification cache** - Store email→intent mappings, reuse for similar emails
- **Query pattern cache** - Cache common queries for Process 3
- **Recurring pattern detection** - Cluster similar tasks to detect patterns
- **User behavior clustering** - Group similar user behaviors

**Match Classification:**
```
┌─────────────────────────────────────────────────────────────────┐
│                    MATCH CLASSIFICATION                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Similar task found?                                            │
│         │                                                        │
│    ┌────┴────┐                                                  │
│    │         │                                                  │
│   NO        YES                                                 │
│    │         │                                                  │
│    ▼         ▼                                                  │
│  CREATE    What type of match?                                  │
│   NEW      (AI determines from context)                         │
│  TASK            │                                              │
│         ┌────────┼────────────┬────────────┐                    │
│         │        │            │            │                    │
│         ▼        ▼            ▼            ▼                    │
│     STATUS   PROGRESS    RECURRING    CORRECTION                │
│     CHANGE   UPDATE      ACTIVITY     (same task,               │
│     (done,   (more work  (daily       wrong values)             │
│     blocked) on same     standup,                               │
│         │    task)       weekly sync) │                         │
│         │        │            │        │                        │
│         ▼        ▼            ▼        ▼                        │
│     UPDATE   ADD NOTE/   LOG NEW    UPDATE                      │
│     STATUS   UPDATE      OCCURRENCE FIELDS                      │
│              PROGRESS    (streak+1)                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Match Type Definitions:**

| Match Type | Description | Example | Action |
|------------|-------------|---------|--------|
| **Status Change** | User indicates task state changed | "Finished the report" (was in_progress) | Update status field (→ completed, blocked, cancelled) |
| **Progress Update** | Continued work on multi-session task | "Worked more on the big report" | Add progress note, update `updated_at`, optionally log hours |
| **Recurring Activity** | Discrete activity done repeatedly | "Had daily standup", "Weekly team sync" | Log new occurrence, update streak, link to pattern |
| **Correction** | Same task but with wrong values | "Actually it's due Thursday not Wednesday" | Update the incorrect field(s) |

**AI Classification Signals:**

| Match Type | Language Signals | Context Signals |
|------------|------------------|-----------------|
| **Status Change** | "finished", "completed", "done with", "blocked on", "cancelled" | Task exists with status ≠ completed |
| **Progress Update** | "worked on", "continued", "made progress", "more work on" | Task exists with status = in_progress |
| **Recurring Activity** | "daily", "weekly", "standup", "sync", "regular" | Similar tasks logged on different days, discrete activities |
| **Correction** | "actually", "should be", "correction", "wrong", "meant to say" | Explicit correction language |

#### Recurring Patterns: Assume → Announce → Correct

Since this is email communication (not a UI), we minimize back-and-forth by:
1. **Assuming** patterns automatically based on detection
2. **Announcing** assumptions in Task Presenter emails
3. **Correcting** only if user replies with disagreement

**Detection Flow:**
```
┌─────────────────────────────────────────────────────────────────┐
│  AUTOMATIC DETECTION (no user interaction required)             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ 1. Language     │  │ 2. Pattern      │  │ 3. User         │ │
│  │    Signals      │  │    Analysis     │  │    Explicit     │ │
│  │                 │  │                 │  │                 │ │
│  │ "daily",        │  │ 3+ similar      │  │ User says       │ │
│  │ "weekly",       │  │ tasks on        │  │ "I do this      │ │
│  │ "standup",      │  │ different days  │  │ every day"      │ │
│  │ "sync"          │  │                 │  │                 │ │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘ │
│           │                    │                    │           │
│           └────────────────────┴────────────────────┘           │
│                                │                                │
│                                ▼                                │
│                   AUTO-CREATE PATTERN                           │
│                   (status: "assumed")                           │
│                                │                                │
│                                ▼                                │
│              ANNOUNCE IN TASK PRESENTER EMAIL                   │
└─────────────────────────────────────────────────────────────────┘
```

**Detection Priority:**
| Priority | Method | Example | Confidence |
|----------|--------|---------|------------|
| 1 | User explicit | "I do this every day" | Highest |
| 2 | Language signals | "Weekly report", "daily standup" | High |
| 3 | Pattern analysis | 3+ similar on different days | Medium |

**New Table: `recurring_patterns`**
| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Primary key |
| `user_id` | UUID | Owner |
| `title` | String | Activity name (e.g., "Daily standup with Acme") |
| `expected_frequency` | String | "daily", "weekly:mon,wed,fri", "monthly" |
| `detection_method` | String | "language", "pattern_analysis", "user_explicit" |
| `status` | String | "assumed", "confirmed", "rejected" |
| `announced_at` | DateTime | When user was notified |
| `last_logged_at` | DateTime | Most recent occurrence |
| `streak_count` | Integer | Consecutive occurrences logged |
| `created_at` | DateTime | When pattern was detected |

**Status Values:**
- `assumed` - System detected and announced, user hasn't responded
- `confirmed` - User explicitly confirmed or corrected frequency
- `rejected` - User said "not recurring", won't track anymore

**Announcement in Task Presenter:**
```
🔄 NEW RECURRING PATTERNS DETECTED:
• "Daily standup with Acme" → marked as DAILY
  (logged 4 times this week on different days)
• "Weekly report to leadership" → marked as WEEKLY
  (detected from language)

↩️ Reply to correct: "Daily standup is not recurring"
   or "Weekly report is actually monthly"
```

**Correction Handling (via Process 1):**
| User Says | Action |
|-----------|--------|
| "X is not recurring" | Set status='rejected', delete pattern |
| "X is weekly, not daily" | Update frequency, set status='confirmed' |
| "Stop tracking X" | Set status='rejected' |

**Missing Activity Check (for Process 2):**
```
⚠️ RECURRING ACTIVITIES NOT LOGGED TODAY:
• "Daily standup with Acme" - usually logged by now
  Reply "Done" if completed, or "Skip today" to note the miss

🔥 STREAK UPDATE:
• "Weekly report" - 8 week streak! Keep it up!
```

#### Task Presenter
| Responsibility | Description |
|----------------|-------------|
| **List Processed Tasks** | Show the tasks extracted and saved from the current email |
| **Confirm Corrections** | Acknowledge any corrections that were applied |
| **Send Response** | Email the response back to user |

### 2.5 Email Response Strategy

| Scenario | Email Type | Reason |
|----------|-----------|--------|
| Tasks processed from email | Reply to original | Confirms what was captured from their email |
| Confirmation of corrections | Reply to original | Context matters, closes the loop |
| Asking for more context | Reply to original | User needs to see what was unclear |

---

## 3. Process 2: Prioritization & Insights Generation

### 3.1 Overview

This process generates daily task priorities and personalized insights by analyzing the user's task database against organizational best practices and the user's own behavioral patterns.

### 3.2 Triggers

**Scheduled Daily Job** (e.g., Cloud Scheduler at 6:00 PM)
- Runs automatically for all active users
- Generates priorities for the next day
- Sends combined email with priorities, insights, and outstanding tasks list

**On-Demand Request** (via Email Router)
- User sends email requesting status (e.g., "Send me my status report", "What are my priorities?")
- Email Router classifies as "Status Request" and routes to Process 2
- Generates and sends combined email immediately for that user only

### 3.3 Flow Diagram

```
┌──────────────┐         ┌───────────────────┐
│   Task DB    │────────▶│  Task Prioritizer │──────────────┐
│              │ reads   │                   │              │
│              │         │ Creates prioritized              │ prioritized
│              │         │ list for next day │              │ task list
│              │         └───────────────────┘              │
│              │                                            │
│              │         ┌───────────────────┐              ▼
│              │────────▶│ Insight Generator │      ┌─────────────────┐
└──────────────┘ reads   │                   │      │  Combined Email │    ┌──────────┐
                         │ - Analyzes backlog│      │     to User     │◀──▶│   USER   │
┌──────────────┐         │ - Considers user  │      │                 │    └──────────┘
│Best Practices│────────▶│   behavior patterns      │ Contains:       │      requests
│      DB      │ reads   │ - Creates advice  │      │ • Priorities    │      on-demand
│              │         │ - What to improve │      │ • Insights      │
└──────────────┘         │                   │      │ • Outstanding   │
                  reads  │                   │      │   tasks list    │
┌──────────────┐────────▶│                   │      └───────▲─────────┘
│User Behaviour│         └─────────┬─────────┘              │
│      DB      │                   │ stores                 │ insights
│              │                   ▼                        │
└──────────────┘         ┌───────────────────┐              │
                         │ Insights Database │──────────────┘
                         │                   │
                         │ (all generated    │
                         │  insights logged) │
                         └───────────────────┘
```

### 3.4 Component Details

#### Task Prioritizer
| Responsibility | Description |
|----------------|-------------|
| **Read Task Backlog** | Fetch all open/pending tasks from Task DB |
| **Apply Prioritization Logic** | Consider due dates, priority flags, dependencies |
| **Generate Priority List** | Create ordered list of tasks for next day |

#### Insight Generator
| Responsibility | Description |
|----------------|-------------|
| **Analyze User Backlog** | Read current state from Task DB |
| **Compare to Best Practices** | Check patterns against Best Practices DB |
| **Consider User Patterns** | Factor in behavioral observations from User Behaviour DB (e.g., "often misses deadlines", "forgets to log hours") |
| **Generate Personalized Advice** | Create actionable recommendations tailored to user's habits |
| **Store Insights** | Log all generated insights to Insights Database |

### 3.5 Example Insights

- "You have 5 overdue tasks - consider addressing these first"
- "You've been logging tasks without due dates - try adding deadlines"
- "Your high-priority queue is growing - consider delegating"
- "Great job completing 12 tasks this week - above your average!"

### 3.6 Combined Email Output

The email contains three distinct sections: suggested priorities, personalized insights, and a full list of outstanding tasks:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 SUGGESTED PRIORITIES FOR TOMORROW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Complete quarterly report (High Priority, Due: Tomorrow)
2. Send client follow-up emails (Medium Priority)
3. Review team submissions (Medium Priority)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 INSIGHTS & RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• You have 3 tasks overdue by more than a week - consider re-prioritizing
• Great job completing 12 tasks this week - above your average!
• Suggestion: Your "Admin" category is growing - block time for admin work

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 ALL OUTSTANDING TASKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OVERDUE (3):
  • Budget review (Due: Dec 20) - High Priority
  • Update documentation (Due: Dec 22) - Low Priority
  • Client proposal draft (Due: Dec 23) - Medium Priority

DUE THIS WEEK (5):
  • Complete quarterly report (Due: Dec 28) - High Priority
  • Send client follow-up emails (Due: Dec 28) - Medium Priority
  • Review team submissions (Due: Dec 29) - Medium Priority
  • Prepare meeting agenda (Due: Dec 30) - Low Priority
  • Submit expense report (Due: Dec 31) - Medium Priority

UPCOMING (4):
  • Q1 planning session prep (Due: Jan 5) - High Priority
  • Annual review self-assessment (Due: Jan 10) - Medium Priority
  • ...
```

---

## 4. Process 3: Query/Analytics System

### 4.1 Overview

This process enables users and managers to query data across all databases with appropriate access controls and privacy safeguards.

### 4.2 Interface

| Phase | Interface | Description |
|-------|-----------|-------------|
| **MVP** | Email | Users send queries via email (e.g., "What tasks do I have due this week?") |
| **Future** | Web UI | Dedicated query interface with chat-like experience |
| **Future** | Slack/Teams | Query via messaging platforms |

### 4.3 Flow Diagram

```
                                                      ┌─────────────────────────────────┐
                                                      │        DATA ACCESS LAYER        │
                                                      ├─────────────────┬───────────────┤
                                                      │   User's Own    │  Firm Total   │
                                                      │   (Full Access) │(Manager Only) │
                                                      ├─────────────────┼───────────────┤
┌──────────┐                                          │                 │               │
│   USER   │──┐                                   ┌──▶│ UB-DB (personal)│ Manager only  │
└──────────┘  │                                   │   │                 │               │
 Own data    │      ┌───────────────────────┐    │   ├─────────────────┼───────────────┤
 only        ├─────▶│     Query Router      │────┼──▶│Task DB (personal│ Manager only  │
              │      │                       │    │   │                 │               │
              │      │ • Breakdown           │    │   ├─────────────────┼───────────────┤
              │      │   (decomposes query)  │    │   │Ins DB (personal)│ Manager only  │
┌──────────┐  │      │                       │    │   │                 │               │
│ MANAGER  │──┘      │ • Coordinator         │    │   ├─────────────────┼───────────────┤
│          │         │   (routes to sources) │    └──▶│  Best Practices │Best Practices │
│ Can also │         │                       │        │                 │               │
│ access:  │         │ • Synthesizer         │        └─────────────────┴───────────────┘
│ • Any    │         │   (combines results)  │
│   user   │         └───────────────────────┘
│ • Firm   │
│   totals │
└──────────┘
```

### 4.4 Access Control Matrix

| Requester | Own Data | Any User | Firm Total |
|-----------|----------|----------|------------|
| **User** | ✅ Full | ❌ No | ❌ No |
| **Manager** | ✅ Full | ✅ Full | ✅ Full |

### 4.5 Query Router Components

| Component | Responsibility |
|-----------|----------------|
| **Breakdown** | Decomposes complex queries into sub-queries |
| **Coordinator** | Routes sub-queries to appropriate data sources |
| **Synthesizer** | Combines results from multiple sources into coherent response |

### 4.6 Query Examples

| Who | Query | Data Source | Response Type |
|-----|-------|-------------|---------------|
| User | "What tasks did I complete this week?" | Task DB (personal) | Full detail |
| User | "What's my completion rate this month?" | Task DB (personal) | Full detail |
| User | "How does the firm compare?" | ❌ Denied | Access denied message |
| Manager | "Show me Alex's overdue tasks" | Task DB (Alex's personal) | Full detail |
| Manager | "What's the firm's completion rate?" | Task DB (firm total) | Full detail |
| Manager | "How many tasks does Sarah have pending?" | Task DB (Sarah's personal) | Full detail |

### 4.7 Access Enforcement

The Query Router enforces strict access control:
- **Users** can only query their own data - any request for other users or firm data is denied
- **Managers** have full access to any user's data and firm-wide aggregates
- Firm data is simply an aggregation of all users' data

**Denied Query Response:**
When a user attempts to access data they don't have permission for:
> "I can only show you information about your own tasks. If you need firm-wide data, please contact a manager."

### 4.8 MVP Scope Note

This process is intentionally **scoped as a foundation** for the spike. Future enhancements may include:
- More granular role-based access control
- Team/department-level views
- Audit logging of sensitive queries
- Permission management UI

---

## 5. Data Model

### 5.1 Databases Overview

| Database | Purpose | Key Data |
|----------|---------|----------|
| **Task DB** | Canonical store for all tasks | Tasks, status, priority, due dates, metadata |
| **User Behaviour DB** | Patterns and habits about user interactions | Behavioral observations, trends, flags |
| **Best Practices DB** | Organizational standards and guidelines | Rules, templates, benchmarks |
| **Insights DB** | Historical record of generated insights | All insights with timestamps, context |

### 5.2 Task DB Schema

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Primary key |
| `user_id` | UUID | Owner of the task |
| `title` | String | Task description |
| `description` | String | Additional details |
| `status` | Enum | pending, in_progress, completed, cancelled |
| `priority` | Enum | low, medium, high, urgent |
| `due_date` | DateTime | When task is due |
| `category` | String | Task category (Admin, Meetings, Development, etc.) |
| `project` | String | Project name (Q4 Budget Review, Website Redesign, etc.) |
| `client` | String | Client name (Acme Corp, Beta Industries, etc.) |
| `classification_source` | Enum | header, explicit, inferred, unknown |
| `created_at` | DateTime | When task was created |
| `updated_at` | DateTime | Last modification |
| `source_email_id` | String | Reference to originating email |
| `metadata` | JSON | Additional flexible data |

**Classification Hierarchy:**
```
Client (highest level)
  └── Project
       └── Category (task type)
            └── Task
```

**Example:**
```
Acme Corp (client)
  └── Q4 Budget Review (project)
       └── Admin (category)
            └── "Send budget summary to CFO" (task)
```

### 5.3 User Behaviour DB Schema

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Primary key |
| `user_id` | UUID | User being tracked |
| `pattern_type` | String | Type of behavior (e.g., "missing_context", "late_logging") |
| `observation` | String | Description of the pattern |
| `frequency` | Integer | How often observed |
| `first_seen` | DateTime | When first noticed |
| `last_seen` | DateTime | Most recent occurrence |
| `is_active` | Boolean | Whether pattern is still relevant |

### 5.4 Best Practices DB Schema

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Primary key |
| `category` | String | Category (productivity, prioritization, etc.) |
| `rule` | String | The best practice rule |
| `description` | String | Detailed explanation |
| `benchmark` | JSON | Quantitative benchmarks if applicable |
| `is_active` | Boolean | Whether currently enforced |

### 5.5 Insights DB Schema

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Primary key |
| `user_id` | UUID | User who received the insight |
| `insight_type` | String | Category of insight |
| `content` | String | The insight text |
| `context` | JSON | Data that drove the insight |
| `created_at` | DateTime | When generated |
| `was_actioned` | Boolean | Whether user acted on it (future) |

---

## 6. Technology Stack

| Layer | Technology | Notes |
|-------|------------|-------|
| **AI/LLM** | Google Gemini 3.0 (Vertex AI) | Task extraction, insight generation, query understanding |
| **Orchestration** | LangGraph | Agent workflow management |
| **API** | FastAPI or Next.js API Routes | HTTP endpoints |
| **Database** | PostgreSQL (Cloud SQL) | All four databases |
| **Email** | Gmail API / IMAP+SMTP | Input and output channel |
| **Deployment** | Google Cloud Run | Containerized services |
| **Secrets** | Google Secret Manager | API keys, credentials |
| **Monitoring** | Cloud Logging & Monitoring | Observability |

---

## 7. Agent Architecture

### 7.1 LangGraph Node Structure

Each process is implemented as a LangGraph workflow with specialized nodes:

#### Process 1 Nodes (Optimized)

**Consolidated LLM Calls:** To minimize latency and cost, we combine multiple extraction steps into single LLM calls.

| Node | Input | Output | LLM Call? |
|------|-------|--------|-----------|
| `EmailIntakeNode` | Raw email | Normalized email object | No |
| `UnifiedExtractionNode` | Normalized email | Intent + JSON tasks + context flags + entities | **Yes (1 call)** |
| `BehaviorAnalyzerNode` | User interaction history | Meta-behavior observations | **Yes** |
| `BehaviorPersistNode` | Observations | Stored to User Behaviour DB | No |
| `TaskComparisonNode` | JSON tasks + DB tasks | Diff (add/update list) | No |
| `TaskPersistNode` | Diff | Updated Task DB | No |
| `PresenterNode` | Task DB state | Email response | Yes |

**UnifiedExtractionNode** combines what was previously 3-4 separate calls:
- Intent classification (activity/correction/context/query)
- Task extraction from free-form text
- JSON transformation to uniform schema
- Missing context detection
- Entity extraction (dates, people, projects)
- Recurring signal detection

**Single Prompt Output:**
```json
{
  "intent": "activity",
  "tasks": [{"title": "...", "due_date": "...", "priority": "..."}],
  "missing_context": [],
  "is_recurring_signal": false,
  "detected_entities": {"dates": [], "people": [], "projects": []}
}
```

**Performance Comparison:**
| Metric | Before (4 calls) | After (1 call) | Improvement |
|--------|------------------|----------------|-------------|
| LLM calls | 4 | 1 | 75% fewer |
| Latency | ~8s | ~2-3s | 60% faster |
| Token cost | 4x | 1x | 75% cheaper |

#### Process 2 Nodes
| Node | Input | Output | LLM Call? |
|------|-------|--------|-----------|
| `TaskFetchNode` | User ID | Current tasks | No |
| `PrioritizerNode` | Tasks | Prioritized list | Yes |
| `BestPracticesFetchNode` | Category | Relevant practices | No |
| `UserBehaviourFetchNode` | User ID | User behavior patterns | No |
| `InsightGeneratorNode` | Tasks + Practices + User Behaviour | Insights | Yes |
| `InsightPersistNode` | Insights | Stored to Insights DB | No |
| `EmailComposerNode` | Priorities + Insights + All Tasks | Combined email | No |

#### Process 3 Nodes (Optimized)

| Node | Input | Output | LLM Call? |
|------|-------|--------|-----------|
| `QueryParserNode` | Natural language query | Parsed query + sub-queries | **Yes (1 call)** |
| `AccessControlNode` | User + Query | Allowed data scopes | No |
| `CoordinatorNode` | Sub-queries + Scopes | Routed queries | No |
| `DataFetchNode` | Routed queries | Raw results | No |
| `ResponseGeneratorNode` | Raw results + Privacy rules | Summarized + synthesized response | **Yes (1 call)** |

**Consolidation:** Reduced from 4 LLM calls to 2:
- `QueryIntakeNode` + `BreakdownNode` → `QueryParserNode`
- `SummarizerNode` + `SynthesizerNode` → `ResponseGeneratorNode`

---

## 8. Prompt Specifications

This section defines the prompt structure for each LLM-calling node. These are the "contracts" that the Gemini model must fulfill.

### 8.1 UnifiedExtractionNode (Process 1)

**Purpose:** Single call to classify intent, extract tasks, transform to JSON, detect missing context, and identify entities.

**Temperature:** 0.3 (conservative for structured output)

**Input:**
```
Email from: {sender_email}
Subject: {subject}
Body:
{email_body}
```

**Prompt Structure:**
```
You are an AI task management assistant. Analyze this email and extract all relevant information.

EMAIL:
From: {sender_email}
Subject: {subject}
Body: {email_body}

INSTRUCTIONS:
1. Classify the PRIMARY intent of this email
2. Extract all tasks mentioned (activities, completions, todos)
3. Convert each task to structured JSON
4. Identify any missing information that needs clarification
5. Detect entities (dates, people, projects)
6. Check for recurring activity signals ("daily", "weekly", "standup", etc.)

RETURN THIS EXACT JSON STRUCTURE:
{
  "intent": "activity" | "correction" | "context_reply" | "query",
  "tasks": [
    {
      "title": "string (required)",
      "description": "string or null",
      "due_date": "YYYY-MM-DD or null",
      "priority": "high" | "medium" | "low",
      "status": "pending" | "in_progress" | "completed",
      "match_type_hint": "new_task" | "status_change" | "progress_update" | "recurring_activity",
      "classification": {
        "category": "Admin | Meetings | Development | etc or null",
        "project": "project name or null",
        "client": "client name or null"
      },
      "classification_source": "header" | "explicit" | "inferred" | "unknown",
      "raw_text": "original text from email"
    }
  ],
  "status_updates": [
    {
      "task_hint": "description of task to update",
      "new_status": "completed" | "blocked" | "cancelled",
      "reason": "why status changed (optional)"
    }
  ],
  "progress_updates": [
    {
      "task_hint": "description of task being worked on",
      "progress_note": "what was done",
      "hours_worked": "number or null"
    }
  ],
  "corrections": [
    {
      "original_task_hint": "what task to correct",
      "field_to_update": "due_date" | "priority" | "status" | "title" | "client" | "project" | "category",
      "new_value": "the corrected value"
    }
  ],
  "missing_context": [
    "Question to ask user if info is missing"
  ],
  "detected_headers": [
    {
      "header_text": "the header as written",
      "header_type": "client" | "project" | "category" | "unknown",
      "applies_to_tasks": [0, 1, 2]
    }
  ],
  "detected_entities": {
    "dates": ["list of dates mentioned"],
    "people": ["list of people mentioned"],
    "clients": ["list of clients mentioned"],
    "projects": ["list of projects mentioned"]
  },
  "is_recurring_signal": true | false,
  "recurring_frequency_hint": "daily" | "weekly" | "monthly" | null
}

CLASSIFICATION RULES:
- Look for HEADERS before groups of tasks (e.g., "Acme Corp:", "Project Alpha:", "Admin tasks:")
- Headers are usually followed by a colon or newline, then indented/bulleted tasks
- Apply the header's classification to ALL tasks listed under it
- classification_source values:
  - "header": Task inherited classification from a header above it
  - "explicit": User explicitly stated classification in the task text
  - "inferred": AI inferred from context (client name in task, project keywords)
  - "unknown": No classification could be determined
- If classification is "unknown" AND the task seems important, add a question to missing_context

MATCH TYPE RULES:
- "status_change": User says "finished", "completed", "done with", "blocked on", "cancelled"
- "progress_update": User says "worked on", "continued", "made progress on", "more work on"
- "recurring_activity": Discrete activities with signals like "daily", "weekly", "standup", "sync"
- "new_task": No indication this relates to an existing task
- When user mentions completing a task, add to BOTH tasks array (with status=completed) AND status_updates array
- When user mentions continued work, add to BOTH tasks array AND progress_updates array

GENERAL RULES:
- If intent is "query", tasks array should be empty
- Always preserve the raw_text for each task
- Infer priority from language: "urgent", "ASAP", "!!!" = high
- Infer dates: "tomorrow" = next day, "next week" = 7 days, etc.
- If unsure about a field, use null
```

**Example Input:**
```
From: john@company.com
Subject: Today's update
Body: 

Acme Corp:
- Had my daily standup with the team
- Reviewed the Q4 contract proposal
- Sent invoice to accounting

Beta Industries:
- Finished the quarterly report (finally!)
- Tomorrow I need to send 5 follow-up emails - this is urgent!

Also, I said the budget review was due Wednesday but it's actually Thursday.
```

**Example Output:**
```json
{
  "intent": "activity",
  "tasks": [
    {
      "title": "Daily standup with team",
      "description": null,
      "due_date": "2025-12-23",
      "priority": "medium",
      "status": "completed",
      "match_type_hint": "recurring_activity",
      "classification": {
        "category": "Meetings",
        "project": null,
        "client": "Acme Corp"
      },
      "classification_source": "header",
      "raw_text": "Had my daily standup with the team"
    },
    {
      "title": "Reviewed Q4 contract proposal",
      "description": null,
      "due_date": null,
      "priority": "medium",
      "status": "completed",
      "match_type_hint": "new_task",
      "classification": {
        "category": null,
        "project": "Q4 Contract",
        "client": "Acme Corp"
      },
      "classification_source": "header",
      "raw_text": "Reviewed the Q4 contract proposal"
    },
    {
      "title": "Sent invoice to accounting",
      "description": null,
      "due_date": null,
      "priority": "medium",
      "status": "completed",
      "match_type_hint": "new_task",
      "classification": {
        "category": "Admin",
        "project": null,
        "client": "Acme Corp"
      },
      "classification_source": "header",
      "raw_text": "Sent invoice to accounting"
    },
    {
      "title": "Quarterly report",
      "description": null,
      "due_date": null,
      "priority": "medium",
      "status": "completed",
      "match_type_hint": "status_change",
      "classification": {
        "category": null,
        "project": null,
        "client": "Beta Industries"
      },
      "classification_source": "header",
      "raw_text": "Finished the quarterly report (finally!)"
    },
    {
      "title": "Send follow-up emails",
      "description": "5 emails",
      "due_date": "2025-12-24",
      "priority": "high",
      "status": "pending",
      "match_type_hint": "new_task",
      "classification": {
        "category": null,
        "project": null,
        "client": "Beta Industries"
      },
      "classification_source": "header",
      "raw_text": "send 5 follow-up emails - this is urgent!"
    }
  ],
  "status_updates": [
    {
      "task_hint": "quarterly report",
      "new_status": "completed",
      "reason": "User said 'finished'"
    }
  ],
  "progress_updates": [],
  "corrections": [
    {
      "original_task_hint": "budget review",
      "field_to_update": "due_date",
      "new_value": "Thursday"
    }
  ],
  "missing_context": [],
  "detected_headers": [
    {
      "header_text": "Acme Corp:",
      "header_type": "client",
      "applies_to_tasks": [0, 1, 2]
    },
    {
      "header_text": "Beta Industries:",
      "header_type": "client",
      "applies_to_tasks": [3, 4]
    }
  ],
  "detected_entities": {
    "dates": ["today", "tomorrow", "Wednesday", "Thursday"],
    "people": [],
    "clients": ["Acme Corp", "Beta Industries"],
    "projects": ["Q4 Contract"]
  },
  "is_recurring_signal": true,
  "recurring_frequency_hint": "daily"
}
```

**Example with Missing Classification:**
```
Input: "Today I sent some emails and worked on the presentation."

Output missing_context would include:
["Which client or project are these tasks for?"]
```

---

### 8.2 BehaviorAnalyzerNode (Process 1)

**Purpose:** Analyze user interaction patterns and generate meta-observations.

**Temperature:** 0.5 (moderate creativity for insights)

**Input:**
```
User: {user_email}
Recent interactions: {last_N_interactions}
Current extraction: {current_extraction_result}
Existing patterns: {existing_patterns_from_db}
```

**Prompt Structure:**
```
You are analyzing user behavior patterns for a task management system.

USER: {user_email}
RECENT INTERACTIONS (last 20):
{interactions_summary}

CURRENT EXTRACTION:
{current_extraction}

EXISTING PATTERNS WE'VE DETECTED:
{existing_patterns}

ANALYZE and identify NEW or REINFORCED patterns. Consider:
- Do they frequently omit due dates?
- Do they often need context clarification?
- Do they correct tasks shortly after submitting?
- Are they consistent with recurring activities?
- Do they use vague language often?

RETURN JSON:
{
  "new_patterns": [
    {
      "pattern_type": "missing_due_dates" | "needs_context" | "frequent_corrections" | etc,
      "observation": "Human-readable observation",
      "confidence": 0.0-1.0,
      "evidence": "What led to this conclusion"
    }
  ],
  "reinforced_patterns": [
    {
      "pattern_id": "existing pattern ID",
      "additional_evidence": "new supporting evidence"
    }
  ]
}
```

---

### 8.3 PrioritizerNode (Process 2)

**Purpose:** Generate prioritized task list for the next day.

**Temperature:** 0.3 (structured output)

**Prompt Structure:**
```
You are a productivity coach helping prioritize tasks for tomorrow.

USER'S OPEN TASKS:
{tasks_json}

TODAY'S DATE: {today}
USER'S PATTERNS: {relevant_behavior_patterns}

Generate a prioritized list for tomorrow. Consider:
1. Due dates (overdue first, then tomorrow, then this week)
2. Priority flags (high > medium > low)
3. User's patterns (if they often miss deadlines, emphasize those)
4. Realistic daily capacity (5-7 focus tasks)

RETURN JSON:
{
  "priority_tasks": [
    {
      "task_id": "uuid",
      "title": "task title",
      "reason": "Why this is prioritized",
      "suggested_time": "morning" | "afternoon" | "anytime"
    }
  ],
  "deferred_tasks": [
    {
      "task_id": "uuid",
      "reason": "Why this can wait"
    }
  ],
  "warnings": [
    "Any urgent warnings (overdue, capacity exceeded, etc.)"
  ]
}
```

---

### 8.4 InsightGeneratorNode (Process 2)

**Purpose:** Generate personalized productivity insights.

**Temperature:** 0.6 (creative but grounded)

**Prompt Structure:**
```
You are a supportive productivity coach generating insights.

USER'S TASK DATA:
{task_summary}

BEST PRACTICES TO COMPARE AGAINST:
{best_practices}

USER'S BEHAVIOR PATTERNS:
{behavior_patterns}

MISSED RECURRING ACTIVITIES:
{missed_recurring}

Generate 2-4 actionable insights. Be:
- Supportive, not critical
- Specific, not generic
- Actionable, not just observational
- Personalized based on the user's behavior patterns (e.g., if they often miss deadlines, emphasize time management; if they forget to log tasks, remind them)
- Aware of their strengths and weaknesses from past observations

RETURN JSON:
{
  "insights": [
    {
      "type": "celebration" | "improvement" | "reminder" | "warning",
      "message": "The insight message",
      "action": "Specific action they can take",
      "related_practice_id": "uuid or null"
    }
  ]
}
```

---

### 8.5 QueryParserNode (Process 3)

**Purpose:** Parse natural language query and decompose into sub-queries.

**Temperature:** 0.3 (structured)

**Prompt Structure:**
```
You are parsing a user's query about their task data.

USER: {user_email}
USER ROLE: {role: user | manager}
QUERY: "{query_text}"

Parse this query and decompose if needed.

RETURN JSON:
{
  "query_type": "personal" | "other_user" | "firm" | "aggregate",
  "time_range": {
    "start": "YYYY-MM-DD or null",
    "end": "YYYY-MM-DD or null",
    "relative": "today" | "this_week" | "this_month" | null
  },
  "sub_queries": [
    {
      "target_db": "tasks" | "insights" | "behaviors" | "best_practices",
      "filter": "description of what to fetch",
      "aggregation": "count" | "list" | "average" | null
    }
  ],
  "requires_summarization": true | false,
  "original_intent": "What the user wants to know"
}
```

---

### 8.6 ResponseGeneratorNode (Process 3)

**Purpose:** Synthesize data into natural language response with privacy rules.

**Temperature:** 0.5 (conversational)

**Prompt Structure:**
```
Generate a response to the user's query.

ORIGINAL QUERY: "{query}"
USER ROLE: {role}
DATA RESULTS:
{raw_results}

ACCESS RULES:
- If role is "user": ONLY return their own data. Deny any requests for other users or firm data.
- If role is "manager": Full access to any user's data and firm-wide aggregates.
- For denied requests, respond with: "I can only show you information about your own tasks."

Generate a helpful, conversational response.

RETURN JSON:
{
  "response": "The natural language response",
  "access_level": "personal" | "other_user" | "firm",
  "access_granted": true | false,
  "follow_up_suggestion": "Optional follow-up question" 
}
```

---

### 8.7 Temperature Guidelines

| Temperature | Use Case | Nodes |
|-------------|----------|-------|
| **0.3** | Structured extraction, parsing | UnifiedExtraction, QueryParser |
| **0.5** | Balanced creativity/structure | BehaviorAnalyzer, ResponseGenerator |
| **0.6** | Creative insights | InsightGenerator |

---

### 8.8 Prompt Engineering Best Practices

1. **Always request JSON** - Structured output is easier to parse
2. **Provide examples** - Few-shot prompts improve accuracy
3. **Set boundaries** - Clear rules prevent hallucination
4. **Include context** - User patterns, date, role
5. **Handle edge cases** - What to return if no tasks, no matches, etc.

---

## 9. Integration Points & Future Extensions

| Integration | Description | Priority |
|-------------|-------------|----------|
| **Gmail Intake** | Process incoming emails automatically | MVP |
| **Email Output** | Send summaries, priorities, insights | MVP |
| **Notion Sync** | Bidirectional sync with Notion databases | Post-MVP |
| **Slack Integration** | Query via Slack, receive alerts | Future |
| **Calendar Integration** | Consider calendar when prioritizing | Future |
| **Team Dashboard** | Manager UI for team oversight | Future |

---

## 10. Observability & Resilience

### 10.1 Logging Strategy
- Every LangGraph node logs entry/exit with run ID
- LLM calls log prompt hash, latency, token count
- Email operations log message IDs for traceability

### 10.2 Error Handling
- Failed extractions trigger context requests (not silent failures)
- Database errors trigger retries with exponential backoff
- Unrecoverable errors logged with full state for replay

### 10.3 Metrics
- Task extraction accuracy (manual review sampling)
- End-to-end latency per process
- User engagement with insights (future)

---

## 11. Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Google Cloud Platform                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Cloud Run    │    │ Cloud Run    │    │ Cloud Run    │       │
│  │ Process 1    │    │ Process 2    │    │ Process 3    │       │
│  │ (Intake)     │    │ (Priority)   │    │ (Query)      │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                   │                   │                │
│         └───────────────────┼───────────────────┘                │
│                             │                                    │
│                    ┌────────▼────────┐                          │
│                    │   Cloud SQL     │                          │
│                    │   (PostgreSQL)  │                          │
│                    │                 │                          │
│                    │ • Task DB       │                          │
│                    │ • User Behav DB │                          │
│                    │ • Insights DB   │                          │
│                    │ • Best Prac DB  │                          │
│                    └─────────────────┘                          │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Cloud        │    │ Secret       │    │ Cloud        │       │
│  │ Scheduler    │    │ Manager      │    │ Logging      │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 12. Next Implementation Steps

1. **Set up repository** with process-based folder structure
2. **Implement Process 1** (Task Intake) as MVP core
3. **Create database schemas** for all four databases
4. **Implement Process 2** (Prioritization & Insights)
5. **Implement Process 3** (Query System) - foundation only
6. **Add email integration** (Gmail IMAP/SMTP)
7. **Deploy to Cloud Run** with basic monitoring
8. **Document learnings** and comparison with current system

---

## 13. MVP Operational Considerations

### 13.1 User Authentication (Simple)

For MVP, use a simple `users` table lookup:

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    role VARCHAR(50) DEFAULT 'user',  -- 'user' or 'manager'
    timezone VARCHAR(50) DEFAULT 'UTC',
    created_at TIMESTAMP DEFAULT NOW()
);
```

**Authentication Flow:**
1. Email arrives → extract sender address
2. Lookup in `users` table
3. If found → proceed with user_id
4. If not found → reply with "Please register" or auto-create (configurable)

> **Note:** Full JWT/OAuth authentication is deferred to Phase 1 (see SPIKE_ROADMAP.md)

### 13.2 Error Handling

| Failure | Handling | User Impact |
|---------|----------|-------------|
| **Gemini API down** | Retry 3x with exponential backoff, then queue for later | Delayed response |
| **Gemini returns malformed JSON** | Fallback prompt or flag for manual review | Generic acknowledgment |
| **Task DB unavailable** | Queue email, alert operator, retry on recovery | Delayed processing |
| **Email send fails** | Retry 3x, log failure, alert operator | User doesn't receive response |
| **Unknown error** | Log full state, send generic "processing delayed" response | Graceful degradation |

### 13.3 Cost Controls

| Control | Implementation |
|---------|----------------|
| **Daily budget alert** | Alert if LLM costs exceed $X/day |
| **Rate limiting** | 50 emails/user/day for MVP |
| **Token monitoring** | Log token usage per process for analysis |

### 13.4 Assumptions & Constraints

| Assumption | Notes |
|------------|-------|
| **Single timezone** | All times in UTC or configured timezone; per-user timezones deferred to Phase 1 |
| **English-only** | Gemini handles other languages, but not formally tested |
| **Email attachments** | Ignored for MVP; logged for future processing |
| **Process 3 read-only** | Queries are read-only; mutations (corrections, updates) go through Process 1 |
| **Empty state handling** | P2 with 0 tasks sends "No tasks to prioritize - time to add some!" |

---

## 14. Summary

This architecture defines three interconnected processes:

| Process | Purpose | Key Innovation |
|---------|---------|----------------|
| **1. Task Intake** | Get tasks from email, process, store | User behavior tracking, AI-powered match classification |
| **2. Prioritization & Insights** | Daily/on-demand priorities + improvement advice | Best practices + user behavior patterns, combined output |
| **3. Query System** | Natural language data access | Strict role-based access (users: own data, managers: full access) |

Together, these processes create an intelligent task management agent that not only tracks tasks but actively helps users improve their productivity through personalized insights and easy data access.
