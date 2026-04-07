<wizard-report>
# PostHog post-wizard report

The wizard has completed a deep integration of PostHog analytics into the Humanoid Scientist Brain Flask research assistant. The Python SDK was initialized using the instance-based `Posthog()` constructor with `enable_exception_autocapture=True`, environment variables loaded from `.env`, and `atexit` shutdown registration to ensure no events are lost on exit. A Flask `@app.errorhandler(Exception)` was added for unhandled exception capture. Events were instrumented across all major user-facing API endpoints in `dashboard/app.py`, tracking the full research workflow from query submission through completion, with rich metadata properties on each event (verdict, confidence, domain, evidence counts). No PII is sent in event properties.

| Event | Description | File |
|---|---|---|
| `research_query_submitted` | Fired when a new (non-cached) research query starts processing | `dashboard/app.py` |
| `research_query_completed` | Fired when a research query succeeds, with verdict, confidence, evidence count, and domain properties | `dashboard/app.py` |
| `simulation_run` | Fired when a SWMS world-model simulation is triggered (non-cached) | `dashboard/app.py` |
| `idea_lab_submitted` | Fired when a user submits an idea to the Idea Lab | `dashboard/app.py` |
| `suggestions_requested` | Fired when follow-up suggestions are requested after a research result | `dashboard/app.py` |
| `background_service_triggered` | Fired when the background knowledge-ingestion service is manually triggered | `dashboard/app.py` |
| `api_error` | Fired when a critical API endpoint raises an exception, with endpoint and error_type properties | `dashboard/app.py` |

## Next steps

We've built some insights and a dashboard for you to keep an eye on user behavior, based on the events we just instrumented:

- **Dashboard — Analytics basics**: https://us.posthog.com/project/371457/dashboard/1436698
- **Research Query Volume** (submitted vs completed over time): https://us.posthog.com/project/371457/insights/46No0CBK
- **Feature Engagement Overview** (research, simulations, idea lab, suggestions): https://us.posthog.com/project/371457/insights/fuf0YowI
- **Research Verdict Breakdown** (outcomes by verdict type): https://us.posthog.com/project/371457/insights/cE90CHfh
- **Research-to-Suggestions Funnel** (query → completion → suggestions): https://us.posthog.com/project/371457/insights/GgLwTnOL
- **API Error Rate** (error monitoring): https://us.posthog.com/project/371457/insights/Nn20Eyla

### Agent skill

We've left an agent skill folder in your project. You can use this context for further agent development when using Claude Code. This will help ensure the model provides the most up-to-date approaches for integrating PostHog.

</wizard-report>
