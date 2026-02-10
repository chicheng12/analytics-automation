# ============================================================================
# Analytics Automation — Databricks Dev Workflow
# ============================================================================
# Usage:
#   make sync        → Live-sync to Prod workspace (watches for changes)
#   make sync-llm    → Live-sync to LLM workspace (watches for changes)
#   make run         → Run on Prod workspace
#   make run-llm     → Run on LLM workspace (has openai-agents)
#   make run-local   → Run locally (no Databricks needed)
#   make auth        → Authenticate with Prod workspace
#   make auth-llm    → Authenticate with LLM workspace
# ============================================================================

# -- Configuration (personalize these) ---------------------------------------
DATABRICKS_USER    ?= $(shell git config user.email)
WORKSPACE_PATH     := /Users/$(DATABRICKS_USER)/analytics_automation
NOTEBOOK_PATH      := $(WORKSPACE_PATH)/orchestrator

# Prod workspace (Thumbtack-Prod)
PROD_HOST          := https://1977877856098707.7.gcp.databricks.com
PROD_PROFILE       := 1977877856098707

# LLM workspace (Thumbtack-LLM) — has openai-agents available
LLM_HOST           := https://2939161070393309.9.gcp.databricks.com
LLM_PROFILE        := 2939161070393309
LLM_CLUSTER_ID     := 5319-003104-rubz5jsn

# Set your cluster ID here (find it in Databricks UI → Compute → your cluster)
CLUSTER_ID         ?= $(error Set CLUSTER_ID: make run CLUSTER_ID=xxxx-xxxxxx-xxxxxxxx)

# -- Defaults for run --------------------------------------------------------
DATE             ?=
MONITORS         ?= revenue_funnel
RUN_AI           ?= True

# ============================================================================
# Targets
# ============================================================================

.PHONY: help auth auth-llm sync sync-llm push push-llm pull run run-llm run-local status

help: ## Show this help
	@echo ""
	@echo "Analytics Automation — Dev Workflow"
	@echo "===================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "First-time setup:"
	@echo "  1. make auth"
	@echo "  2. make sync   (in a background terminal)"
	@echo "  3. Edit code → auto-syncs → run in Databricks"
	@echo ""

auth: ## Authenticate with Prod workspace (one-time)
	databricks auth login --host $(PROD_HOST)
	@echo ""
	@echo "✓ Authenticated with Prod. You can now run 'make sync' to start syncing."

auth-llm: ## Authenticate with LLM workspace (one-time)
	databricks auth login --host $(LLM_HOST)
	@echo ""
	@echo "✓ Authenticated with LLM. You can now run 'make sync-llm' to start syncing."

sync: ## Live-sync files to Prod workspace (watches for changes)
	@echo "Syncing to Prod: $(WORKSPACE_PATH) ..."
	@echo "Files will auto-sync on every save. Press Ctrl+C to stop."
	@echo ""
	databricks sync . $(WORKSPACE_PATH) --profile $(PROD_PROFILE) --watch

sync-llm: ## Live-sync files to LLM workspace (watches for changes)
	@echo "Syncing to LLM: $(WORKSPACE_PATH) ..."
	@echo "Files will auto-sync on every save. Press Ctrl+C to stop."
	@echo ""
	databricks sync . $(WORKSPACE_PATH) --profile $(LLM_PROFILE) --watch

push: ## One-time sync to Prod workspace (no watch)
	databricks sync . $(WORKSPACE_PATH) --profile $(PROD_PROFILE)
	@echo ""
	@echo "✓ Synced to Prod: $(WORKSPACE_PATH)"

push-llm: ## One-time sync to LLM workspace (no watch)
	databricks sync . $(WORKSPACE_PATH) --profile $(LLM_PROFILE)
	@echo ""
	@echo "✓ Synced to LLM: $(WORKSPACE_PATH)"

pull: ## Pull workspace files back to local
	databricks sync ws:$(WORKSPACE_PATH) . --profile $(PROD_PROFILE)
	@echo ""
	@echo "✓ Pulled from $(WORKSPACE_PATH)"

run: ## Run on Prod workspace (data only, no AI agent)
	@echo "Submitting run on Prod cluster $(CLUSTER_ID)..."
	@echo "Notebook: $(NOTEBOOK_PATH)"
	@echo ""
	databricks jobs submit \
		--profile $(PROD_PROFILE) \
		--run-name "analytics-automation-dev" \
		--json '{ \
			"tasks": [{ \
				"task_key": "run_orchestrator", \
				"existing_cluster_id": "$(CLUSTER_ID)", \
				"notebook_task": { \
					"notebook_path": "$(NOTEBOOK_PATH)", \
					"base_parameters": { \
						"date": "$(DATE)", \
						"monitors": "$(MONITORS)", \
						"run_ai_agent": "$(RUN_AI)" \
					} \
				} \
			}] \
		}'
	@echo ""
	@echo "✓ Run complete. Output shown above."

run-llm: ## Run on LLM workspace (with AI agent)
	@echo "Submitting run on LLM cluster $(LLM_CLUSTER_ID)..."
	@echo "Notebook: $(NOTEBOOK_PATH)"
	@echo ""
	databricks jobs submit \
		--profile $(LLM_PROFILE) \
		--run-name "analytics-automation-dev" \
		--json '{ \
			"tasks": [{ \
				"task_key": "run_orchestrator", \
				"existing_cluster_id": "$(LLM_CLUSTER_ID)", \
				"notebook_task": { \
					"notebook_path": "$(NOTEBOOK_PATH)", \
					"base_parameters": { \
						"date": "$(DATE)", \
						"monitors": "$(MONITORS)", \
						"run_ai_agent": "$(RUN_AI)" \
					} \
				} \
			}] \
		}'
	@echo ""
	@echo "✓ Run complete. Output shown above."

run-local: ## Run locally (no Databricks needed)
	python orchestrator.py \
		$(if $(DATE),--date $(DATE)) \
		$(if $(filter False,$(RUN_AI)),--no-ai) \
		--monitors $(MONITORS)

status: ## Check auth status for both workspaces
	@echo "Prod workspace:"
	@databricks auth env --host $(PROD_HOST) 2>&1 || true
	@echo ""
	@echo "LLM workspace:"
	@databricks auth env --host $(LLM_HOST) 2>&1 || true
