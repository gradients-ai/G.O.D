GRADIENTS.IO / G.O.D / RAYON API SKILL FILE

Purpose
This file is for an LLM or agent that needs to understand Gradients.io's training product, its public API, and the G.O.D. subnet that powers training jobs and tournaments.

Short version
Gradients.io is a training orchestration system built on Bittensor subnet 56 ("G.O.D", Gradients on Demand). Users create paid fine-tuning jobs through the public API. The API bills an account, forwards the job to a private validator, and the validator coordinates miners/trainers to produce a trained model. The same ecosystem also runs open tournaments where miners submit open-source training repos and compete on standardized tasks.

Public URLs
- Main product site: https://gradients.io
- API base URL: https://api.gradients.io
- Human-friendly API docs: https://api.gradients.io/docs
- FastAPI swagger docs: https://api.gradients.io/swagger
- Tournament results page: https://gradients.io/app/research/tournament/{TOURNAMENT_ID}
- Tournament fees: GET https://api.gradients.io/tournament/fees
- Tournament balance lookup: GET https://api.gradients.io/tournament/balance/{coldkey}

Important framing
- Use https://api.gradients.io for creating and monitoring jobs.
- Use https://gradients.io for the product website and tournament/research pages.
- This is primarily a training/fine-tuning platform, not a generic chat completion API.
- The repo named G.O.D is not a website frontend. It is the subnet/validator/miner/trainer system that executes training jobs and tournaments.

What the system does
1. Accepts training requests for text, chat, DPO, GRPO, image, and environment tasks.
2. Prices jobs based on model size and hours requested.
3. Charges the user's account balance.
4. Sends the task to a private validator on subnet 56.
5. The validator stores the task, schedules training/evaluation, and coordinates trainer infrastructure.
6. Training artifacts and resulting models are tracked through the task record.
7. The broader subnet also runs recurring tournaments where miners expose a repo endpoint and compete with open-source training code.

Main product capabilities
- Fine-tune text instruction models.
- Fine-tune chat models.
- Fine-tune DPO preference models.
- Fine-tune GRPO / reward-driven models.
- Fine-tune image models such as SDXL and Flux variants.
- Launch training from a Hugging Face dataset reference or from pre-prepared dataset URLs.
- Check prices before creating jobs.
- Poll task state and fetch result breakdowns.
- View public network status and recent completed jobs.
- Deploy LoRA adapters to Chutes for inference after training.
- View tournament data, fees, balances, analytics, and performance projections.

Public API auth model
- End-user automation should normally use an API key in the Authorization header.
- The middleware accepts either "Authorization: Bearer <token>" or a raw token value, but Bearer is the safest choice.
- Scheduler auth exists via X-Scheduler-Auth, but that is an internal service token and should not be assumed to be available to third-party agents.

Account bootstrap flow
If an agent needs to fully bootstrap a user account from scratch:
1. POST /account-create with a username.
2. Receive a fingerprint.
3. POST /auth-with-fingerprint with that fingerprint to create a session token.
4. Use the session token in Authorization.
5. POST /api-key-create to mint a long-lived API key.
6. Use the API key for training endpoints.

Useful account endpoints
- POST /account-create
- POST /auth-with-fingerprint
- POST /api-key-create
- POST /account-get-info
- POST /account-get-public-key

Billing model
- Text jobs are priced by model size bucket and hours requested.
- Image jobs use a flat hourly rate.
- Current code-level defaults:
  - <=1B text: $10/hour
  - <=7B text: $15/hour
  - <=40B text: $25/hour
  - >40B text: $50/hour
  - image: $5/hour
- Always check current pricing through the API before creating large batches.

Pricing endpoints
- POST /v1/tasks/text/check_price
- POST /v1/tasks/image/check_price
- GET /v1/prices

Task types you can create
- InstructTextTask
- ChatTask
- DpoTask
- GrpoTask
- ImageTask
- EnvTask

Core task creation endpoints
- POST /v1/tasks/create
- POST /v1/tasks/create_chat
- POST /v1/tasks/create_dpo
- POST /v1/tasks/create_grpo
- POST /v1/tasks/create_image
- POST /v1/tasks/create_custom_dataset_text
- POST /v1/tasks/create_custom_dataset_chat

Task monitoring and retrieval endpoints
- GET /v1/tasks/{task_id}
- GET /v1/tasks
- GET /v1/tasks/account/{account_id}
- GET /v1/tasks/breakdown/{task_id}
- DELETE /v1/tasks/delete/{task_id}
- GET /v1/tasks/organic/completed
- GET /v1/network/status

Public read endpoints (no API key required)
- GET /v1/network/status
- GET /v1/performance/latest-tournament-weights
- GET /v1/performance/weight-projection
- GET /v1/performance/weight-projection-static
- GET /v1/performance/last-boss-battle
- GET /auditing/tasks
- GET /auditing/tasks/hotkey/{hotkey}
- GET /auditing/tasks/{task_id}
- GET /auditing/scores-url
- GET /tournament/fees


Relevant resources to train a model:
- API docs URL: https://api.gradients.io/docs
- API base URL: https://api.gradients.io
- A valid API key

"Read this skill file first, then use https://api.gradients.io/docs as the schema reference. Use the Gradients API to create a dataset, estimate price, launch one or more fine-tuning jobs, and monitor them until task IDs are returned."

Best mental model for training:
- Gradients is job-based, not chat-based.
- The goal is to create one or more training tasks, not to open a websocket or run a long interactive inference session.
- The most important outputs are task IDs, account billing effects, task status, and trained model repositories.

Minimal workflow for text fine-tuning
1. Decide task type: instruct, chat, DPO, or GRPO.
2. Choose a base model repo, usually a Hugging Face model ID.
3. Choose a dataset source:
   - Hugging Face dataset repo via ds_repo and file_format=hf
   - Prebuilt dataset URLs via create_custom_dataset_text or create_custom_dataset_chat with file_format=s3
4. Call the price check endpoint.
5. Create the task.
6. Poll GET /v1/tasks/{task_id}.

Minimal workflow for image fine-tuning
1. Prepare presigned URLs for image/text pairs.
2. Choose a base image model repo.
3. Choose model_type.
4. Call POST /v1/tasks/image/check_price.
5. Call POST /v1/tasks/create_image.
6. Poll GET /v1/tasks/{task_id}.

Task payload expectations

Instruct text task
- Endpoint: POST /v1/tasks/create
- Important fields:
  - ds_repo
  - model_repo
  - file_format
  - hours_to_complete
  - field_instruction
  - field_input (optional)
  - field_output (optional)
  - field_system (optional)
  - result_model_name (optional)
  - yarn_factor (optional)

Example instruct payload
{
  "ds_repo": "yahma/alpaca-cleaned",
  "model_repo": "Qwen/Qwen2.5-Coder-32B-Instruct",
  "file_format": "hf",
  "hours_to_complete": 1,
  "field_instruction": "instruction",
  "field_input": "input",
  "field_output": "output"
}

Chat task
- Endpoint: POST /v1/tasks/create_chat
- Important fields:
  - ds_repo
  - model_repo
  - file_format
  - hours_to_complete
  - chat_template
  - chat_column (optional)
  - chat_role_field
  - chat_content_field
  - chat_user_reference (optional)
  - chat_assistant_reference (optional)

Example chat payload
{
  "ds_repo": "Magpie-Align/Magpie-Pro-300K-Filtered",
  "model_repo": "Qwen/Qwen2.5-7B-Instruct",
  "file_format": "hf",
  "hours_to_complete": 2,
  "chat_template": "chatml",
  "chat_column": "conversations",
  "chat_role_field": "from",
  "chat_content_field": "value",
  "chat_user_reference": "user",
  "chat_assistant_reference": "assistant"
}

DPO task
- Endpoint: POST /v1/tasks/create_dpo
- Important fields:
  - ds_repo
  - model_repo
  - file_format
  - hours_to_complete
  - field_prompt
  - field_chosen
  - field_rejected
  - field_system (optional)
  - prompt_format / chosen_format / rejected_format (optional)

GRPO task
- Endpoint: POST /v1/tasks/create_grpo
- Important fields:
  - ds_repo
  - model_repo
  - file_format
  - hours_to_complete
  - field_prompt
  - reward_functions
- reward_functions is a list of reward references, each with:
  - reward_id
  - reward_weight

Image task
- Endpoint: POST /v1/tasks/create_image
- Important fields:
  - model_repo
  - image_text_pairs
  - ds_id
  - hours_to_complete
  - result_model_name (optional)
  - model_type
- image_text_pairs is a list of:
  - image_url
  - text_url

Custom dataset endpoints
Use these when the dataset has already been prepared and uploaded somewhere the trainer can fetch it from.

Text custom dataset
- Endpoint: POST /v1/tasks/create_custom_dataset_text
- Important fields:
  - training_data
  - test_data (optional)
  - ds_repo (optional original source)
  - file_format should usually be s3
  - plus the normal instruct-text schema fields

Chat custom dataset
- Endpoint: POST /v1/tasks/create_custom_dataset_chat
- Important fields:
  - training_data
  - test_data (optional)
  - ds_repo (optional original source)
  - file_format should usually be s3
  - plus the normal chat schema fields

What a successful create call returns
- success
- task_id
- created_at
- account_id

What task detail records contain
- id
- account_id
- status
- created_at
- started_at
- finished_at
- hours_to_complete
- task_type
- result_model_name
- trained_model_repository

Common task states
- pending
- preparing_data
- ready
- looking_for_nodes
- training
- preevaluation
- evaluating
- success
- failure
- delayed

Interpreting outputs
- The main handle is task_id.
- Keep polling until status is success or failure.
- On success, inspect trained_model_repository and result_model_name.
- For score or miner-level details, call GET /v1/tasks/breakdown/{task_id}.

Important safety and practical notes for agents
- Check pricing before submitting large batches.
- Check account balance and rate limits if the account system is available to you.
- Use the correct endpoint for the task type instead of forcing everything through /v1/tasks/create.
- For external users, assume scheduler auth is unavailable.
- This API is a public gateway that proxies to a private validator; not every internal subsystem is directly exposed.
- Treat /tournament/* and /auditing/* as read-oriented product endpoints, not training submission endpoints.

Tournament system overview
- Tournaments are separate from paid organic jobs, but share the same ecosystem.
- Miners expose GET /training_repo/{task_type} from their miner.
- Validators pull repo URLs and commit hashes, build miner code in Docker, and score performance.
- Tournament types include text, image, and environment.
- Typical cadence:
  - environment tournaments start Mondays
  - text/image tournaments start Thursdays
- Fees are burned and can be queried from the public API.

Tournament miner endpoint
- Miners expose /training_repo/{task_type}
- Response contains:
  - github_repo
  - commit_hash
  - github_token (optional for private repos)

Trainer-side expectations inside the subnet
- Training repos are cloned by trainer infrastructure.
- Repos are expected to provide standardized Dockerfiles and CLI entrypoints.
- Output paths are fixed so validators can pick up results reliably.
- This matters for miners and tournament participants, not normal API consumers.

Chutes deployment support
Gradients also exposes Chutes deployment for a base model + LoRA combination.

Endpoints
- POST /v1/chutes/deploy
- GET /v1/chutes/status/{chute_id}

Deploy payload
{
  "model_id": "base-model-repo",
  "lora_id": "lora-adapter-repo"
}

Performance and analytics endpoints
Useful for research agents, dashboards, and tournament analysis.

Endpoints
- GET /v1/performance/latest-tournament-weights
- GET /v1/performance/weight-projection
- GET /v1/performance/weight-projection-static
- GET /v1/performance/last-boss-battle

When an agent should use Gradients
- When the goal is to launch one or more fine-tuning jobs on hosted infrastructure.
- When the user has a dataset or can generate one.
- When the user wants to compare many fine-tuning runs.
- When the user wants a trained artifact rather than an inference response.

When an agent should not use Gradients
- When the goal is real-time chat completion.
- When no billable account or API key exists.
- When the user expects direct shell access to training containers.
- When the user wants a single monolithic batch endpoint instead of many explicit jobs.

Suggested reusable agent instructions
- "Use https://api.gradients.io as the source of truth for job creation."
- "Use https://api.gradients.io/docs for schema discovery before constructing payloads."
- "Prefer Bearer API key auth."
- "Check price before launch."
- "Persist every task_id."
- "Poll task status until terminal state."
- "Return task IDs, statuses, and any trained model repositories."

Example user intents this system supports
- "Create a dataset and use this API to train a model."
- "Launch 10 DPO runs against different base models."
- "Fine-tune 30 models on the same dataset and compare outputs."
- "Show me the current tournament fees and the latest tournament weights."
- "Deploy the winning LoRA to Chutes."