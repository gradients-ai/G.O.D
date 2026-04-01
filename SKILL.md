G.O.D SKILL FILE

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


Adding balance (funding an account)
Users fund their Gradients account by sending TAO (Bittensor native token) to their account's deposit address.

Step-by-step flow for agents:
1. Get the user's deposit address:
   - Call POST /account-get-info (requires session token in Authorization header).
   - The response includes bittensor_public_key — this is the SS58 deposit address.
   - If bittensor_public_key is null, call POST /account-get-public-key to generate one on demand. This returns { "public_key": "<ss58_address>", "keypair_created_at": "...", "network": "finney" }.

2. Send TAO to the deposit address:
   - The user transfers TAO from their Bittensor wallet to the bittensor_public_key address.
   - This is a standard Bittensor transfer, e.g.: btcli wallet transfer --dest <bittensor_public_key> --amount <amount>
   - The system automatically detects incoming transfers and credits the account balance.

3. Verify the balance was credited:
   - Wait a few minutes for the transfer to be processed.
   - Login to gradients.io with your fingerprint and confirm.

Important notes on balance:
- Balance is denominated in USD internally. TAO transfers are converted at the current rate.
- Always check pricing (POST /v1/tasks/text/check_price or POST /v1/tasks/image/check_price) before creating tasks so the user knows the cost.
- If a task creation fails due to insufficient balance, advise the user to send more TAO to their deposit address.

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

Operational rules for agents

1. Decision logic (state -> action)
- If status is pending, preparing_data, ready, looking_for_nodes, training, preevaluation, evaluating, or delayed: continue polling and do not create a replacement task unless explicitly asked.
- If status is success: stop polling, extract outputs, and optionally fetch breakdown details.
- If status is failure: inspect the available task details, classify the failure, and only retry if the cause appears transient.

2. Task lifecycle rules (strict ordering)
- Check price before task creation for any non-trivial run.
- Create the task first, persist task_id immediately, then monitor with GET /v1/tasks/{task_id}.
- Do not treat a task as complete until it reaches success or failure.
- Do not run post-training actions such as Chutes deployment until training reaches success.

3. Idempotency and duplicate prevention
- Do not create multiple tasks for the same objective, dataset, model, and hour budget unless the user explicitly asks for a sweep or comparison run.
- If a relevant task_id already exists and is still active, reuse it and continue monitoring instead of creating a new task.
- Before retrying a failed run, confirm that the retry is not duplicating a task that is already pending or training.

4. Task ownership and tracking
- Persist task_id immediately after creation.
- Track, at minimum, task_id, task_type, model_repo, dataset source, hours_to_complete, created_at, status, result_model_name, and trained_model_repository.
- Reuse the same persisted task_id for all follow-up status, breakdown, delete, and deployment operations.

5. System capabilities
- Can launch paid fine-tuning jobs for text, chat, DPO, GRPO, image, and environment tasks.
- Can launch multiple tasks for comparison across datasets, models, or reward configurations.
- Can check pricing, poll task states, inspect breakdowns, and deploy a base-model-plus-LoRA pair to Chutes.
- Can read public network, auditing, tournament, and performance endpoints without an API key where documented.

6. System limitations
- This is a job-based training API, not a real-time inference or chat-completions system.
- External users should assume X-Scheduler-Auth is unavailable.
- No public endpoint is documented for editing task parameters after creation; if the configuration is wrong, create a new task.
- Internal validator, miner, and trainer subsystems are not fully exposed through the public API.

7. Non-goals and misuse prevention
- Do not use Gradients for real-time chat or inference serving.
- Do not treat /tournament/* or /auditing/* as training submission endpoints.
- Do not assume direct shell access to training containers or internal validator services.

8. Failure types
- Account or auth failures: missing API key, expired token, insufficient balance, or rate limiting.
- Validation failures: invalid model_repo, invalid dataset reference, malformed payload, or wrong endpoint for the task type.
- Data-preparation failures: dataset schema mismatch, inaccessible dataset files, or unsupported file_format.
- Runtime failures: trainer allocation issues, delayed scheduling, training failure, or evaluation failure.

9. Failure detection
- If task creation fails immediately, assume auth, billing, schema, or endpoint misuse before assuming infrastructure failure.
- If status becomes failure before started_at is set, treat it as likely validation or data-preparation failure.
- If status remains delayed or looking_for_nodes for an extended period, treat it as a scheduling or capacity issue.
- If status reaches failure after training or evaluating began, treat it as a runtime or evaluation failure.

10. Retry policy
- Retry at most once for likely transient failures such as temporary scheduling or network issues.
- Do not retry until the prior task is confirmed to be in a terminal failure state.
- For retries, keep the original parameters unless there is evidence they caused the failure.

11. Abort conditions
- Do not retry if the model repo, dataset source, file_format, or payload schema is invalid.
- Do not retry if the user has insufficient balance or lacks valid auth.
- Stop and ask the user before launching more paid runs if repeated failures indicate a systematic configuration problem.

12. Parallelization strategy
- Prefer parallel runs when the user explicitly wants comparison across multiple models, datasets, or reward setups.
- Monitor all active task_ids concurrently and report each task separately.
- Keep each run independently tracked so one failure does not overwrite the state of another.

13. Experimentation strategy
- When the user wants exploration, vary one major factor at a time such as base model, dataset, or training hours.
- Prefer small initial runs to validate dataset and payload assumptions before scaling out.
- Use repeated runs only when the goal is comparison, ablation, or robustness testing.

14. Resource awareness
- Always check price before launching expensive runs or batches.
- Be especially cautious with larger text models and long-duration jobs.
- Prefer low-hour validation runs before committing to long or many-job experiments.

15. Result extraction rules
- On success, extract task_id, status, result_model_name, trained_model_repository, started_at, and finished_at.
- If available and relevant, also fetch GET /v1/tasks/breakdown/{task_id}.

16. Evaluation interpretation
- Use /v1/tasks/breakdown/{task_id} for score details and miner-level information when the user wants analysis beyond top-level task status.
- Treat trained_model_repository and result_model_name as the primary outputs of a successful paid run.

17. Persistence rules
- Save task_id, current status, timestamps, task type, base model, dataset source, hour budget, and final model outputs for every run.
- Preserve this state across follow-up operations so the agent does not lose task ownership.

18. API usage rules
- Use the task-specific endpoint that matches the workload instead of forcing every task through /v1/tasks/create.
- Use read endpoints for monitoring and analysis, and submission endpoints only for creating paid work.
- Prefer Bearer API key auth for end-user automation.

19. Schema validation behavior
- Always consult https://api.gradients.io/docs before constructing or changing payloads.
- Match dataset field names to the task schema instead of guessing column names.
- Do not hallucinate undocumented fields or endpoints.

20. Auth constraints
- Use Authorization: Bearer <token> for session tokens and API keys unless the docs explicitly require otherwise.
- Do not assume access to X-Scheduler-Auth or other internal credentials.
- If auth fails, stop and refresh or recreate credentials before attempting paid operations again.

21. Polling strategy
- Poll every 10 to 30 minutes for active tasks.
- Continue polling until the task reaches success or failure.
- Slow the polling rate for long-running jobs and when monitoring many tasks at once.

22. Patience and waiting rules
- Do not interrupt or replace tasks simply because they remain in training, preevaluation, or evaluating.
- Treat delayed and looking_for_nodes as waiting states first, not immediate failures.

23. No-assumption rule
- Never assume task creation succeeded without capturing the returned task_id.
- Never assume training succeeded without confirming status == success.
- Never assume outputs exist until trained_model_repository or equivalent result fields are present.

24. Decision priority rules
- Prioritize completing or monitoring existing tasks before creating new paid work.
- Prioritize validation, pricing, and task reuse before expansion to parallel experiments.

25. Multi-task coordination
- Track every active task_id separately.
- Report each task's state, cost context, and outputs independently.
- If one task succeeds and others fail, do not collapse them into a single summary state.

26. Cost optimization strategy
- Start with low-hour runs before scaling up expensive models or large experiment grids.
- Use price checks to compare candidate models before launching many tasks.
- Avoid duplicate runs unless they serve a deliberate comparison objective.

27. Debugging guidance
- If failures repeat, verify endpoint choice, auth, dataset accessibility, dataset schema, and model repo before retrying.
- If the same configuration fails more than once, change the dataset, model, or payload instead of repeating the same request.
- When possible, use the task state timeline and breakdown endpoints to determine whether the issue happened during validation, scheduling, training, or evaluation.

Example user intents this system supports
- "Create a dataset and use this API to train a model."
- "Launch 10 DPO runs against different base models."
- "Fine-tune 30 models on the same dataset and compare outputs."
- "Show me the current tournament fees and the latest tournament weights."
- "Deploy the winning LoRA to Chutes."
- "Continuously train a model on a large dataset over multiple iterations."
- "Run a long-running training job with 500k samples across 10 iterations."

---

Gradients Scheduler (Long-Running / Multi-Iteration Training)

When to use the scheduler
The single-shot Gradients API (POST /v1/tasks/create) is ideal for one-off training jobs. The scheduler is for scenarios that require:
- Large datasets that must be split into multiple training chunks across iterations.
- Multiple sequential training iterations where each iteration merges the LoRA adapter back into the base model and feeds the result into the next round.
- Continuous training loops that run unattended over hours or days.
- Multiple datasets combined and stratified across training chunks.

The scheduler is a hosted service. Users access it through the public Gradients API — no self-hosting, cloning, or infrastructure setup is required. The Gradients API proxies all scheduler requests to the backend service automatically.

How the scheduler works (lifecycle)
1. User creates a job via POST /v1/scheduler/jobs/create with model, datasets, samples_per_training, hours_to_complete, etc.
2. Job is stored as "pending".
3. The scheduler picks up the job, downloads and merges all datasets, standardizes column names, shuffles, and saves to disk.
4. Datasets are split into chunks of samples_per_training size.
5. For each training iteration:
   a. The current chunk is uploaded to storage and a presigned URL is generated.
   b. A training task is created on the Gradients API with the dataset URL.
   c. The scheduler polls the task until it reaches success or failure.
   d. On success: the best miner's trained LoRA model is merged with the base model, producing a new merged model.
   e. The merged model becomes the base model for the next iteration.
   f. The next chunk is selected (cycling through chunks via training_number % num_chunks).
6. This continues until all iterations complete, the job is suspended, or consecutive failures (3) cause the job to be marked as failed.

Scheduler API endpoints

All scheduler endpoints are on the public Gradients API (https://api.gradients.io). Authenticate with your Gradients API key using Authorization: Bearer <API_KEY>. The API automatically associates jobs with your account — no X-Account-ID header is needed.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /v1/scheduler/health | Health check |
| POST | /v1/scheduler/jobs/create | Create a new training job |
| GET | /v1/scheduler/jobs | List your jobs |
| GET | /v1/scheduler/jobs/{job_id} | Get job status and config |
| GET | /v1/scheduler/jobs/{job_id}/results | Get job details with all training results |
| DELETE | /v1/scheduler/jobs/{job_id} | Delete job and all associated data |

Creating a training job

Required information to ask the user:
1. task_type: "InstructText", "Chat", or "CustomDatasetChat"
2. model_repo: HuggingFace model ID (e.g. "Qwen/Qwen2.5-1.5B-Instruct")
3. hours_to_complete: Hours allocated per training iteration (integer, e.g. 1)
4. samples_per_training: Number of samples per training chunk (e.g. 80000)
5. final_test_size: Proportion held out for final test set (float between 0 and 1, e.g. 0.1)
6. datasets: At least one dataset with:
   - For InstructText: name (HF dataset ID), field_instruction (required), field_input (optional), field_output (optional), max_rows (optional)
   - For Chat: name, chat_column (optional), chat_role_field (optional), chat_content_field (optional), chat_user_reference (optional), chat_assistant_reference (optional), chat_template (optional), max_rows (optional)

Optional fields:
- name: Human-readable job name
- random_seed: Default 42
- min_days, max_days, min_hours, max_hours: Scheduling interval between iterations (all default 0 = immediate)
- per_chunk_test_proportion: Test proportion per chunk for CustomDatasetChat (default 0.001)

Example curl for creating an InstructText job:
```
curl -X POST "https://api.gradients.io/v1/scheduler/jobs/create" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <API_KEY>" \
  -d '{
    "name": "my-training-job",
    "task_type": "InstructText",
    "model_repo": "Qwen/Qwen2.5-1.5B-Instruct",
    "hours_to_complete": 1,
    "samples_per_training": 80000,
    "final_test_size": 0.1,
    "datasets": [
      {
        "name": "yahma/alpaca-cleaned",
        "field_instruction": "instruction",
        "field_input": "input",
        "field_output": "output"
      }
    ]
  }'
```

The response returns job_id, status, and a message.

Monitoring a training job

Monitoring workflow:
1. After creating the job, poll GET /v1/scheduler/jobs/{job_id} to check job status.
2. Job statuses: pending -> running -> completed/suspended/failed.
3. Poll GET /v1/scheduler/jobs/{job_id}/results for detailed training results per iteration.
4. Each result in the results array contains: task_id, training_number, status (pending/running/success/failure), base_model_repo, trained_model_repo, merged_model_repo, test_loss, quality_score, winner_hotkey, error_message.
5. The final trained model is the merged_model_repo from the last successful task result.
6. If the job completes or is suspended, the latest merged_model_repo is the final output model.

Example monitoring curl:
```
curl "https://api.gradients.io/v1/scheduler/jobs/{job_id}/results" \
  -H "Authorization: Bearer <API_KEY>"
```

Providing the final model to the user:
- Extract merged_model_repo from the last successful task result in the results response.
- This is a HuggingFace model repository ID (e.g. "username/merged-model-name").
- The user can use this model directly from HuggingFace for inference.

Scheduler job states
- pending: Job created, waiting for scheduler to pick it up.
- running: Scheduler is actively processing (preparing data, training, or between iterations).
- waiting_to_suspend: User requested suspension; scheduler will suspend after current task completes.
- suspended: Job paused.
- completed: All training iterations finished successfully.
- failed: Job failed after 3 consecutive task failures.

Scheduler-specific operational rules for agents

1. Always ask the user for model_repo, dataset details, hours_to_complete, and samples_per_training before creating a job.
2. Recommend final_test_size of 0.05-0.15 unless the user specifies otherwise.
3. For large datasets (>100k rows), suggest samples_per_training of 50000-100000 to create multiple training chunks.
4. The number of training iterations equals ceil(total_train_samples / samples_per_training).
5. Each iteration costs one Gradients API training task (billed at the model's hourly rate * hours_to_complete).
6. Total cost = num_iterations * price_per_iteration. Warn the user about total cost for large jobs.
7. Monitor jobs by polling /v1/scheduler/jobs/{job_id}/results periodically.
8. The final deliverable is the merged_model_repo from the last successful task result.
9. If the user wants to stop early, use DELETE /v1/scheduler/jobs/{job_id}.
10. If the user needs the intermediate model at any point, the merged_model_repo from any successful result can be used.

When an agent should use the scheduler vs single-shot API
- Use single-shot API (POST /v1/tasks/create): One-off training, small datasets, quick experiments, no iteration needed.
- Use the scheduler: Large datasets requiring chunking, continuous iterative training, multi-dataset jobs, unattended long-running training, or when each iteration should build on the previous merged model.

Troubleshooting
- If tasks fail: Verify the API key has sufficient balance and the model_repo / dataset are valid.
- If dataset preparation fails: Check that the dataset name is a valid HuggingFace dataset ID and the field names match actual columns in the dataset.
- Job stuck in running: Check the task_id in the results and poll the Gradients API directly (GET https://api.gradients.io/v1/tasks/{task_id}).