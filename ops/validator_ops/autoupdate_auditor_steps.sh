# Steps to take to update the auditor automatically
# Change each time but take caution

. $HOME/.venv/bin/activate

# Package install (extras + cyscale handling) lives in `task install`.
# Only restart the auditor if the install succeeded so a broken venv does not
# replace a still-running process.
task install && task auditor
