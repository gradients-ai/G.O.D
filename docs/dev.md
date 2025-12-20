## For dev without docker:

0. Clone the repo
```bash
git clone https://github.com/rayonlabs/G.O.D.git
cd G.O.D
```

1. Run bootstrap.sh
```bash
sudo -E bash bootstrap.sh
source $HOME/.bashrc
source $HOME/.venv/bin/activate
```

2. Install dependencies
```bash
find . -path "./venv" -prune -o -path "./.venv" -prune -o -name "requirements.txt" -exec pip install -r {} \;
./install_axolotl.sh
pip install "git+https://github.com/besimray/fiber.git@v2.6.0#egg=fiber[full]"

```

3. Setup dev env

```bash
task setup
```

4.  Run validator in dev mode

```bash
task validator
```

## Dev vali/miner communication

To test validator and miner communication locally:

1. Register a miner in dev mode in the testnet (requires test TAOs): [Miner Setup Guide](docs/miner_setup.md).

2. Under `miner/endpoints/tuning.py`, comment out the API  `dependencies`.

3. [Optional] add a dependency to only accept requests from your specific dev vali IP address.

4. Start a validator in another terminal in dev mode: [Validator Setup Guide](validator_setup.md).

5. [Optional] Use the `Python Debugger: Validator Cycle` configuration in `.vscode/launch.json` to attach to the validator and debug the connection using your preset breakpoints.
