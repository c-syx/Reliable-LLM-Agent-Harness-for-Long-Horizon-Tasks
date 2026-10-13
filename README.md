# Reliable LLM agent harness for long horizon tasks (ESP3201A)

An LLM agent that tidies a messy folder of student files. It reads each file, works out which module it belongs to, moves it into a folder for that module, and writes a log. The baseline agent is a plain tool loop with raw history. There is no verification, no checkpointing and no recovery. The improved harness will be built on top of this same loop so the two can be compared.

There is nothing hard coded about modules. No module list, no topic to module mapping and no file name rules are given to the agent. It has to read the contents (PDF, Word, PowerPoint, Excel, notebooks, code and text) and decide. Your original files are never changed, because the agent works on a copy.

## Setup

```
pip install -r requirements.txt
python -m pytest
```

You should see all tests pass.

## API key

1. Make an account at openrouter.ai, go to Keys, and create a key.
2. In the repo folder, create a file called `.env` containing one line.

```
OPENROUTER_API_KEY=paste_your_key_here
```

The `.env` file is ignored by git, so the key will not be uploaded. Never paste the key into any other file in the repo.

3. Free models allow 50 requests a day on a new account. Adding at least 10 dollars of credit once raises that to 1000 a day. You do not have to spend the credit if you stay on free models.

## Get your Google Drive files

Right click the ESP3201A Test Files folder in Google Drive and choose Download. Drive gives you a zip, sometimes several. Unzip everything into a folder called `test_files` inside the repo folder. Git ignores that folder, so personal files never get uploaded by accident.

## Run it on real files

```
python organise.py --input test_files --output organised_output --model qwen/qwen3.8-27b:free
```

The model id comes from the model's page on OpenRouter. Check that the model supports tool calling, since the agent needs it. If it does not, the first call fails with an error and you can pick another model.

The output folder must be new or empty. When the run ends you get these.

- `organised_output/Organised/` has the files sorted into one folder per module, plus the agent's own log.
- `organised_output/moves_report.csv` shows where every original file ended up.
- `organised_output/_left_behind/` holds anything the agent did not move.
- `organised_output/_run/transcript.json` is the full record of every step, for your report.

A run on about 55 files takes roughly 130 model calls, which is several minutes on the free tier.

## Scoring a run

Reading the output by eye does not scale. For a measurable result, write an answer key once.

```
python organise.py --input test_files --make-labels labels.csv
```

Open labels.csv and fill in the module column for each file, using the module code. Use Unsorted for files that are not coursework. Rows left blank are not scored. The agent never sees this file. Then run with it.

```
python organise.py --input test_files --output run2 --model qwen/qwen3.8-27b:free --labels labels.csv
```

It prints the accuracy and lists every wrong file. Keep labels.csv in your shared Drive folder rather than the repo, since file names may contain personal details.

## What the agent can do

It has six tools, which are list_dir, read_file, make_dir, move_file, write_file and finish. All paths are confined to the workspace. A move fails if the destination already exists, so the agent has to cope with name clashes. Images and videos have no readable text, so the agent only gets a note saying so and has to use file names and neighbouring files, or put them in Unsorted. This is deliberate. It is where reading and reasoning matter most.

## Generated benchmark for repeated experiments

`run_experiment.py` builds seeded messy folders with a known correct answer and runs the agent over them several times. It is useful for quick checks while developing the harness. Its files state their module on a line, and its prompt explains a naming rule, so it is easier than real files. Treat it as a development set and use the real folder as the main test.

```
python run_experiment.py --mock --repeats 2 --out results_dryrun
python run_experiment.py --model qwen/qwen3.8-27b:free --n-tasks 2 --repeats 1
```

The first line is a dry run with no API key. Results are saved as JSON per run, with success, file accuracy, false success, wrong moves and how many moves follow the first mistake.

## Layout

- organise.py, the command line tool for real folders
- run_experiment.py, the generated benchmark runner
- harness/agent.py, the baseline loop
- harness/real.py, the real mode prompt, export and scoring
- harness/readers.py, text extraction for each file type
- harness/sandbox.py and harness/tools.py, the workspace and tool definitions
- harness/llm.py, the OpenRouter client with throttling and retries
- harness/tasks.py and harness/scoring.py, the generated benchmark
- harness/mock.py, scripted models used only by tests and the dry run
- tests, pytest checks
