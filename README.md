# Reliable LLM agent harness for long horizon tasks (ESP3201A)

An LLM agent that tidies a messy folder of student files. It reads each file, works out which module it belongs to, moves it into a folder for that module, and writes a log. The baseline agent is a plain tool loop with raw history. The verified agent adds deterministic action verification. The improved agent combines verification, folder structure handling and checkpointing so an interrupted run can be resumed.

By default there is nothing hard coded about modules. No module list, no topic to module mapping and no file name rules are given to the agent. The optional timetable (see below) gives it a list of module codes to expect, but it still has to read the contents of each file (PDF, Word, PowerPoint, Excel, notebooks, code and text) to decide. Your original files are never changed, because the agent works on a copy.

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
python organise.py --input test_files --output organised_output --model cohere/north-mini-code:free
```

The model id comes from the model's page on OpenRouter. Check that the model supports tool calling, since the agent needs it. If it does not, the first call fails with an error and you can pick another model.

Free models sometimes return 429 (too many requests) or 503 (overloaded) errors. The program waits and retries up to 10 times and keeps any work already done. If a model keeps failing, try another model and record it in MODEL_LOG.md.

The output folder must be new or empty. When the run ends you get these.

- `organised_output/Organised/` has the files sorted into one folder per module, plus the agent's own log.
- `organised_output/moves_report_<date>_<time>.csv` shows where every original file ended up. The name carries the time the run finished, for example `moves_report_20261025_12.20pm.csv`.
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
python organise.py --input test_files --output run2 --model cohere/north-mini-code:free --labels labels.csv
```

It prints the accuracy and lists every wrong file. Keep labels.csv in your shared Drive folder rather than the repo, since file names may contain personal details.

## Using your timetable

Many files never write their module code inside them, and without help those end up in Unsorted. Give the agent your timetable and it will know which modules to expect.

```
python organise.py --input test_files --output organised_output --model cohere/north-mini-code:free --timetable timetable.pdf
```

The timetable can be a PDF, Word, Excel, ICS or plain text file. A screenshot will not work, because the program needs text to read. NUSMods can export an ICS file, which is the easiest option. Keep the timetable outside the input folder, otherwise it gets sorted along with everything else.

What happens
1. The model reads the timetable once and lists the modules in it. Every code it returns is checked against the timetable text, so an invented module is dropped.
2. The list is saved to `module_registry.json`. Open it and check it looks right. This is the agent's memory of your modules.
3. The list goes into the agent's instructions. For a file with no module code, the agent picks the most likely module from the list using the topic and the neighbouring files. It only uses Unsorted for non coursework or files that fit nothing.
4. Next time, pass `--timetable module_registry.json` and the extraction step is skipped, which saves a request.

The agent's own log now says stated or inferred for each file, so you can see which choices were written in the file and which were guessed. The run summary also shows how many files are unsorted and whether the agent made folders that are not on the timetable.

Expect a trade off. Fewer files in Unsorted, but some of the ones that move will be wrong guesses. Score both with a labels file and report both numbers, a run with the timetable against a run without it is a good comparison for your report.

To build one registry from several timetables (different semesters, any of .ics, .pdf, .txt, .docx, .xlsx), use make_registry.py. It uses a pattern match and makes no model calls.

```
python make_registry.py timetable_s1.ics timetable_s2.pdf --out module_registry.json
```

Open the result and check the codes. The pattern can pick up things that look like module codes but are not.

module_registry.json and timetable files are gitignored because they can contain personal details, so they stay on your computer.

## Models tried

See [MODEL_LOG.md](MODEL_LOG.md) for a record of models tried, their results, and notes.

## What the agent can do

The baseline and verified agents have six tools: list_dir, read_file, make_dir, move_file, write_file and finish. The improved agent also gets move_folder. All paths are confined to the workspace. A move fails if the destination already exists, so the agent has to cope with name clashes. Images and videos have no readable text, so the agent only gets a note saying so and has to use file names and neighbouring files, or put them in Unsorted. This is deliberate. It is where reading and reasoning matter most.

## Agent variants

- `--agent baseline` is the plain loop with no checks or recovery.
- `--agent verified` snapshots actions and checks that a move really happened without changing the file's content, and that make_dir and write_file left what they claimed.
- `--agent improved` does everything the verified agent does, keeps subfolder structure with move_folder, and saves a checkpoint after every step.

Use the improved agent for long real runs:

```
python organise.py --input test_files --output organised_output --model <model> --agent improved
```

If the model becomes unavailable, run the same command again with `--resume`. The workspace and conversation history are restored from `organised_output/_run/checkpoint.json`. Add `--fresh-context` to resume with a clean history and let the agent inspect the workspace instead.

## Generated benchmark for repeated experiments

`run_experiment.py` builds seeded messy folders with a known correct answer and runs the agent over them several times. It is useful for quick checks while developing the harness. Its files state their module on a line, and its prompt explains a naming rule, so it is easier than real files. Treat it as a development set and use the real folder as the main test.

```
python run_experiment.py --mock --repeats 2 --out results_dryrun
python run_experiment.py --model cohere/north-mini-code:free --n-tasks 2 --repeats 1
```

The first line is a dry run with no API key. Results are saved as JSON per run, with success, file accuracy, false success, wrong moves and how many moves follow the first mistake.

## Layout

- organise.py, the command line tool for real folders
- run_experiment.py, the generated benchmark runner
- harness/agent.py, the shared tool loop and hooks
- harness/verified_agent.py and harness/verifier.py, deterministic action verification
- harness/improved.py, verification plus folder handling and checkpointing
- harness/folder_check.py, the subfolder-structure check
- harness/checkpoint.py, atomic save and load for resumable runs
- harness/real.py, the real mode prompt, export and scoring
- harness/readers.py, text extraction for each file type
- harness/sandbox.py and harness/tools.py, the workspace and tool definitions
- harness/llm.py, the OpenRouter client with throttling and retries
- harness/tasks.py and harness/scoring.py, the generated benchmark
- harness/mock.py, scripted models used only by tests and the dry run
- make_registry.py, builds a module registry from timetables without any model call
- MODEL_LOG.md, record of the models tried
- tests, pytest checks