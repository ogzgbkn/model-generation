# On the Impact of Requirements Smells in Prompts: The Case of Automated Sequence Diagram Creation

This is the dataset for the master's thesis _On the Impact of Requirements Smells in Prompts: The Case of Automated Sequence Diagram Creation_. It contains the following data:

| Path                                 | Description                                                                  |
| ------------------------------------ | ---------------------------------------------------------------------------- |
| `evaluation_analysis/`                 | The matplotlib codes for diagram creation and the diagrams in the thesis report.                             |
| `evaluations/`                 | The generated diagrams and their evaluation files for each variant of each game.                            |
| `games/`                             | The code and requirements in CSV format for all games.                       |
| `ground_truths/`                      | The ground truth diagrams for all games (visual and plantUML forms)                                              |
| `llm-request-code/`                  | The code used to generate the samples and prompt the LLMs via APIs.          |
| `prompts/`                           | The prompts used with the LLMs.                                              |
| `batch_id.txt`                           | The batch id from OpenAI after running the Run phase.                                              |
| `batch.jsonl`                           | The batch that is being sent to OpenAI batch API to generate the sequence diagrams.                                              |
| `batch_results.jsonl`                           | The result of the batch API request, again as a batch.                                              |
| `experiments.json`                           | The conducted experiments from the Run phase.                                              |
| `updated_interpretation_table_v2.csv`                           | The results of conducted statistical tests.                                              |

## How to Run the Project
- Create a Python virtual environment and activate it.
- Run `pip install -r requirements.txt` to install the necessary packages.
- Create a copy of `llm-request-code/.env.example` to the same directory and rename it as `.env`.
- Fill the environment variables. If you want to configure the code to work with a llama model instead of GPT from OpenAI, analyze the code and do the necessary changes.
- Run the project in 2 phases. The run phase and batch to results phase. Below, there are VS Code debugging configurations for 2 phases (wait after the run phase for OpenAI to finish the batch):

```
{
    "name": "Run Phase",
    "type": "python",
    "request": "launch",
    "program": "${workspaceFolder}/llm-request-code/main.py",
    "console": "integratedTerminal",
    "justMyCode": true,
    "args": ["run"]
},
{
    "name": "Batch To Results Phase",
    "type": "python",
    "request": "launch",
    "program": "${workspaceFolder}/llm-request-code/main.py",
    "console": "integratedTerminal",
    "justMyCode": true,
    "args": ["batchtoresults"]
}
```
