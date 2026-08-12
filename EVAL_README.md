# Address-Parser accuracy eval — setup for you

Thanks for helping run this! It just needs Ollama installed (my machine is too slow for it).
Copy-paste these, in order:

```bash
# 1. Install Ollama from https://ollama.com if you don't have it, then pull the model:
ollama pull mistral

# 2. Get the code
git clone https://github.com/Apoorv012/Address-Parser
cd Address-Parser
git checkout evaluation

# 3. Set up a virtual environment and install dependencies
python -m venv venv
venv\Scripts\activate        # Windows — use "source venv/bin/activate" on Mac/Linux
pip install -r requirements.txt

# 4. Run the eval (takes a few minutes — ~64 Ollama calls total)
python scripts/eval_rag_pipeline.py --model mistral
python scripts/eval_full_pipeline.py --model mistral
```

That's it. Step 4 prints a report to the console and also writes two files:
- `eval/results_rag.json`
- `eval/results_full.json`

**Please just send me those two `.json` files back** (zip them or drag them into Slack/email/
WhatsApp, whatever's easiest) — that's all I need, nothing else to run.
