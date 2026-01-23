<p align="center"> 
    <img src="figures/mirrorbench-logo.png" align="center" height="150" alt="MirrorBench"></img>
</p>

<h2 align="center">Evaluating Realism of User-Proxy Agents</h2>

<div align="center">

[![ArXiv](https://img.shields.io/badge/arxiv-2601.08118-8A2BE2?logoColor=white)](https://arxiv.org/abs/2601.08118)
[![GitHub](https://img.shields.io/badge/github-mirrorbench-ff69b4?logoColor=white)](https://github.com/SAP/mirrorbench)
[![REUSE status](https://api.reuse.software/badge/github.com/SAP/mirrorbench)](https://api.reuse.software/info/github.com/SAP/mirrorbench)
[![License](https://img.shields.io/badge/license-Apache%202.0-orange?logoColor=white)](./LICENSE)
[![All Contributors](https://img.shields.io/github/all-contributors/SAP/mirrorbench?color=ff69b4)](#contributors)
<!-- [![Python version](https://img.shields.io/pypi/v/mirrorbench?color=5B5BD6)](https://pypi.org/project/mirrorbench/) -->
</div>

MirrorBench is an automatic, extensible Framework to Evaluate User-Proxy Agents for Human-Likeness. It provides a modular architecture to benchmark different User-Proxy Agents against a variety of realism metrics. MirrorBench is designed to be extensible, allowing researchers and developers to bring their own agents and metrics into the framework.

⭐ Drop a star to help us grow!

## Requirements and Setup

The project requires Python 3.12 or higher. It is recommended to use a virtual environment to manage dependencies. You can install the project as a dependency using pip:

```bash
pip install https://github.com/SAP/mirrorbench.git
```

Alternatively, you can install it in editable/development mode by cloning the repository and installing it locally:

```bash
git clone https://github.com/SAP/mirrorbench.git

cd mirrorbench
pip install -e .[dev]
```

## Quick Start

To get started with benchmarking your User-Proxy Agents, you can either use the code or [CLI](#mirrorbench-cli).

In order to run a benchmark, you need to define a job configuration in a YAML file. Below is an example of a simple job configuration:

```yaml
# Job run settings (seed, sync/async, concurrency, cache, observability etc.)
run:
  name: my_run
  ...(trimmed for brevity)...

# Define User-Proxies to benchmark
user_proxies:
- name: proxy:langchain/claude-3.7-sonnet
  ...(trimmed for brevity)...

# Define datasets to use for benchmarking
datasets:
- name: dataset:jsonl/chatbot_arena_mirror
  ...(trimmed for brevity)...

# Define metrics
metrics:
- name: metric:judge/gteval
  ...(trimmed for brevity)...

task_drivers:
  dataset:jsonl/chatbot_arena_mirror:
    driver: task:mirror/conversation
    ...(trimmed for brevity)...
```

As shown above, the job configuration consists of several sections, including `run`, `user_proxies`, `datasets`, `metrics`, and `task_drivers`. Each section allows you to specify the components of your benchmark. You can find more examples of job configurations in the [`configs`](./configs/) directory.

We provide a quick code snippet to run a benchmark using the above job configuration:

```python
from mirrorbench.core.config import load_job_config
from mirrorbench.core.runner import Runner

job_cfg = load_job_config("path/to/your/job_config.yaml")
runner = Runner(job_cfg)
result_summary = runner.run()
```

## LLM Usage

To use LLMs or any external API services, you will most likely need to set up and use API keys or authentication tokens. The package by default accesses environment variables addedn in `.env` file in your working directory. Alternatively, you can set the environment variables directly in your system using the following code snippet:

```python
import os

os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
```

The package has built-in support for Langchain based LLM clients. In case you would like to support other LLM clients, you can implement and register a custom LLM wrapper as shown for [`LangChainChatClient`](mirrorbench/model_clients/langchain/chat.py).

## MirrorBench CLI

MirrorBench provides a command-line interface (CLI) to facilitate running benchmarks, managing runs & cache as well as validating job configs. Below, we provide an overview of the available commands. For detailed usage instructions, you can run `mirrorbench --help`.

### `mirrorbench plan`

The `mirrorbench plan` command allows you to inspect and validate your job configuration file before executing a benchmarking job. It generates a summary file `plan.json` consisting of the components defined in the job configuration, including User-Proxies, datasets, metrics, and task drivers.

```bash
mirrorbench plan -c path/to/your/job_config.yaml
```

### `mirrorbench dryrun`

The `mirrorbench dryrun` command allows you to perform a dry run with credential checks and dependency validation without actual execution of benchmarking tasks. As a result, it generates a `manifest.json` file containing detailed parsed information (units and episodes) which would be executed in a real run.

```bash
mirrorbench dryrun -c path/to/your/job_config.yaml
```

### `mirrorbench run`

This command executes or resumes a benchmarking job based on the provided job configuration file. It manages the execution of tasks, computes metrics, and aggregates results.

```bash
# Execute a job from scratch
mirrorbench run -c path/to/your/job_config.yaml

# Resume a previously interrupted job
mirrorbench run -c path/to/your/job_config.yaml --resume
```

### `mirrorbench report`

The CLI command `mirrorbench report` generates a comprehensive report of the benchmarking results from a completed run.

```bash
# Currently only JSON report generation is supported
mirrorbench report json <run-id> --output path/to/output/report.json
```

### `mirrorbench runs`

The `mirrorbench runs` command has multiple subcommands to manage and inspect previous benchmarking runs. You can list all runs, view details of a specific run, or delete runs.

```bash
# List all previous runs
mirrorbench runs list

# Inspect the output of a specific episode of a run
mirrorbench runs inspect <run_id> --index <episode-index> --output episode.json

# Delete an existing run
mirrorbench runs delete <run_id> --force
```

### `mirrorbench cache`

This command provides subcommands to check statistics of the cache or clear the cache.

```bash
# Show cache statistics
mirrorbench cache stats

# Clear the cache
mirrorbench cache purge
```

Cache is by default retained for 24 hours unless specified otherwise in the job configuration.

## Support, Feedback, Contributing

This project is open to feature requests/suggestions, bug reports etc. via [GitHub issues](https://github.com/SAP/mirrorbench/issues). Contribution and feedback are encouraged and always welcome. For more information about how to contribute, the project structure, as well as additional contribution information, see our [Contribution Guidelines](CONTRIBUTING.md).

## Security / Disclosure
If you find any bug that may be a security problem, please follow our instructions at [in our security policy](https://github.com/SAP/mirrorbench/security/policy) on how to report it. Please do not create GitHub issues for security-related doubts or problems.

## Code of Conduct

We as members, contributors, and leaders pledge to make participation in our community a harassment-free experience for everyone. By participating in this project, you agree to abide by its [Code of Conduct](https://github.com/SAP/.github/blob/main/CODE_OF_CONDUCT.md) at all times.

## Licensing

Copyright 2025 SAP SE or an SAP affiliate company and mirrorbench contributors. Please see our [LICENSE](LICENSE) for copyright and license information. Detailed information including third-party components and their licensing/copyright information is available [via the REUSE tool](https://api.reuse.software/info/github.com/SAP/mirrorbench).

## Citation

If you like our work and find MirrorBench useful in your research, please consider citing the following paper:

```
@misc{hathidara2026mirrorbenchextensibleframeworkevaluate,
      title={MirrorBench: An Extensible Framework to Evaluate User-Proxy Agents for Human-Likeness}, 
      author={Ashutosh Hathidara and Julien Yu and Vaishali Senthil and Sebastian Schreiber and Anil Babu Ankisettipalli},
      year={2026},
      eprint={2601.08118},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2601.08118}, 
}
```

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://ashutoshhathidara.com"><img src="https://avatars.githubusercontent.com/u/20843596?v=4?s=100" width="100px;" alt="Ashutosh Hathidara"/><br /><sub><b>Ashutosh Hathidara</b></sub></a><br /><a href="#research-ashutosh1919" title="Research">🔬</a> <a href="#code-ashutosh1919" title="Code">💻</a> <a href="#design-ashutosh1919" title="Design">🎨</a> <a href="#ideas-ashutosh1919" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-ashutosh1919" title="Maintenance">🚧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/sebastian-schreiber-sap"><img src="https://avatars.githubusercontent.com/u/153761180?v=4?s=100" width="100px;" alt="sebastian-schreiber-sap"/><br /><sub><b>sebastian-schreiber-sap</b></sub></a><br /><a href="#ideas-sebastian-schreiber-sap" title="Ideas, Planning, & Feedback">🤔</a> <a href="#mentoring-sebastian-schreiber-sap" title="Mentoring">🧑‍🏫</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/vaishaalli224"><img src="https://avatars.githubusercontent.com/u/219760723?v=4?s=100" width="100px;" alt="Vaishali Senthil"/><br /><sub><b>Vaishali Senthil</b></sub></a><br /><a href="#ideas-vaishaalli224" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/aanilbabu"><img src="https://avatars.githubusercontent.com/u/6667247?v=4?s=100" width="100px;" alt="aanilbabu"/><br /><sub><b>aanilbabu</b></sub></a><br /><a href="#ideas-aanilbabu" title="Ideas, Planning, & Feedback">🤔</a> <a href="#mentoring-aanilbabu" title="Mentoring">🧑‍🏫</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/yyu56253"><img src="https://avatars.githubusercontent.com/u/76017044?v=4?s=100" width="100px;" alt="Yue (Julien) Yu"/><br /><sub><b>Yue (Julien) Yu</b></sub></a><br /><a href="#ideas-yyu56253" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
