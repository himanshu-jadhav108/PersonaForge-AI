# Contributing to PersonaForge AI

Thank you for your interest in contributing to **PersonaForge AI**! We welcome contributions from the community to make local-first, privacy-preserving face transformation faster, more robust, and higher quality.

---

## 1. Code of Conduct

We are committed to providing a welcoming, inclusive, and harassment-free environment for everyone. Please be respectful, constructive, and ethical in all interactions and contributions.

---

## 2. Development Setup

1. **Fork and Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/PersonaForge-AI.git
   cd PersonaForge-AI
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\Activate.ps1
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install Runtime & Developer Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

---

## 3. Testing & Code Quality Standards

All pull requests must pass the complete test suite and code style standards before merging.

### Running the Test Suite
```bash
pytest tests/ -v -p no:cacheprovider
```

### Running the Linter & Formatter
We use [Ruff](https://github.com/astral-sh/ruff) for fast, rigorous linting:
```bash
ruff check .
ruff format --check .
```

To automatically fix formatting or autofixable lint errors:
```bash
ruff check --fix .
ruff format .
```

---

## 4. Commit Message Guidelines

We adhere to the [Conventional Commits](https://www.conventionalcommits.org/) specification:

* `feat(scope): ...` — New user-facing feature or enhancement.
* `fix(scope): ...` — Bug fix or correction.
* `docs(scope): ...` — Documentation updates.
* `test(scope): ...` — Adding or updating test cases.
* `refactor(scope): ...` — Code improvement without functional changes.
* `perf(scope): ...` — Performance optimizations.

Examples:
* `feat(restoration): add GFPGAN v1.4 crop adapter with identity drift guard`
* `fix(pipeline): prevent division by zero in landmark jitter calculation`
* `docs(gpu): add CUDA 12.4 verification troubleshooting steps`

---

## 5. Architectural Safeguards

When contributing to PersonaForge AI, please respect our project safeguards:
1. **Model Transparency**: Any new model integration must include upstream license, source, and limitations in `docs/model-registry.md`.
2. **Restoration Transparency**: AI models and classic fallbacks must report their status explicitly; never silently replace AI restoration with classic filters.
3. **No Simulated Benchmarks**: Telemetry and benchmarks must represent true measured CPU/GPU execution metrics.
4. **Human-Centered UX**: Use friendly terminology (`Detected Person 1..N`) rather than raw array indices or machine jargon in user-facing components.
