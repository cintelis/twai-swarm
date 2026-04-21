# Templates

A template is a pre-designed project scaffold that the Coder agent forks as a starting point for a workflow, instead of generating every file from scratch.

**Architect picks the template.** The Architect's `tech_choices` output includes a `template` field naming one of the templates below (or `null` for generate-from-scratch).

**Coder customises it.** The Coder copies `scaffold/` into the sandbox filesystem, then modifies files per the SE plan. Most files stay identical; the Coder touches only the ones specific to the customer's brief.

**Verify with `verify.sh`.** After customisation, the Coder runs `verify.sh` inside the sandbox. Non-zero exit = broken; the Coder iterates until it passes.

---

## Template contract

Every template lives in `templates/<name>/` with this shape:

```
templates/python-fastapi-postgres/
├── template.json              # metadata — see schema below
├── verify.sh                  # verification script run after customisation
└── scaffold/                  # the actual files copied into the workflow sandbox
    ├── README.md
    ├── pyproject.toml
    ├── …
```

### `template.json` schema

```jsonc
{
  "name": "python-fastapi-postgres",
  "display_name": "Python + FastAPI + Postgres",
  "description": "One-sentence summary shown to humans choosing a template.",
  "language": "python | typescript | java | go | rust | …",
  "framework": "fastapi | nextjs | spring-boot | …",
  "runtime_version": "python 3.12+",

  "included": [
    "Human-readable bullets describing what's pre-wired",
    "One bullet per notable piece"
  ],

  "selection_hints": [
    "Phrases the Architect should look for in the brief that make THIS template a good fit",
    "e.g. 'REST API', 'backend service with database', 'Python is the chosen language'"
  ],

  "anti_hints": [
    "Phrases that should STEER AWAY from this template",
    "e.g. 'frontend-only', 'no database', 'serverless-first'"
  ],

  "verification": {
    "script": "verify.sh",
    "description": "What must be true for the customised scaffold to pass."
  },

  "customisation_guide": {
    "primary_files": [
      "Files the Coder is EXPECTED to modify — with a one-line hint for each"
    ],
    "preserve": [
      "Files the Coder should leave alone unless the brief explicitly requires a change"
    ]
  }
}
```

### `verify.sh` contract

- Runs in a working directory that IS `scaffold/` (customised copy).
- Runs inside the sandbox container, which has the language toolchain installed.
- Must exit `0` if the scaffold is valid, non-zero otherwise.
- Should print compact diagnostics on failure — the Coder reads the output to decide what to fix.
- Should cover at minimum: lint, type-check (if applicable), and one smoke-level test.
- **Should NOT** require network access beyond language package managers (pip, npm, maven). The sandbox may be egress-restricted.

Keep `verify.sh` fast (<60s target). Slow verification starves the iterate-fix loop of budget.

---

## How a workflow uses a template

```
BA → Researcher → Architect
                     │
                     └── tech_choices includes: {"template": "python-fastapi-postgres"}
                     │
                  … ──▶ Coder
                           │
                           ├── Read templates/python-fastapi-postgres/template.json
                           ├── Copy scaffold/* to sandbox /workspace
                           ├── Modify files per SE plan (LLM tool calls)
                           ├── Run verify.sh
                           ├── If fail: read output, fix, re-run
                           └── If pass: zip /workspace, upload
```

---

## Current catalogue

| Name | Language | Framework | Status |
|---|---|---|---|
| `python-fastapi-postgres` | Python 3.12+ | FastAPI + async SQLAlchemy + Alembic | ✓ Reference implementation |
| `nextjs-ts-prisma` | TypeScript | Next.js 15 + Prisma + Tailwind | planned |
| `spring-boot-jpa` | Java 21 | Spring Boot 3 + JPA + PostgreSQL | planned |
| `vite-react-tailwind` | TypeScript | Vite + React + Tailwind + Zustand | planned |

---

## Portability

Templates are designed to be **language-agnostic to the swarm that uses them**. A template's scaffold + verify.sh + metadata work identically whether the agent loop runs in Python (current `twai-swarm`) or Java (co-founder's implementation). The Architect's `template` field is a string; the Coder is whatever language loads it. This is the integration point to standardise between implementations.

## Adding a new template

1. Create `templates/<name>/` with the files above.
2. Fill in `template.json`. Be honest in `selection_hints` and `anti_hints` — vague hints mean the Architect picks the wrong template.
3. Write `scaffold/` as a working, tested project. It should pass `verify.sh` in its pristine, un-customised state.
4. Run `bash templates/<name>/verify.sh` from inside `templates/<name>/scaffold/` to confirm.
5. Keep it SMALL — 15-30 files is the sweet spot. More = more to break during customisation.

## Anti-patterns

- ❌ **Templates that only work with specific AWS services.** Keep them cloud-agnostic unless you're building a cloud-specific template family.
- ❌ **Templates with no tests.** `verify.sh` has nothing to run against.
- ❌ **Templates that pin heavy ML deps (torch, tensorflow) by default.** Makes pristine verification slow; add them as optional extras.
- ❌ **Templates that take >2 min to set up.** Iteration feels sluggish; move heavy deps to opt-in.
