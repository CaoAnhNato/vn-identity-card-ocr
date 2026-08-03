# Workspace Rules for vn-identity-card-ocr

## Mandatory Step-by-Step Workflow for Implementation and Deployments
1. **Always run a test first** to evaluate any code edits or new scripts (e.g. running `--test` mode or a quick test case).
2. **Report the test results** back to the user clearly.
3. **Wait for the user's explicit confirmation** before committing, pushing, or implementing/deploying the final code.

## Mandatory Jupyter Notebook Rule
- All operations, execution, modifications, or checks related to Jupyter Notebook files (`.ipynb`) **MUST strictly and exclusively use the native Jupyter MCP tools** (e.g., `use_notebook`, `read_notebook`, `insert_cell`, `execute_cell`, `overwrite_cell_source`, etc.).
- Never use python CLI, shell commands (such as `nbconvert` or custom scripts) or any other external tools to manipulate or execute notebook files.
