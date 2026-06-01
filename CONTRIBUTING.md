# Contributing to ETH-TAD

Contributions are welcome! Please follow these guidelines.

## Getting Started

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Open a pull request

## Code Style

- Follow PEP 8 for Python code
- Keep functions focused and documented
- All reusable functions belong in `functions/`, not in notebooks

## Notebooks

- Notebooks are controllers only — no function definitions inline
- The configuration cell should be the only cell users need to edit
- Clear all outputs before committing (`jupyter nbconvert --clear-output --inplace *.ipynb`)

## Reporting Issues

Please open a GitHub issue with:
- A clear description of the problem
- The notebook and cell where it occurs
- The full error traceback
- Your Python version and OS
