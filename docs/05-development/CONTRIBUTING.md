# Contributing to aArabic

Thank you for your interest in contributing to aArabic!

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/plturrell/aArabic.git
   cd aArabic
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

5. Copy environment file:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

## Code Style

- Follow PEP 8 for Python code
- Use `black` for formatting: `black backend/ tests/`
- Use `ruff` for linting: `ruff check backend/`
- Add type hints where possible

## Testing

- Write tests for new features
- Run tests: `pytest`
- Aim for >80% code coverage

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Run tests and linting
4. Commit with clear messages
5. Push to your fork
6. Create a pull request

## Project Structure

- `backend/` - Python backend code
- `frontend/` - Frontend HTML/JS
- `tests/` - Test files
- `docs/` - Documentation

See `README.md` for more details.

