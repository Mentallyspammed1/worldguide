# Custom Style Guide for Gemini Code Review

This guide helps Gemini focus its review comments towards our project's standards.

## General Principles
- **Clarity over Cleverness:** Code should be straightforward and easy for others (and future you) to understand. Avoid overly complex or obscure constructs if simpler alternatives exist.
- **Consistency:** Adhere to existing project patterns, naming conventions, and architectural choices. New code should feel like it belongs.
- **DRY (Don't Repeat Yourself):** Avoid copy-pasting code blocks. Use functions, classes, or other abstractions to promote reuse.
- **Focused Changes:** Pull requests should ideally represent a single logical change or feature.
- **Tested Code:** New logic or bug fixes should be accompanied by relevant unit or integration tests. (Gemini primarily reviews the diff, but can sometimes spot missing test coverage).

## Python Specifics (If applicable)
- **PEP 8:** Follow standard Python style guidelines (e.g., via linters like Flake8, Black, Ruff). Pay attention to line length, naming conventions (snake_case for functions/variables, PascalCase for classes), imports order, and whitespace.
- **Type Hinting:** Use type hints (Python 3.6+) for function signatures (arguments and return types) to improve code clarity and enable static analysis.
- **Docstrings:** Provide clear and concise docstrings for all public modules, classes, functions, and methods (e.g., following Google, NumPy, or reStructuredText style). Explain the *what* and *why*, not just the *how*.
- **Error Handling:** Use specific exception types rather than broad `except Exception:`. Handle potential errors gracefully and provide informative error messages. Log errors appropriately.
- **Resource Management:** Use `with` statements for managing resources like files, network connections, locks, etc., to ensure they are properly released even if errors occur.
- **List Comprehensions/Generator Expressions:** Prefer these over `map()`/`filter()` or simple `for` loops when they improve readability and conciseness for creating lists or iterables.
- **Logging:** Use the `logging` module for application logging instead of `print()` statements for better control over levels and output destinations.

## Security Focus (Critical)
- **Input Validation/Sanitization:** Treat ALL external input (user input, API responses, file content, environment variables) as potentially malicious. Validate formats, ranges, and types. Sanitize data appropriately before using it in queries, commands, or HTML output (prevent XSS, Injection).
- **Secrets Management:** **ABSOLUTELY NO** hardcoded API keys, passwords, certificates, or other sensitive credentials in the codebase. Use GitHub Secrets, environment variables injected securely, or a dedicated secrets management service.
- **Least Privilege:** Ensure code runs with the minimum necessary permissions. File permissions and access controls should be appropriately restrictive.
- **Dependency Security:** Keep dependencies up-to-date to patch known vulnerabilities. Check if the PR introduces new dependencies or updates existing ones, considering their security implications. Use tools like `pip-audit` or GitHub Dependabot alerts.
- **Authentication & Authorization:** Verify that changes correctly implement or respect authentication and authorization checks.

## Performance Considerations
- **Algorithmic Efficiency:** Be mindful of algorithm complexity (e.g., avoid O(n^2) loops if O(n) or O(n log n) is feasible).
- **Database Queries:** Avoid N+1 query problems in ORMs. Fetch data efficiently using appropriate joins or prefetching. Ensure database queries are indexed where necessary.
- **Resource Usage:** Be mindful of memory consumption and CPU usage, especially in loops or data processing tasks. Avoid blocking operations in asynchronous code.
- **Caching:** Consider caching for expensive computations or frequently accessed, rarely changing data where appropriate.

