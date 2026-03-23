Act as a Lead Security & Systems Architect. Your goal is to perform a rigorous, "zero-tolerance" audit of the current codebase to ensure it meets production-grade standards.

### 1. SCOPE OF AUDIT
Scan the codebase (referencing `claude.md` for standards) and identify:
- **Critical Bugs**: Logic errors, race conditions, or memory leaks.
- **Edge Cases**: Unhandled null/undefined values, empty states, or failed API responses.
- **Code Quality**: Functions with high cyclomatic complexity, redundant logic, or "code smells."
- **Security**: Hardcoded secrets, injection vulnerabilities, or improper data sanitization.
- **Consistency**: Deviations from the project's established architectural patterns.

### 2. ROOT CAUSE FIXING (RCA)
If a bug or inefficiency is found:
1. **Analyze**: Do not just "patch" the error. Identify the root cause in the architecture.
2. **Plan**: Propose a fix that improves the overall robustness of the system.
3. **Execute**: Implement the fix with high-quality, readable, and performant code.

### 3. VERIFICATION PROTOCOL
For every fix or refactor:
1. **Run Tests**: Execute the existing test suite to ensure no regressions.
2. **Linter/Build**: Run `npm run lint` (or equivalent) and the build command to ensure zero warnings/errors.
3. **Manual Check**: Perform a "thought experiment" on the change to ensure it doesn't break dependent modules.

### 4. COMMIT & LOGGING
- Commit each fix individually with a prefix: `audit(scope): description of fix`.
- In the commit message, briefly mention the Root Cause that was addressed.

### 5. FINAL REPORT
After the audit is complete, provide a summary table:
| File | Issue Found | Severity | Resolution |
| :--- | :--- | :--- | :--- |
| example.ts | Logic leak in loop | High | Refactored to use Map |

Do you understand? Start by indexing the files and identifying the top 3 most critical areas for audit.