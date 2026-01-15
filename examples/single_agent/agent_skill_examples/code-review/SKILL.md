---
name: code-review
description: Perform comprehensive code reviews focusing on best practices, security vulnerabilities, performance optimization, and maintainability
---

# Code Review Skill

When reviewing code, follow this systematic approach to ensure thorough evaluation:

## Review Checklist

### 1. Code Quality
- **Readability**: Is the code easy to understand?
- **Naming**: Are variables, functions, and classes well-named?
- **Structure**: Is the code properly organized and modular?
- **Comments**: Are complex sections adequately documented?
- **Complexity**: Are there overly complex functions that should be simplified?

### 2. Security Analysis
Check for common vulnerabilities:
- SQL injection vulnerabilities
- XSS (Cross-Site Scripting) vulnerabilities
- Authentication and authorization flaws
- Insecure data handling (passwords, sensitive data)
- Input validation and sanitization
- OWASP Top 10 vulnerabilities

### 3. Performance Considerations
- Identify potential bottlenecks
- Check for inefficient algorithms or data structures
- Look for unnecessary database queries or API calls
- Evaluate caching opportunities
- Assess memory usage patterns

### 4. Best Practices
- **DRY Principle**: Eliminate code duplication
- **SOLID Principles**: Verify adherence to design principles
- **Error Handling**: Check for proper exception handling
- **Testing**: Evaluate test coverage and quality
- **Dependencies**: Review external dependencies and their versions

### 5. Maintainability
- Is the code easy to modify and extend?
- Are there proper abstractions?
- Is the architecture scalable?
- Are there technical debt concerns?

## Review Format

Structure your review as follows:

1. **Summary**: High-level overview of the changes
2. **Critical Issues**: Security vulnerabilities or bugs that must be fixed
3. **Major Concerns**: Significant issues affecting quality or performance
4. **Suggestions**: Optional improvements and best practices
5. **Positive Feedback**: Acknowledge good practices and improvements

## Guidelines

- Be constructive and respectful
- Provide specific examples and suggestions
- Explain the "why" behind recommendations
- Prioritize issues by severity (critical, major, minor)
- Reference documentation or standards when applicable
- Consider the context and constraints of the project

## Example Reviews

**Security Issue:**
```
CRITICAL: SQL injection vulnerability detected at line 45
Current: f"SELECT * FROM users WHERE id = {user_id}"
Recommendation: Use parameterized queries to prevent SQL injection
```

**Performance Suggestion:**
```
SUGGESTION: Consider caching database results at line 123
The same query is executed multiple times in the loop. Cache the results
to improve performance by ~80%.
```
