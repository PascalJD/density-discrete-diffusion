## Contributing Guidelines

Thank you for contributing to this repository! Please follow these guidelines:

### Committing
Below is the general format for commit log messages:

```
ABBR: Commit message.
```

**ABBR options:**

- `IMP`: Development and implementation of a new feature.
- `FIX`: A fix of an existing bug.
- `CFG`: Changes to configuration.
- `TST`: Modifications to tests.
- `OPT`: Any optimization performed.
- `REF`: Any refactors to code.

### Branching

Create and switch to a new branch:

```bash
git checkout -b <my-branch> main
``` 

Branches should be clearly named based on their purpose (e.g., feature/new-algorithm, fix/boundary-condition).

### Pull Requests

When submitting a pull request (PR), please ensure the following:
 
- Your PR has a clear and descriptive title.
- Include a concise summary of changes, clearly stating what was changed and why.
- Ensure your code passes all existing tests.
- Add relevant tests for new features or bug fixes.