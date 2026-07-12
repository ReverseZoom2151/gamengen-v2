# Contributing to GameNGen

Thank you for your interest in contributing to GameNGen! This document provides guidelines for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)

---

## Code of Conduct

This project follows a Code of Conduct that all contributors are expected to adhere to. Please be respectful and constructive in all interactions.

**Expected behavior:**

- Be welcoming and inclusive
- Respect differing viewpoints
- Accept constructive criticism gracefully
- Focus on what's best for the community

---

## How Can I Contribute?

### Reporting Bugs

**Before submitting a bug report:**

- Check existing issues to avoid duplicates
- Test on the latest version
- Gather relevant information (error messages, environment details)

**Submit bugs using our issue template** with:

- Clear, descriptive title
- Steps to reproduce
- Expected vs. actual behavior
- Environment details (GPU, PyTorch version, etc.)
- Error messages and logs

### Suggesting Enhancements

We welcome feature requests and improvements!

**Good enhancement proposals include:**

- Clear use case
- Expected behavior
- Why this would be valuable
- Potential implementation approach

### Contributing Code

Areas where contributions are especially welcome:

- **Bug fixes** - Fix issues in existing code
- **Documentation** - Improve guides and examples
- **New games** - Add environment wrappers
- **Metrics** - Implement additional evaluation metrics
- **Optimizations** - Performance improvements
- **Tests** - Increase test coverage
- **Features** - Text conditioning, longer context, etc.

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/gameNgen-v2.git
cd gameNgen-v2
```

### 2. Install Dependencies

```bash
# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Install project dependencies
pip install -e '.[dev]'

# Install development dependencies
pip install -r requirements-dev.txt  # If available

# Install in development mode
pip install -e .
```

### 3. Verify Setup

```bash
# Run tests
python tests/test_all_tiers.py

# Should see:
# [READY] Tier 1: Chrome Dino
# [READY] Tier 2: DOOM Lite
# [READY] Tier 3: Full DOOM
```

### 4. Create Branch

```bash
git checkout -b feature/your-feature-name
```

---

## Coding Standards

### Python Style

- Follow **PEP 8** style guide
- Use **type hints** where possible
- Maximum line length: **100 characters**
- Use **meaningful variable names**

**Example:**

```python
def train_model(
    config: dict,
    data_dir: str,
    num_steps: int = 1000
) -> dict:
    """
    Train diffusion model.

    Args:
        config: Configuration dictionary
        data_dir: Path to training data
        num_steps: Number of training steps

    Returns:
        Training metrics dictionary
    """
    # Implementation
    pass
```

### Documentation

- **All functions** should have docstrings
- Use **Google-style docstrings**
- Include **type hints**
- Explain **why**, not just **what**

### Code Organization

- Keep functions **focused and small**
- Use **descriptive names**
- **Avoid** deep nesting (max 3-4 levels)
- **Extract** complex logic into helper functions

### Testing

- **Write tests** for new features
- Ensure existing tests **still pass**
- Aim for **meaningful coverage**, not just high percentage

```bash
# Run tests before committing
python tests/test_all_tiers.py
```

---

## Commit Guidelines

### Commit Messages

Follow **Conventional Commits** format:

```text
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**

```text
feat(diffusion): Add text conditioning support

Implements CLIP-based text conditioning for game generation.
Allows users to specify game content via text prompts.

Closes #42
```

```text
fix(agent): Correct reward calculation in DQN

Fixed off-by-one error in reward accumulation that caused
training instability.
```

```text
docs(readme): Update installation instructions

Added troubleshooting section for common CUDA issues.
```

### Commit Best Practices

- **One logical change** per commit
- Write **clear, descriptive** messages
- Reference **issue numbers** where applicable
- Keep commits **focused and atomic**

---

## Pull Request Process

### 1. Before Submitting

**Checklist:**

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] Branch is up to date with main

```bash
# Update your branch
git fetch upstream
git rebase upstream/main

# Run tests
python tests/test_all_tiers.py

# Check formatting (if using pre-commit)
pre-commit run --all-files
```

### 2. Create Pull Request

**Title:** Clear, descriptive summary

**Description should include:**

- **What** does this PR do?
- **Why** is this change needed?
- **How** does it work?
- **Testing** performed
- **Related issues** (if any)

**PR Template:**

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed code
- [ ] Commented complex code
- [ ] Updated documentation
- [ ] Added tests
- [ ] Tests pass locally
```

### 3. Review Process

- Maintainers will review your PR
- Address any feedback
- Make requested changes
- Once approved, we'll merge!

**Be patient:**

- Reviews may take a few days
- Feedback is meant to improve code quality
- Multiple review rounds are normal

---

## Development Workflow

### Typical Contribution Flow

```bash
# 1. Update main branch
git checkout main
git pull upstream main

# 2. Create feature branch
git checkout -b feature/my-feature

# 3. Make changes
# ... edit code ...

# 4. Test changes
python tests/test_all_tiers.py

# 5. Commit changes
git add .
git commit -m "feat: Add my feature"

# 6. Push to your fork
git push origin feature/my-feature

# 7. Create PR on GitHub
```

### Working with Multiple Commits

If your PR has multiple commits, we may ask you to:

- **Squash commits** into logical units
- **Rebase** instead of merge
- **Clean up** commit history

```bash
# Interactive rebase to clean up commits
git rebase -i HEAD~3  # Last 3 commits
```

---

## Testing Guidelines

### Running Tests

```bash
# Run all tests
python tests/test_all_tiers.py

# Run specific tests
python tests/test_diffusion_simple.py

# Run with coverage (if installed)
pytest --cov=src tests/
```

### Writing Tests

**Location:** Place tests in `tests/` directory

**Naming:** `test_<module>_<function>.py`

**Structure:**

```python
def test_my_feature():
    """Test description"""
    # Setup
    model = create_model()

    # Execute
    result = model.do_something()

    # Verify
    assert result == expected_value
    print("✓ Test passed")
```

---

## Questions?

- **Check existing issues** - Your question may be answered
- **Open a discussion** - For general questions
- **Ask in PR** - For code-specific questions

---

## Recognition

Contributors will be:

- Listed in release notes
- Mentioned in documentation
- Credited appropriately
