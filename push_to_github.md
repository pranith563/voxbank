# Instructions to Push to GitHub

## Prerequisites
1. Install Git from https://git-scm.com/download/win
2. Create a new repository on GitHub (https://github.com/new)
   - Name it: `voxbank` (or your preferred name)
   - Do NOT initialize with README, .gitignore, or license (we already have these)

## Steps to Push

Open PowerShell or Git Bash in this directory and run:

```bash
# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: VoxBank AI Voice Banking Assistant"

# Add your GitHub repository as remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/voxbank.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

## If you need to authenticate

If prompted for credentials:
- Use a Personal Access Token (not your password)
- Create one at: https://github.com/settings/tokens
- Select scope: `repo` (full control of private repositories)

## Alternative: Using GitHub CLI

If you have GitHub CLI installed:

```bash
gh repo create voxbank --public --source=. --remote=origin --push
```

