# VoxBank Git Setup Script
# Run this script after installing Git and creating a GitHub repository

Write-Host "VoxBank Git Setup" -ForegroundColor Green
Write-Host "==================" -ForegroundColor Green
Write-Host ""

# Check if git is available
try {
    $gitVersion = git --version
    Write-Host "Git found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Git is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Git from: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

# Initialize repository
Write-Host "Initializing git repository..." -ForegroundColor Cyan
git init

# Add all files
Write-Host "Adding all files..." -ForegroundColor Cyan
git add .

# Create initial commit
Write-Host "Creating initial commit..." -ForegroundColor Cyan
git commit -m "Initial commit: VoxBank AI Voice Banking Assistant"

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Create a repository on GitHub: https://github.com/new" -ForegroundColor White
Write-Host "2. Run the following commands (replace YOUR_USERNAME):" -ForegroundColor White
Write-Host "   git remote add origin https://github.com/YOUR_USERNAME/voxbank.git" -ForegroundColor Cyan
Write-Host "   git branch -M main" -ForegroundColor Cyan
Write-Host "   git push -u origin main" -ForegroundColor Cyan
Write-Host ""
Write-Host "Or use GitHub CLI (if installed):" -ForegroundColor Yellow
Write-Host "   gh repo create voxbank --public --source=. --remote=origin --push" -ForegroundColor Cyan

