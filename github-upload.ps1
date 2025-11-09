# PowerShell script to upload Urdu OCR project to GitHub

Write-Host "🚀 Urdu OCR - GitHub Upload Helper" -ForegroundColor Green
Write-Host "==================================" -ForegroundColor Green

# Check if git is installed
try {
    git --version | Out-Null
    Write-Host "✅ Git is installed" -ForegroundColor Green
} catch {
    Write-Host "❌ Git is not installed. Please install Git first." -ForegroundColor Red
    Write-Host "   Download from: https://git-scm.com/download/win"
    exit 1
}

# Check if git is initialized
if (-not (Test-Path ".git")) {
    Write-Host "📦 Initializing Git repository..." -ForegroundColor Yellow
    git init
    Write-Host "✅ Git repository initialized" -ForegroundColor Green
} else {
    Write-Host "✅ Git repository already exists" -ForegroundColor Green
}

# Configure git user (if not set)
$gitUser = git config user.name
$gitEmail = git config user.email

if (-not $gitUser) {
    $userName = Read-Host "Enter your name for Git commits"
    git config user.name "$userName"
}

if (-not $gitEmail) {
    $userEmail = Read-Host "Enter your email for Git commits"
    git config user.email "$userEmail"
}

# Add all files except those in .gitignore
Write-Host "📁 Adding files to Git..." -ForegroundColor Yellow
git add .

# Check if there are any changes to commit
$status = git status --porcelain
if (-not $status) {
    Write-Host "ℹ️ No changes to commit" -ForegroundColor Blue
} else {
    # Create commit
    Write-Host "💾 Creating commit..." -ForegroundColor Yellow
    $commitMsg = Read-Host "Enter commit message (or press Enter for default)"
    if ([string]::IsNullOrEmpty($commitMsg)) {
        $commitMsg = "Initial commit: Complete Urdu OCR Neural Network System"
    }
    git commit -m "$commitMsg"
    Write-Host "✅ Commit created" -ForegroundColor Green
}

# Check if remote origin exists
$remoteUrl = git remote get-url origin 2>$null
if (-not $remoteUrl) {
    # Add remote repository
    Write-Host "🔗 Setting up GitHub remote..." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "📋 Steps to create a GitHub repository:" -ForegroundColor Cyan
    Write-Host "1. Go to https://github.com/new"
    Write-Host "2. Repository name: urdu-ocr-neural-network"
    Write-Host "3. Description: Complete Urdu OCR system using CNN+LSTM+CTC"
    Write-Host "4. Make it Public or Private"
    Write-Host "5. Don't initialize with README (we already have one)"
    Write-Host "6. Copy the repository URL"
    Write-Host ""
    
    $repoUrl = Read-Host "Enter your GitHub repository URL (e.g., https://github.com/username/urdu-ocr.git)"
    
    if (-not [string]::IsNullOrEmpty($repoUrl)) {
        git remote add origin "$repoUrl"
        Write-Host "✅ Remote repository added" -ForegroundColor Green
        
        # Push to GitHub
        Write-Host "⬆️ Pushing to GitHub..." -ForegroundColor Yellow
        git branch -M main
        try {
            git push -u origin main
            Write-Host "🎉 Successfully uploaded to GitHub!" -ForegroundColor Green
            Write-Host "🌐 Your repository: $repoUrl" -ForegroundColor Cyan
        } catch {
            Write-Host "❌ Push failed. You may need to authenticate with GitHub." -ForegroundColor Red
            Write-Host "   Try: git push -u origin main" -ForegroundColor Yellow
        }
    } else {
        Write-Host "❌ No repository URL provided. Please add remote manually:" -ForegroundColor Red
        Write-Host "   git remote add origin <your-repo-url>" -ForegroundColor Yellow
        Write-Host "   git push -u origin main" -ForegroundColor Yellow
    }
} else {
    Write-Host "🔗 Remote origin already exists: $remoteUrl" -ForegroundColor Green
    Write-Host "⬆️ Pushing to GitHub..." -ForegroundColor Yellow
    try {
        git push
        Write-Host "🎉 Successfully pushed to GitHub!" -ForegroundColor Green
    } catch {
        Write-Host "❌ Push failed. You may need to authenticate." -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "📋 Next steps:" -ForegroundColor Cyan
Write-Host "1. Go to your GitHub repository"
Write-Host "2. Add topics: machine-learning, ocr, urdu, pytorch, deep-learning"
Write-Host "3. Add a repository description"
Write-Host "4. Star your own repository! ⭐"
Write-Host "5. Share with the community"
Write-Host ""
Write-Host "🎊 Your Urdu OCR project is ready for the world!" -ForegroundColor Green

# Show project statistics
Write-Host ""
Write-Host "📊 Project Statistics:" -ForegroundColor Cyan
$pythonFiles = (Get-ChildItem -Recurse -Include "*.py" | Measure-Object).Count
$totalLines = (Get-ChildItem -Recurse -Include "*.py" | Get-Content | Measure-Object -Line).Lines
Write-Host "   Python files: $pythonFiles"
Write-Host "   Lines of code: $totalLines"
Write-Host "   Documentation files: 7"
Write-Host "   Model architecture: CNN + BiLSTM + CTC"
Write-Host "   Parameters: ~10.27 million"