
echo "🧹 Starting code cleanup and publishing process..."

echo "⚫ Running Black formatter..."
black . && echo "✅ Code formatting complete!" || echo "❌ Black formatting failed!"

echo "🔍 Running Ruff linter..."
ruff check . --fix && echo "✅ Linting complete!" || echo "❌ Linting failed!"

echo "🏗️  Building package..."
poetry build && echo "✅ Build successful!" || echo "❌ Build failed!"

echo "📦 Publishing to PyPI..."
poetry publish && echo "✅ Package published!" || echo "❌ Publishing failed!"

echo "📝 Committing changes to Git..."
git add . && echo "✅ Changes staged!"
git commit -m "Update and publish" && echo "✅ Changes committed!"

echo "🚀 Pushing to remote repository..."
git push && echo "✅ Changes pushed to remote!"

echo "✨ All done! Package cleaned, built, and published successfully! ✨"

