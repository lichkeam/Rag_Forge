# Python code formatting script

echo "=== Starting Python code formatting ==="

# 1. Remove unused imports and variables
echo "Step 1: Removing unused imports..."
autoflake --in-place \
  --remove-all-unused-imports \
  --remove-unused-variables \
  --recursive .
  
# 2. Auto-format with autopep8
echo "Step 2: Formatting code with autopep8..."
autopep8 --in-place \
  --aggressive \
  --aggressive \
  --recursive .

# 3. Final formatting with Black (optional)
echo "Step 3: Final formatting with Black..."
black .

echo "=== Formatting complete! ==="
echo "Run 'git diff' to see changes"