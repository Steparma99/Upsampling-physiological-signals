import sys
from pathlib import Path

# Add project root to sys.path so that 'src.*' imports work regardless of
# where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).parents[1]))
