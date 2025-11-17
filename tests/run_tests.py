import sys
import subprocess

def run_tests():
    result = subprocess.run(
        ['pytest', 'tests/doc_tool/', '-v', '--tb=short'],
        capture_output=False
    )
    sys.exit(result.returncode)

if __name__ == "__main__":
    run_tests()