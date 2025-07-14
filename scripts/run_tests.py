#!/usr/bin/env python3
"""
Test runner script for the Data-Scrape-to-RAG Lakehouse project.

This script provides a convenient way to run different types of tests
with proper environment setup and reporting.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def setup_test_environment():
    """Set up the test environment variables."""
    os.environ['PYTHONPATH'] = str(Path(__file__).parent.parent)
    os.environ['TESTING'] = 'true'
    os.environ['LOG_LEVEL'] = 'WARNING'
    
    # Set test-specific environment variables
    test_env = {
        'MINIO_ENDPOINT': 'localhost:9000',
        'MINIO_ACCESS_KEY': 'minioadmin',
        'MINIO_SECRET_KEY': 'minioadmin',
        'MINIO_BUCKET_NAME': 'test-lakehouse',
        'CHROMA_HOST': 'localhost',
        'CHROMA_PORT': '8000',
        'OLLAMA_HOST': 'localhost',
        'OLLAMA_PORT': '11434',
        'SPARK_MASTER_URL': 'local[2]',
        'AIRFLOW_HOME': '/tmp/airflow_test'
    }
    
    for key, value in test_env.items():
        os.environ.setdefault(key, value)


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        return False


def run_unit_tests(verbose=False, coverage=False):
    """Run unit tests."""
    cmd = ['python', '-m', 'pytest', 'tests/', '-m', 'unit']
    
    if verbose:
        cmd.append('-v')
    
    if coverage:
        cmd.extend(['--cov=src', '--cov-report=html', '--cov-report=term-missing'])
    
    return run_command(cmd, "Unit Tests")


def run_integration_tests(verbose=False):
    """Run integration tests."""
    cmd = ['python', '-m', 'pytest', 'tests/', '-m', 'integration']
    
    if verbose:
        cmd.append('-v')
    
    return run_command(cmd, "Integration Tests")


def run_specific_module_tests(module, verbose=False):
    """Run tests for a specific module."""
    test_file = f"tests/test_{module}.py"
    
    if not Path(test_file).exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    cmd = ['python', '-m', 'pytest', test_file]
    
    if verbose:
        cmd.append('-v')
    
    return run_command(cmd, f"{module.title()} Module Tests")


def run_all_tests(verbose=False, coverage=False):
    """Run all tests."""
    cmd = ['python', '-m', 'pytest', 'tests/']
    
    if verbose:
        cmd.append('-v')
    
    if coverage:
        cmd.extend(['--cov=src', '--cov-report=html', '--cov-report=term-missing'])
    
    return run_command(cmd, "All Tests")


def run_linting():
    """Run code linting."""
    success = True
    
    # Run flake8 if available
    if run_command(['python', '-m', 'flake8', 'src/', 'tests/'], "Flake8 Linting"):
        print("✅ Flake8 linting passed")
    else:
        print("⚠️  Flake8 not available or failed")
        success = False
    
    # Run black check if available
    if run_command(['python', '-m', 'black', '--check', 'src/', 'tests/'], "Black Code Formatting Check"):
        print("✅ Black formatting check passed")
    else:
        print("⚠️  Black not available or formatting issues found")
        success = False
    
    return success


def run_type_checking():
    """Run type checking with mypy."""
    return run_command(['python', '-m', 'mypy', 'src/'], "MyPy Type Checking")


def generate_test_report():
    """Generate a comprehensive test report."""
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE TEST REPORT")
    print("="*60)
    
    cmd = [
        'python', '-m', 'pytest', 'tests/',
        '--cov=src',
        '--cov-report=html:htmlcov',
        '--cov-report=xml:coverage.xml',
        '--cov-report=term-missing',
        '--junit-xml=test-results.xml',
        '--html=test-report.html',
        '--self-contained-html'
    ]
    
    success = run_command(cmd, "Comprehensive Test Report Generation")
    
    if success:
        print("\n📊 Test reports generated:")
        print("  - HTML Coverage Report: htmlcov/index.html")
        print("  - XML Coverage Report: coverage.xml")
        print("  - JUnit XML Report: test-results.xml")
        print("  - HTML Test Report: test-report.html")
    
    return success


def main():
    """Main function to handle command line arguments and run tests."""
    parser = argparse.ArgumentParser(
        description="Test runner for Data-Scrape-to-RAG Lakehouse project"
    )
    
    parser.add_argument(
        'test_type',
        choices=['unit', 'integration', 'all', 'lint', 'type', 'report', 'scraper', 'etl', 'embeddings', 'api', 'dag'],
        help='Type of tests to run'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Run tests in verbose mode'
    )
    
    parser.add_argument(
        '-c', '--coverage',
        action='store_true',
        help='Run tests with coverage reporting'
    )
    
    parser.add_argument(
        '--no-setup',
        action='store_true',
        help='Skip test environment setup'
    )
    
    args = parser.parse_args()
    
    # Setup test environment unless skipped
    if not args.no_setup:
        setup_test_environment()
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Run the specified tests
    success = False
    
    if args.test_type == 'unit':
        success = run_unit_tests(args.verbose, args.coverage)
    elif args.test_type == 'integration':
        success = run_integration_tests(args.verbose)
    elif args.test_type == 'all':
        success = run_all_tests(args.verbose, args.coverage)
    elif args.test_type == 'lint':
        success = run_linting()
    elif args.test_type == 'type':
        success = run_type_checking()
    elif args.test_type == 'report':
        success = generate_test_report()
    elif args.test_type in ['scraper', 'etl', 'embeddings', 'api', 'dag']:
        success = run_specific_module_tests(args.test_type, args.verbose)
    
    # Print final result
    print("\n" + "="*60)
    if success:
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("💥 SOME TESTS FAILED!")
        sys.exit(1)


if __name__ == '__main__':
    main()