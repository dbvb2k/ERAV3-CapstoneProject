#!/usr/bin/env python3
"""
Quick Setup Verification Script
Verifies that all components are ready for travel model training
"""

import os
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

def check_datasets():
    """Check if all required datasets are present."""
    console.print("[blue]📊 Checking datasets...[/blue]")
    
    datasets = [
        "../FINAL_TRAINING_DATASET_LLAMA8B.jsonl",
        "../FINAL_VALIDATION_DATASET_LLAMA8B.jsonl", 
        "../FINAL_TEST_DATASET_LLAMA8B.jsonl"
    ]
    
    results = []
    for dataset in datasets:
        if os.path.exists(dataset):
            size = os.path.getsize(dataset) / (1024 * 1024)  # MB
            with open(dataset, 'r') as f:
                lines = sum(1 for _ in f)
            results.append((dataset, True, f"{size:.1f} MB", lines))
        else:
            results.append((dataset, False, "Missing", 0))
    
    return results

def check_scripts():
    """Check if all required scripts are present."""
    console.print("[blue]🔧 Checking scripts...[/blue]")
    
    scripts = [
        "setup_runpod_travel.py",
        "train_travel_llama8b.py",
        "test_travel_model.py",
        "start_travel_training.sh",
        "requirements.txt"
    ]
    
    results = []
    for script in scripts:
        exists = os.path.exists(script)
        executable = os.access(script, os.X_OK) if script.endswith('.sh') else True
        results.append((script, exists, executable))
    
    return results

def check_python_packages():
    """Check if key Python packages can be imported."""
    console.print("[blue]📦 Checking Python packages...[/blue]")
    
    packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("peft", "PEFT"),
        ("rich", "Rich"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy")
    ]
    
    results = []
    for package, name in packages:
        try:
            __import__(package)
            results.append((name, True, "✅"))
        except ImportError:
            results.append((name, False, "❌"))
    
    return results

def main():
    """Main verification function."""
    console.print(Panel.fit("🔍 TRAVEL MODEL SETUP VERIFICATION", style="bold blue"))
    
    # Check datasets
    dataset_results = check_datasets()
    dataset_table = Table(title="📊 Dataset Status", show_header=True)
    dataset_table.add_column("Dataset", style="cyan")
    dataset_table.add_column("Status", style="green")
    dataset_table.add_column("Size", style="yellow")
    dataset_table.add_column("Examples", style="magenta")
    
    all_datasets_ok = True
    for dataset, exists, size, lines in dataset_results:
        status = "✅ Found" if exists else "❌ Missing"
        if not exists:
            all_datasets_ok = False
        dataset_table.add_row(os.path.basename(dataset), status, size, str(lines))
    
    console.print(dataset_table)
    
    # Check scripts
    script_results = check_scripts()
    script_table = Table(title="🔧 Script Status", show_header=True)
    script_table.add_column("Script", style="cyan")
    script_table.add_column("Exists", style="green")
    script_table.add_column("Executable", style="yellow")
    
    all_scripts_ok = True
    for script, exists, executable in script_results:
        exist_status = "✅" if exists else "❌"
        exec_status = "✅" if executable else "❌"
        if not exists or not executable:
            all_scripts_ok = False
        script_table.add_row(script, exist_status, exec_status)
    
    console.print(script_table)
    
    # Check packages
    package_results = check_python_packages()
    package_table = Table(title="📦 Package Status", show_header=True)
    package_table.add_column("Package", style="cyan")
    package_table.add_column("Status", style="green")
    
    all_packages_ok = True
    for package, available, status in package_results:
        if not available:
            all_packages_ok = False
        package_table.add_row(package, status)
    
    console.print(package_table)
    
    # Overall status
    console.print("\n" + "="*60)
    if all_datasets_ok and all_scripts_ok and all_packages_ok:
        console.print("[bold green]🎉 ALL CHECKS PASSED! Ready to start training![/bold green]")
        console.print("[green]Run: ./start_travel_training.sh[/green]")
    else:
        console.print("[bold red]❌ SOME CHECKS FAILED[/bold red]")
        if not all_datasets_ok:
            console.print("[red]• Missing dataset files - ensure they're in parent directory[/red]")
        if not all_scripts_ok:
            console.print("[red]• Missing or non-executable scripts[/red]")
        if not all_packages_ok:
            console.print("[red]• Missing Python packages - run: python setup_runpod_travel.py[/red]")
    
    console.print("="*60)

if __name__ == "__main__":
    main() 