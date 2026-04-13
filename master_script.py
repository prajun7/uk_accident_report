import os
import importlib.util
import time

def run_script(script_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(base_dir, script_name)
    
    if not os.path.exists(script_path):
        print(f"ERROR: Cannot find {script_name} at {script_path}")
        return False
        
    print(f"==================================================")
    print(f" EXECUTING: {script_name}")
    print(f"==================================================")
    
    try:
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location("module.name", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Execute run()
        if hasattr(module, 'run'):
            start = time.time()
            module.run()
            end = time.time()
            print(f">>> Completed {script_name} in {end - start:.2f} seconds\n")
            return True
        else:
            print(f"ERROR: {script_name} does not have a run() function.")
            return False
            
    except Exception as e:
        print(f"FATAL ERROR executing {script_name}: {e}")
        return False

def main():
    print("Welcome to the UK Traffic Accident Big Data Pipeline")
    print("Beginning automated execution of steps 3 through 8...\n")
    
    scripts = [
        "3_data_acquisition_filtering.py",
        "4_data_extraction.py",
        "5_data_validation_cleansing.py",
        "6_data_aggregation_representation.py",
        "7_data_analysis.py",
        "8_data_visualization.py"
    ]
    
    for script in scripts:
        success = run_script(script)
        if not success:
            print(f"Pipeline artificially halted at {script} due to failure.")
            break
            
    print("Pipeline execution complete! Check output/ directory for results.")

if __name__ == "__main__":
    main()
