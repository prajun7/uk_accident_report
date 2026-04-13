import os
import joblib
import pandas as pd
import warnings

# Suppress sklearn warnings for unfeature-names etc.
warnings.filterwarnings("ignore")

class AccidentSeverityPredictor:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(base_dir, 'output')
        
        model_path = os.path.join(self.output_dir, '7_rf_model.joblib')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError("Missing trained model. Please re-run the pipeline with the latest updates.")
            
        self.model = joblib.load(model_path)
        
        # Severity mapping based on UK DfT dataset context
        self.severity_map = {
            1: "Fatal",
            2: "Serious",
            3: "Slight"
        }

    def predict(self, input_data: dict) -> str:
        """
        Takes a dictionary of numerical features (matching the DfT coding) and predicts the severity.
        """
        df = pd.DataFrame([input_data])
        
        expected_cols = [
            'Speed_limit', 'Road_Type', 'Light_Conditions', 
            'Weather_Conditions', 'Road_Surface_Conditions', 
            'Day_of_Week', 'Urban_or_Rural_Area', 
            'Hour', 'IsNight', 'Month'
        ]
        
        # Fill missing features with fallback defaults
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
                
        # Must be in specific target order
        df = df[expected_cols]

        # Predict
        prediction = self.model.predict(df)[0]
        
        # Get probabilities
        probs = self.model.predict_proba(df)[0]
        prob_str = f"Probabilities: Fatal={probs[0]:.2f}, Serious={probs[1]:.2f}, Slight={probs[2]:.2f}"
        
        return f"{self.severity_map.get(prediction, 'Unknown')} ({prob_str})"


if __name__ == "__main__":
    predictor = AccidentSeverityPredictor()
    
    # ---------------------------------------------------------
    # Example 1: Use DfT Numeric Codes
    # (High speed 60, Rural=2, Dark=4, Rain=2, Single Carriageway=6)
    # ---------------------------------------------------------
    scenario_1 = {
        'Speed_limit': 60,
        'Road_Type': 6,
        'Light_Conditions': 4,
        'Weather_Conditions': 2,
        'Road_Surface_Conditions': 2,
        'Day_of_Week': 6,
        'Urban_or_Rural_Area': 2,
        'Hour': 23,
        'IsNight': 1,
        'Month': 11
    }
    
    # ---------------------------------------------------------
    # Example 2: Typical urban scenario
    # (Low speed 30, Urban=1, Daylight=1, Fine=1)
    # ---------------------------------------------------------
    scenario_2 = {
        'Speed_limit': 30,
        'Road_Type': 3,
        'Light_Conditions': 1,
        'Weather_Conditions': 1,
        'Road_Surface_Conditions': 1,
        'Day_of_Week': 2,
        'Urban_or_Rural_Area': 1,
        'Hour': 14,
        'IsNight': 0,
        'Month': 6
    }
    
    print("--- INFERENCE TESTS ---")
    print(f"Scenario 1 (Rural / Dark / 60mph): {predictor.predict(scenario_1)}")
    print(f"Scenario 2 (Urban / Day / 30mph):  {predictor.predict(scenario_2)}")
