
import pandas as pd

def match_division(prediction_set, draw_set):
    """Matches a prediction set to a draw and returns the division hit"""
    matches = len(set(prediction_set).intersection(draw_set))
    if matches == 7:
        return "Div 1"
    elif matches == 6:
        return "Div 2"
    elif matches == 5:
        return "Div 3"
    elif matches == 4:
        return "Div 4"
    elif matches == 3:
        return "Div 5"
    elif matches == 2:
        return "Div 6"
    else:
        return "No Win"

def simulate_generation(predictions, historical_draws_df):
    """Simulates a generation's predictions against historical draw data"""
    results = []
    for idx, pred_set in enumerate(predictions):
        outcome = {"ID": f"Set-{idx+1}", "Prediction": pred_set}
        div_counts = {
            "Div 1": 0, "Div 2": 0, "Div 3": 0,
            "Div 4": 0, "Div 5": 0, "Div 6": 0,
            "No Win": 0
        }
        for _, draw in historical_draws_df.iterrows():
            draw_set = list(draw.values)
            division = match_division(pred_set, draw_set)
            div_counts[division] += 1
        outcome.update(div_counts)
        results.append(outcome)
    return pd.DataFrame(results)
