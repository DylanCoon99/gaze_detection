# Report — takes predictions + ground truth, runs all metrics, returns structured result


'''
The entry point. Takes predictions and ground truth, calls accuracy, safety, and slicing, and assembles everything into one structured result (a
dataclass or dict). This is what the runner calls — one function in, one report out. This result is what gets logged to MLflow later.
'''