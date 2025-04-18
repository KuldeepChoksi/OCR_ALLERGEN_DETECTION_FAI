import io
from contextlib import redirect_stdout
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load the model from the Keras file.
model = load_model('custom_cnn_model_all.keras')

# Option 1: Print the model summary directly.
model.summary()

# Option 2: Capture the summary as a string.
def get_model_summary(model):
    """
    Capture the model summary as a string.
    
    Parameters:
        model (tf.keras.Model): Your loaded Keras model.
    
    Returns:
        str: The model summary.
    """
    stream = io.StringIO()
    with redirect_stdout(stream):
        model.summary()
    summary_str = stream.getvalue()
    stream.close()
    return summary_str

summary_text = get_model_summary(model)
#print("Captured Model Summary:\n")
#print(summary_text)

# Additional Statistics:
total_params = model.count_params()
#print("Total Parameters: ", total_params)
