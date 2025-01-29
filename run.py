# import os
# from main import app

# if __name__ == '__main__':
#     port = int(os.environ.get("PORT", 5000))  # Render assigns a dynamic port
#     app.run(host="0.0.0.0", port=port, debug=True)
import os
import tensorflow as tf
from main import app

# Disable GPU for Render / Railway
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use Railway-assigned port
    app.run(host="0.0.0.0", port=port)  # Remove debug=True

