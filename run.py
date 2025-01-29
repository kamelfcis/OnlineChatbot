# import os
# from main import app

# if __name__ == '__main__':
#     port = int(os.environ.get("PORT", 5000))  # Render assigns a dynamic port
#     app.run(host="0.0.0.0", port=port, debug=True)
import os
import tensorflow as tf
from main import app

# Disable GPU to prevent CUDA-related errors on Render
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Suppress TensorFlow performance warnings
tf.get_logger().setLevel('ERROR')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render assigns a dynamic port
    app.run(host="0.0.0.0", port=port, debug=True)
