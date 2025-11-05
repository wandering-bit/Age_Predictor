🧠 Age Predictor — Linear Regression from Scratch

This project implements a simple linear regression model from scratch using Python and NumPy, without relying on machine learning libraries like Scikit-Learn.
It demonstrates the fundamental concepts of gradient descent, error minimization, and model training through pure mathematical operations.

🚀 Features

Implements forward propagation and manual gradient descent.

Visualizes training progress with a live plot showing the regression line fitting the data.

Predicts salaries (target variable) based on age (input variable).

Calculates and displays the average prediction error after training.

🧩 How It Works

Initializes random weights and bias.

Iteratively updates them using gradients computed from prediction errors.

Displays real-time visualization using Matplotlib to show the line adjusting over epochs.

📊 Example

Each dot represents a data point (age, salary). The model learns the best-fit line that predicts salary based on age.
<img width="667" height="415" alt="image" src="https://github.com/user-attachments/assets/38b0e066-c1e2-4e05-a0d9-13f0e6d24daf" />
<p>Before Training:</p>
<img width="480" height="358" alt="image" src="https://github.com/user-attachments/assets/2aff16e0-42ca-49aa-ae79-c4434ad63078" />
<P>After Training:</P>
<img width="480" height="358" alt="image" src="https://github.com/user-attachments/assets/1e7abf8b-2818-4b99-9bc5-3de7699267cb" />
