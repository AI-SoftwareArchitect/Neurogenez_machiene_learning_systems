🧠 GrowNet: Dynamic Neuron Growth in Deep Learning
Welcome to the GrowNet project! 🚀 This project explores a new approach to deep learning, where the neural network's architecture grows during training by adding new neurons to the hidden layers. It's like giving your model a brain boost every epoch! 🧠💡

🚀 What's this about?
We start with a small neural network, train it on the MNIST dataset 🧑‍🏫, and after each epoch, we expand the number of neurons in the hidden layers.

The idea? Mimic the biological brain's ability to grow and adapt over time! 🌱

📚 Features
Dynamic neuron growth during training 🌱

Progressive increase in model complexity as training continues 🧠

Built using PyTorch 🔥

Performance improvement as more neurons are added! 🎯

📊 Model Architecture
Layer	Input Size	Output Size	Growth per Epoch
Input Layer	28x28 (784)	784	N/A
Hidden Layer	784	64 → 104	+8 neurons/epoch
Output Layer	64 → 104	10 (digits)	N/A
Input Layer: Takes the MNIST image (28x28 pixels).

Hidden Layer: Grows with +8 neurons after each epoch.

Output Layer: Classifies into 10 possible digit classes (0-9).

🛠️ How it Works?
GrowLinear Layer: Custom layer that allows neurons to grow by adding new ones at the end of training epochs.

Neural Growth: After each epoch, the model's neurons grow! 🎉 Each growth round adds new neurons to the hidden layer, increasing model capacity.

Training: The model trains using MNIST data, which is a dataset of handwritten digits.

📈 Performance
The model improves progressively over time as the neurons grow! Here’s a quick look at how it evolved:

Epoch	Accuracy	Hidden Layer Size	Neurons Added
1	89.57%	72	+8
2	94.20%	80	+8
3	94.24%	88	+8
4	94.26%	96	+8
5	94.26%	104	+8
Each epoch, we grow a bit more! 🌱 By the end, the model has significantly improved accuracy. 🌟

🔍 Confusion Matrix
Here’s how the model performed across different digits! The confusion matrix shows us where the model did well (diagonal) and where it struggled (off-diagonal). 📊


💻 Installation and Setup
To run the project, you'll need the following libraries:

PyTorch (for neural network magic 🧙‍♂️)

TorchVision (for the MNIST dataset 📚)

Seaborn (for visualizing confusion matrix 📈)

Clone this repository:

bash
Kopyala
Düzenle
git clone https://github.com/your-username/grownet.git
cd grownet
Install dependencies:

bash
Kopyala
Düzenle
pip install -r requirements.txt
Run the model:

bash
Kopyala
Düzenle
python main.py
🔧 Dependencies
PyTorch

TorchVision

Seaborn

Matplotlib

📸 Visualization
Here’s a glimpse of how the confusion matrix looks after training. As you can see, the model starts classifying digits pretty well! The darker the shade, the more accurate the predictions. 💪


🔮 Future Enhancements
🚀 Dynamic pruning of neurons: Only keep the best neurons for more efficient growth!

🌍 Expand the model to handle other datasets like CIFAR-10 or ImageNet.

🧠 Biologically inspired learning mechanisms like synaptic plasticity.

👥 Contributing
Want to contribute? 🎉 We welcome PRs to make the brain grow even smarter. Check the issues section for any open tasks!
