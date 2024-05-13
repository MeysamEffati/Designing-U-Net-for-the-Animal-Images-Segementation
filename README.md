# Convolutional Neural Network Architectural Components

This repository contains Python code for building components of a Convolutional Neural Network (CNN). These components include functions for downsampling, bottleneck layers, upsampling, and the final output cell. The code is written using the Keras library.

## Functions

### 1. Downsampling
```python
def Downsampling(Input, N_filters):
  # ... (refer to the code for the complete function)
```
Performs downsampling on the input by applying convolutional layers followed by max-pooling.

### 2. Bottleneck
```python
def Bottleneck(Input, N_filters):
  # ... (refer to the code for the complete function)
```
Creates a bottleneck layer by applying convolutional layers followed by upsampling.

### 3. Upsampling
```python
def Upsampling(In_Enc, In_Bott_Dec, Num_Filters):
  # ... (refer to the code for the complete function)
```
Combines input from the encoder and bottleneck decoder, performs convolution, and applies upsampling.

### 4. Output Cell
```python
def OutCell(In_Enc, In_Bott_Dec, Num_Filters):
  # ... (refer to the code for the complete function)
```
Takes input from the encoder and bottleneck decoder, applies convolution, and produces the final output layer.

## Example Usage

Here is an example showing how to create a model using the provided functions:

```python
Input_Layer = Input(shape=(572, 572, 1))
Downsampling1, lastlayer1 = Downsampling(Input_Layer, 64)
Downsampling2, lastlayer2 = Downsampling(Downsampling1, 128)
Downsampling3, lastlayer3 = Downsampling(Downsampling2, 256)
Downsampling4, lastlayer4 = Downsampling(Downsampling3, 512)
BottleNeck, layer_Bottleneck = Bottleneck(Downsampling4, 1024)

Upsampled, lastlayer = Upsampling(lastlayer4, BottleNeck, 512)
Upsampled, lastlayer = Upsampling(lastlayer3, Upsampled, 256)
Upsampled, lastlayer = Upsampling(lastlayer2, Upsampled, 128)
OutLayer = OutCell(lastlayer1, Upsampled, 64)

model = Model(Input_Layer, OutLayer)
model.summary()
```

This example demonstrates constructing a CNN model by sequentially applying the downsampling, bottleneck, and upsampling functions, and finally creating the output layer.

## Additional Resources

- To see these functions in action, you can run the provided [Google Colab Notebook](https://colab.research.google.com/drive/1IAN2Or4BI8XVp6WYEthBH2jl-avfSqre#scrollTo=YwFEKQlK-ecp&line=1&uniqifier=1).

**Note:** Ensure that you have the necessary dependencies, such as Keras, installed in your environment.

## License

This code is provided under the [MIT License](LICENSE).
