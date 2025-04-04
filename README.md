# Webcam Frame Analysis with LLM Ollama 
This project demonstrates how to capture frames from a webcam, process them by reducing resolution, and then send the frame to the Llama3.2 Vision model through the Ollama API for description. 
The description is generated and printed in the terminal, along with the processing time.

## Prerequisites
Before running the project, ensure you have the following dependencies installed:
- Python 3.x
- OpenCV (opencv-python)
- Ollama Python API (ollama)
  
To install the required libraries, you can run:\
bash
`pip install opencv-python ollama`

Additionally, you need an Ollama API key to access the Llama3.2 Vision model.\
Follow Ollama’s documentation for setup.

## Project Overview
This script performs the following tasks:
- Opens the webcam and captures frames.
- Reduces the resolution of each captured frame for faster processing.
- Sends the frame to the Ollama API for analysis using the Llama3.2 Vision model.
- Prints the description of the image generated by the model.
