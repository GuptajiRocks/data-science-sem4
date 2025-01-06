import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import cv2
from moviepy.editor import VideoFileClip


def visualize_tabular_data():
    df = pd.read_csv('Practical\\jan6\\data.csv')
    print("CSV Data Properties:", df.info())

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df)
    plt.title('Line Plot')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=df.columns[0], y=df.columns[1], data=df)
    plt.title('Bar Plot')
    plt.show()

    plt.figure(figsize=(10, 6))
    df.hist()
    plt.title('Histogram')
    plt.show()

def visualize_image():
    img = cv2.imread('Practical\\jan6\\image.jpg')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title('Image Display')
    plt.show()

def visualize_3d_image():
    X, Y = np.meshgrid(range(10), range(10))
    Z = np.sin(X) * np.cos(Y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    plt.title('3D Plot')
    plt.show()

def play_video():
    clip = VideoFileClip('video.mp4')
    clip.preview()

def visualize_text():
    with open('Practical\\jan6\\text.txt', 'r') as file:
        text = file.read()

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud')
    plt.show()

if __name__ == "__main__":
    visualize_tabular_data()
    visualize_image()
    visualize_3d_image()
    #play_video()
    visualize_text()
