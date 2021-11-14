import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig=plt.figure(figsize=(8, 8))
def animation_frame(i):
    print(i)
    im = plt.imread('sgd/'+str(i)+'.png')
    plt.imshow(im)

animation = FuncAnimation(fig, func=animation_frame, frames=np.arange(0,250,1), interval=10)
animation.save('sgd/animation.gif', writer='imagemagick', fps=5)
plt.close()

fig=plt.figure(figsize=(8, 8))
def animation_frame(i):
    print(i)
    im = plt.imread('mahal/'+str(i)+'.png')
    plt.imshow(im)

animation = FuncAnimation(fig, func=animation_frame, frames=np.arange(0,250,1), interval=10)
animation.save('mahal/animation.gif', writer='imagemagick', fps=5)
plt.close()

fig=plt.figure(figsize=(8, 8))
def animation_frame(i):
    print(i)
    im = plt.imread('l2/'+str(i)+'.png')
    plt.imshow(im)

animation = FuncAnimation(fig, func=animation_frame, frames=np.arange(0,250,1), interval=10)
animation.save('l2/animation.gif', writer='imagemagick', fps=5)
