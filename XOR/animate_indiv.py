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

# High init too high mahal Done
# Redo perfect hyper-params with high init Done
# Small init Done

# Set LR Smaller and use more hidden neurons. Possible this is just a dynamic of unstable LR
# Try plot decision boundard network
# Turn off L2 (infinitely small L2 trained for an infinite time)
