import numpy as np
#np.random.seed(0) #this ensures that the random number generated for x,y,w and b are always the same.

N = 100
x = np.random.uniform(-3,3,size=N)
y = 2.5 * x + np.random.normal(scale=1.0, size=N)

w = np.random.randn()
b = np.random.randn()
lr = 0.01
for epoch in range(1000):
    y_pred = w * x + b
    error = y_pred - y
    loss = (error**2).mean()
    # gradients
    dw = (2.0 / N) * (error * x).sum()
    db = (2.0 / N) * error.sum()
    # update
    w -= lr * dw
    b -= lr * db
    if epoch % 200 == 0:
        print(f'ep {epoch} loss {loss:.4f} w {w:.4f} b {b:.4f}')
print('final', w, b)