import numpy as np
import matplotlib.pyplot as plt


def plotOriPlot(x, y):
    plt.plot(x, y)
    plt.grid(True)
    plt.gca().set_xlim(0, 1)
    plt.ylabel("amp")
    plt.xlabel("t(s)")
    plt.show()


def plotDFT(k, result, N):
    plt.gca().set_xlim(0, N)
    plt.ylabel("amp")
    plt.xlabel("Hz")
    plt.plot(k, result)
    plt.show()
    plt.gca().set_xlim(0, N / 2)
    plt.ylabel("amp")
    plt.xlabel("Hz")
    plt.plot(k, result)
    plt.show()


def dft(N, y):
    real, imag = [], []
    k = np.linspace(0, N - 1, N)  # 0-99
    for i in k:
        sum_real, sum_imag = 0., 0.
        for n in range(N):  # 0-100
            sum_real = sum_real + y[n] * np.cos(2 * np.pi * i * n / N)
            sum_imag = sum_imag - y[n] * np.sin(2 * np.pi * i * n / N)
        real.append(sum_real)
        imag.append(sum_imag)
    result = np.sqrt(np.power(real, 2) + np.power(imag, 2))
    return k, real, imag, result


'''傅里叶逆变换'''


def idft(x, N):
    amp_idft = []
    for n in range(N):
        sum = 0.
        for k in range(N):
            sum = sum + (1 / N) * np.real(x[k]) * np.cos(2 * np.pi * n * k / N) \
                  - (1 / N) * np.imag(x[k]) * np.sin(2 * np.pi * n * k / N)
        amp_idft.append(sum)
    return amp_idft


'''希尔伯特变换'''


def hilbertTrans(x, N):
    z = []
    for k in range(N):
        if k == 0 or k == N / 2:
            z.append(0.)
        elif k > 0 and k <= (N / 2 - 1):
            z.append(x[k])
        elif k > N / 2 and k < N:
            z.append(-x[k])
    z = np.dot(-1j, z)
    plt.subplot(2, 1, 1)
    plt.plot(x, color='green', label='x')
    plt.subplot(2, 1, 2)
    plt.plot(z, color='red', label='z')
    plt.legend()
    plt.grid(True)
    plt.show()
    hilb = idft(z, N)
    return hilb


N = 250  # 长度为100
x = np.linspace(0, 1 - 1 / N, N)
y = 20 * np.sin(2 * np.pi * 25 * x) + 10 * np.cos(2 * np.pi * 10 * x)  # 信号+10*np.cos(2*np.pi*20*x)
plotOriPlot(x, y)

# 对序列进行傅里叶变换
k, real, imag, result = dft(N, y)
plotDFT(k, result, N)

hilb = hilbertTrans(real + np.dot(1j, imag), N)
plt.plot(x, hilb, 'g^:', label='Hilbert transformation')
plt.plot(x, y, color='red', label='Original signal')
plt.legend()
plt.gca().set_xlim(0, 0.2)
plt.grid(True)
plt.title("hilbert")
plt.show()
