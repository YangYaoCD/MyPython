import matplotlib.pyplot as plt
import numpy as np

def ft(x,y,N,kx):
    z = []
    for k in kx:
        sum = 0
        for n in range(N):
            # sum+=y[n]*np.exp(-1j*2*np.pi*k*n/N)
            sum+=y[n]*(np.cos(2*np.pi*k*n/N)-1j*np.sin(2*np.pi*k*n/N))
            if n==N:
                print("到N了")
        z.append(sum)
    return z


N=1001
x=np.linspace(0,1,N)
k=np.linspace(0,500,501)
y=5*np.sin(2*np.pi*50*x)+np.cos(2*np.pi*250*x)+np.cos(2*np.pi*150*x)+10*np.cos(2*np.pi*290*x)
print(max(y))
plt.plot(x[0:100],y[0:100])
plt.xlabel("simple dot number:"+str(N))
plt.title("sandiantu")
plt.show()
z=ft(x, y, N,k)
print(z)
print(np.max(z))
plt.plot(k,np.abs(z))

# plt.plot(np.linspace(0,1000,1001),np.real(z))
# plt.plot(np.real(z),np.imag(z))
plt.xlabel("simple dot number:"+str(N))
plt.title("f")
plt.show()