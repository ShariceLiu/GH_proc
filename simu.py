from scipy import special, stats
import numpy as np
import matplotlib.pyplot as plt

n=10
m = 20
l_lgs = np.linspace(0,5,num=n, endpoint=False)
omg_lgs = np.linspace(0,2,num=n, endpoint=False)
omega = 1
l = 1
plt.figure()
for om_lg in omg_lgs:
    omega = np.exp(om_lg)
    xs = np.linspace(0,20, num=50)
    pdfs = stats.geninvgauss.pdf(xs, p=l, b=omega)
    plt.plot(xs, pdfs, label = f'log omega {om_lg:.1f}')

plt.legend()
plt.savefig('figure/gig/wrt_omega.png')
plt.show()