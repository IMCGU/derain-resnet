
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data
data16 = pd.read_csv('C:\\Users\\HanHan\\derain-SWRL\\28854\\b16loss.csv',sep=r'\s*,\s*',header=0, encoding='ascii', engine='python')
data8 = pd.read_csv('C:\\Users\\HanHan\\derain-SWRL\\28854\\b16woloss.csv',sep=r'\s*,\s*',header=0, encoding='ascii', engine='python')

print(data16['Rain100L_psnr'])
print(data8['Rain100L_psnr'])
# Plot
plt.figure(figsize=(6.8, 4.2))
plt.style.use('ggplot')#畫出格線
ax = plt.axes()#畫上刻度
ax.set_ylim([31,33])#控制Y的範圍
# ax.set_xlim([25000,125000])#控制Y的範圍
# plt.ylim(28, 36)#控制Y的範圍
# plt.plot(data16['iteration'],data16['Rain100H_ssim'],label="block16",color='red')
plt.plot(data16['iteration'],data16['Rain100L_psnr'],label="connect",color='red')
plt.plot(data8['iteration'],data8['Rain100L_psnr'],label="no connect",color='blue')
# plt.plot(data8['iteration'],data8['Rain100H_ssim'],label="block8",color='blue')
# plt.yticks(np.arange(31, max(data16['Rain100L_psnr'])+1, 1))
# plt.yticks(np.arange(31, max(data8['Rain100L_psnr'])+1, 1))

# plt.yticks(np.arange(26, 32, 1))
# plt.yticks(np.arange(26, 32, 1))

plt.xlabel('iteration')
# plt.ylabel('Rain12_psnr')
plt.ylabel('Rain100L_psnr')
plt.legend(loc='upper right', shadow=True)
# plt.legend(loc='lower right', shadow=True)
# plt.xticks(0.2,data16['iteration'])
# plt.xticks(0.2,data8['iteration'])
plt.show()



