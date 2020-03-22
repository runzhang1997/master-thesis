import numpy as np
import matplotlib.pyplot as plt
from random import seed
from random import randint
seed(10)
n = 100

#tile = 'S2A_MSIL1C_20170522T110621_N0205_R137_T30UXC_20170522T110912.SAFE/'
tile = 'S2A_MSIL1C_20170616T223701_N0205_R072_T59GLL_20170616T223702.SAFE/'
#tile = 'S2A_MSIL2A_20170419T155901_N9999_R097_T16PEU_20191102T153355.SAFE/'
#tile = 'S2A_MSIL2A_20170522T110621_N9999_R137_T30UXC_20191102T160850.SAFE/'
prefix1 = '/p/project/joaiml/remote_sensing/run_zhang/DSen2/data/test60_96_interp/'
prefix2 = '/p/project/joaiml/remote_sensing/run_zhang/DSen2/data/test60_192/'
#prefix1 = '/p/project/joaiml/remote_sensing/run_zhang/DSen2/data/test20_32_interp_bilinear/'
Dsen2 = np.load(prefix1 + tile + 's2_030_-predict.npy')
#Ours = np.load(prefix2 + tile  + 'self_skip_subpixel.npy')
gt = np.load(prefix1 + tile + "no_tiling/data60_gt.npy") 
Ours = np.load(prefix1 + tile  + 'pretrain_60m.npy')
#prefix3 = '/p/project/joaiml/remote_sensing/run_zhang/DSen2/data/_test20_32_16_GT/'
Interpolation = np.load(prefix2 + tile + "bicubicInterp.npy")

"""
wgan = np.load(prefix1 + tile + "pretrain_20m_real_wgan.npy")
hinge= np.load(prefix1 + tile + "pretrain_20m_real_hinge.npy")
vanila= np.load(prefix1 + tile + "pretrain_20m_real_vanila.npy")
relative= np.load(prefix1 + tile + "pretrain_20m_real_relativistic.npy")

gt = gt[:,:,[5,3,0]]
Ours = Ours[:,:,[5,3,0]]
Interpolation = Interpolation[:,:,[5,3,0]]
Dsen2 = Dsen2[:,:,[5,3,0]]
wgan = wgan[:,:,[5,3,0]]
hinge = hinge[:,:,[5,3,0]]
vanila = vanila[:,:,[5,3,0]]
relative = relative[:,:,[5,3,0]]

print (gt.shape)
print (Ours.shape)

for i in range(Ours.shape[2]):
    a, b = np.percentile(Ours[:, :, i], (1, 99))
    gt[:, :, i] = np.clip(gt[:, :, i], a, b)
    gt[:, :, i] = (gt[:, :, i] - a) / (b - a)
    Ours[:, :, i] = np.clip(Ours[:, :, i], a, b)
    Ours[:, :, i] = (Ours[:, :, i] - a) / (b - a)
    Dsen2[:, :, i] = np.clip(Dsen2[:, :, i], a, b)
    Dsen2[:, :, i] = (Dsen2[:, :, i] - a) / (b - a)
    Interpolation[:, :, i] = np.clip(Interpolation[:, :, i], a, b)
    Interpolation[:, :, i] = (Interpolation[:, :, i] - a) / (b - a)
    
    wgan[:, :, i] = np.clip(wgan[:, :, i], a, b)
    wgan[:, :, i] = (wgan[:, :, i] - a) / (b - a)   
    hinge[:, :, i] = np.clip(hinge[:, :, i], a, b)
    hinge[:, :, i] = (hinge[:, :, i] - a) / (b - a)
    vanila[:, :, i] = np.clip(vanila[:, :, i], a, b)
    vanila[:, :, i] = (vanila[:, :, i] - a) / (b - a)
    relative[:, :, i] = np.clip(relative[:, :, i], a, b)
    relative[:, :, i] = (relative[:, :, i] - a) / (b - a)
    

for i in range(10):
    random = randint(500, 3000)
    print (random)
    #a = 64
    #b = 384
    a = 128
    b = 256
    plt.clf()
    fig = plt.figure()
    plt.imshow(gt[random:random+a, random:random+a, :]) 
    plt.axis('off')
    fig.savefig("original%d.png"%i, format='png',bbox_inches='tight', pad_inches=0)
    
    random = random * 2
    plt.clf()
    fig = plt.figure()
    plt.imshow(Ours[random:random+b, random:random+b, :])
    plt.axis('off')
    fig.savefig("ours%d.png"%i, format='png',bbox_inches='tight', pad_inches=0)

    plt.clf()
    fig = plt.figure()
    plt.imshow(Dsen2[random:random+b, random:random+b, :])
    plt.axis('off')
    fig.savefig("Dsen2%d.png"%i, format='png',bbox_inches='tight', pad_inches=0)

    plt.clf()
    fig = plt.figure()
    plt.imshow(Interpolation[random:random+b, random:random+b,:])
    plt.axis('off')
    fig.savefig("Interpolation%d.png"%i, format='png',bbox_inches='tight', pad_inches=0)
    
    plt.clf()
    fig = plt.figure()
    plt.imshow(wgan[random:random+b, random:random+b, :])
    plt.axis('off')
    fig.savefig("wgan%d.png"%i, format='png',bbox_inches='tight', pad_inches=0)
    
    plt.clf()
    fig = plt.figure()
    plt.imshow(hinge[random:random+b, random:random+b,:])
    plt.axis('off')
    fig.savefig("hinge%d.png"%i, format='png',bbox_inches='tight', pad_inches=0)

    plt.clf()
    fig = plt.figure()
    plt.imshow(vanila[random:random+b, random:random+b,:])
    plt.axis('off')
    fig.savefig("vanila%d.png"%i, format='png',bbox_inches='tight', pad_inches=0)

    plt.clf()
    fig = plt.figure()
    plt.imshow(relative[random:random+b, random:random+b,:])
    plt.axis('off')
    fig.savefig("relative%d.png"%i, format='png',bbox_inches='tight', pad_inches=0)
"""   

for band in range(0,2):
    dsen2 = np.abs(Dsen2[:, :, band] - gt[:, :, band])
    ours = np.abs(Ours[:, :, band] - gt[:, :, band])
    interpolation = np.abs(Interpolation[:, :, band] - gt[:, :, band])
    # Stretching the image for better visualization
    #for i in range(ours.shape[2]):
   
    plt.clf()
    fig = plt.figure()
    plt.imshow(ours, vmin=0, vmax=400)
    plt.axis('off')
    fig.savefig("ours-%d.png"%band, format='png', transparent=True, bbox_inches='tight', pad_inches=0)

    plt.clf()
    fig = plt.figure()
    plt.imshow(dsen2, vmin=0, vmax=400)
    plt.axis('off')
    fig.savefig("dsen2-%d.png"%band, format='png', transparent=True, bbox_inches='tight', pad_inches=0)

    plt.clf()
    fig = plt.figure()
    plt.imshow(interpolation, vmin=0, vmax=400)
    plt.axis('off')
    fig.savefig("interpolation-%d.png"%band, format='png', transparent=True, bbox_inches='tight', pad_inches=0)

a, b = np.percentile(gt, (1, 99))
#ours = np.clip(ours, a, b)
#ours = (ours - a) / (b - a)
#dsen2 = np.clip(dsen2, a, b)
#dsen2 = (dsen2 - a) / (b - a)
gt = np.clip(gt, a, b)
gt = (gt - a) / (b - a)
#interpolation = np.clip(interpolation, a, b)
#interpolation = (interpolation - a) / (b - a)
for band in range(0,2): 
    plt.clf()
    fig = plt.figure()
    plt.imshow(gt[:, :, band])
    plt.axis('off')
    fig.savefig("gt-%d.png"%band, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
