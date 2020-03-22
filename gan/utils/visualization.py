import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SupResS2.')
    parser.add_argument('--dis', help='Path of train data.')
    parser.add_argument('--gene',help='generator loss')
    parser.add_argument('--pic_name', help='Path of train data.')
    args = parser.parse_args()  

    path = 'network_data/'
    disc_loss = np.load(path + args.dis)
    gene_loss = np.load(path + args.gene)

    f = plt.figure(1)
    ax1 = f.add_subplot(2,1,1)
    ax1.plot(disc_loss)
    ax2 = f.add_subplot(2,1,2)
    ax2.plot(gene_loss)
#plt.show()
    if not os.path.isdir(args.pic_name):
        os.mkdir(args.pic_name)
    f.savefig(args.pic_name+"loss.pdf")



'''
predict = np.load("/p/project/joaiml/remote_sensing/run_zhang/DSen2/data/test32/S2A_MSIL1C_20170616T223701_N0205_R072_T59GLL_20170616T223702.SAFE/gen_model_500-predict.npy")
gt = np.load("/p/project/joaiml/remote_sensing/run_zhang/DSen2/data/test32/S2A_MSIL1C_20170616T223701_N0205_R072_T59GLL_20170616T223702.SAFE/no_tiling/data20_gt.npy")
plt.clf()
fig = plt.figure()
ax1 = fig.add_subplot(161)
ax1.imshow(predict[:,:,0])
ax1.set_title('B5')

ax2 = fig.add_subplot(162)
ax2.imshow(predict[:,:,1])
ax2.set_title('B6')

ax3 = fig.add_subplot(163)
ax3.imshow(predict[:,:,2])
ax3.set_title('B7')

ax4 = fig.add_subplot(164)
ax4.imshow(predict[:,:,3])
ax4.set_title('B8')

ax5 = fig.add_subplot(165)
ax5.imshow(predict[:,:,4])
ax5.set_title('B8b')

ax6 = fig.add_subplot(166)
ax6.imshow(predict[:,:,5])
ax6.set_title('B9')

##################################

ax7 = fig.add_subplot(261)
ax7.imshow(gt[:,:,0])


ax8 = fig.add_subplot(262)
ax8.imshow(gt[:,:,1])


ax9 = fig.add_subplot(263)
ax9.imshow(gt[:,:,2])


ax10 = fig.add_subplot(264)
ax10.imshow(gt[:,:,3])


ax11 = fig.add_subplot(265)
ax11.imshow(gt[:,:,4])


ax12 = fig.add_subplot(266)
ax12.imshow(gt[:,:,5])


fig.savefig("visu_case.pdf", format='pdf', dpi=2000)
#####################################
fig2 = plt.figure()
ax13 = fig2.add_subplot(161)
ax13.imshow(pow(predict[:,:,0]-gt[:,:,0],2))

ax14 = fig2.add_subplot(162)
ax14.imshow(pow(predict[:,:,1]-gt[:,:,1],2))

ax15 = fig2.add_subplot(163)
ax15.imshow(pow(predict[:,:,2]-gt[:,:,2],2))

ax17 = fig2.add_subplot(164)
ax17.imshow(pow(predict[:,:,3]-gt[:,:,3],2))

ax18 = fig2.add_subplot(165)
ax18.imshow(pow(predict[:,:,4]-gt[:,:,4],2))

ax19 = fig2.add_subplot(166)
ax19.imshow(pow(predict[:,:,5]-gt[:,:,5],2))
#plt.clf()
#fig2.colorbar()
#
fig2.savefig("diff.pdf", format='pdf', dpi=2000)
'''
