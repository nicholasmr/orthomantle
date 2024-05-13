# N. M. Rathmann <rathmann@nbi.ku.dk>, 2023-2024

import os
import numpy as np

#-----------------------

case = 3
#N = 4495
N = 2000

density   = 175 
framerate = 7

scale = 600
numfmt = '%04d'

fout = 'experiments/case-%i-animation'%(case)

#-----------------------

os.system('mkdir -p experiments/case-%i/frames/'%(case))

if 1:
    os.system('rm experiments/case-%i/frames/*.png'%(case))
    for ii, nn in enumerate(np.concatenate(([0,1,2,3,4], np.arange(50,N,50)))):
#    for ii, nn in enumerate(np.concatenate(([0,1,2,3,4], ()))):  
    
#        print('pdftocairo frame %i as %i'%(nn,ii))
#        pdfin  = 'C2-experiments/case-%i/diagnostic-%04d.pdf'%(case,nn)
#        pngout = 'C2-experiments/case-%i/frames/diagnostic-%04d'%(case,ii)
#        os.system('pdftocairo -singlefile -png -r %i  %s %s'%(density, pdfin, pngout))

        fin_  = 'experiments/case-%i/diagnostic-%04d.png'%(case,nn)
        fout_ = 'experiments/case-%i/frames/diagnostic-%04d.png'%(case,ii)
#        os.system('cp %s %s'%(fin_, fout_))
        os.system('convert %s -gravity South -chop 0x990 %s'%(fin_, fout_))


os.system('rm %s.avi'%(fout))
os.system('ffmpeg -y -f image2 -framerate %i -stream_loop 0 -i experiments/case-%i/frames/diagnostic-%s.png -vf scale=1024:-1 -vcodec libx264 -crf 20  -pix_fmt yuv420p %s.avi'%(framerate, case, numfmt, fout))

os.system('rm %s.gif'%(fout))
os.system('ffmpeg -i %s.avi -vf "fps=%i,scale=%i:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 %s.gif'%(fout, 3, scale, fout))
