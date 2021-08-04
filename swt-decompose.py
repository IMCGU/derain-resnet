import numpy as np
import matplotlib.pyplot as plt
import pywt
import pywt.data
import cv2
from PIL import Image

def batch_Swt(batch, level=1):
    '''
    Args:
        batch: Input batch RGB image [batch_size, img_h, img_w, 3]
    Returns:
        dwt_batch: Batch  DWT result [batch_size, img_h, img_w, 12]
    '''
    # print(len(batch.shape))
    assert (len(batch.shape) == 4 ),"Input batch Shape error"
    assert (batch.shape[3] == 3 ),"Color channel error"

    Swt_batch = np.zeros([batch.shape[0], batch.shape[1], batch.shape[2], 12*level])

    for i in range(batch.shape[0]):
        coeffs_R = pywt.swt2(batch[i,:,:,0], 'haar', level=level)
        coeffs_G = pywt.swt2(batch[i,:,:,1], 'haar', level=level)
        coeffs_B = pywt.swt2(batch[i,:,:,2], 'haar', level=level)

        for k in range(level):
            # print((level-k)-1)
            (LL_r, (LH_r, HL_r, HH_r)) = coeffs_R[level-k-1]
            (LL_g, (LH_g, HL_g, HH_g)) = coeffs_G[level-k-1]
            (LL_b, (LH_b, HL_b, HH_b)) = coeffs_B[level-k-1]

            coeffs_stack_R = np.stack([LL_r,LH_r, HL_r, HH_r], axis=-1)
            coeffs_stack_G = np.stack([LL_g,LH_g, HL_g, HH_g], axis=-1)
            coeffs_stack_B = np.stack([LL_b,LH_b, HL_b, HH_b], axis=-1)

            # print(coeffs_stack_R.shape)
            coeffs= np.concatenate([coeffs_stack_R, coeffs_stack_G, coeffs_stack_B], axis=-1)

            Swt_batch[i,:,:,12*k:12*(k+1)] = coeffs

    return Swt_batch

def batch_ISwt(batch):
    '''
    Args:
        batch: Input batch RGB image [batch_size, img_h, img_w, 3]
    Returns:
        dwt_batch: Batch  DWT result [batch_size, img_h, img_w, 12]
    '''
    # print(len(batch.shape))
    # assert (len(batch.shape) == 4 ),"Input batch Shape error"
    # assert (batch.shape[3] == 3 ),"Color channel error"

    Swt_batch = np.zeros([batch.shape[0], batch.shape[1], batch.shape[2], 3])



    for i in range(batch.shape[0]):
        Iswt_level_1_R = (batch[i,:,:,0],(batch[i,:,:,1],batch[i,:,:,2],batch[i,:,:,3]))
        Iswt_level_2_R = (batch[i,:,:,12],(batch[i,:,:,13],batch[i,:,:,14],batch[i,:,:,15]))

        Iswt_level_1_G = (batch[i,:,:,4],(batch[i,:,:,5],batch[i,:,:,6],batch[i,:,:,7]))
        Iswt_level_2_G = (batch[i,:,:,16],(batch[i,:,:,17],batch[i,:,:,18],batch[i,:,:,19]))        

        Iswt_level_1_B = (batch[i,:,:,8],(batch[i,:,:,9],batch[i,:,:,10],batch[i,:,:,11]))
        Iswt_level_2_B = (batch[i,:,:,20],(batch[i,:,:,21],batch[i,:,:,22],batch[i,:,:,23]))

        Iswt_R = pywt.iswt2([Iswt_level_2_R,Iswt_level_1_R], wavelet='haar')
        Iswt_G = pywt.iswt2([Iswt_level_2_G,Iswt_level_1_G], wavelet='haar')
        Iswt_B = pywt.iswt2([Iswt_level_2_B,Iswt_level_1_B], wavelet='haar')

        coeffs = cv2.merge([Iswt_R, Iswt_G, Iswt_B])
        Swt_batch[i,:,:,:] = coeffs


    return Swt_batch

# def batch_Swt(batch):
#     '''
#     Args:
#         batch: Input batch RGB image [batch_size, img_h, img_w, 3]
#     Returns:
#         dwt_batch: Batch  DWT result [batch_size, img_h, img_w, 12]
#     '''
#     # print(len(batch.shape))
#     assert (len(batch.shape) == 4 ),"Input batch Shape error"
#     assert (batch.shape[3] == 3 ),"Color channel error"

#     Swt_batch = np.zeros([batch.shape[0], batch.shape[1], batch.shape[2], 12])

#     for i in range(batch.shape[0]):
#         (LL_r, (LH_r, HL_r, HH_r)) = pywt.swt2(batch[i,:,:,0], 'haar', level=1)[0]
#         coeffs_R = np.stack([LL_r,LH_r, HL_r, HH_r], axis=-1)

#         (LL_g, (LH_g, HL_g, HH_g)) = pywt.swt2(batch[i,:,:,1], 'haar', level=1)[0]
#         coeffs_G = np.stack([LL_g,LH_g, HL_g, HH_g], axis=-1)

#         (LL_b, (LH_b, HL_b, HH_b)) = pywt.swt2(batch[i,:,:,2], 'haar', level=1)[0]
#         coeffs_B = np.stack([LL_b,LH_b, HL_b, HH_b], axis=-1)

#         coeffs = np.concatenate([coeffs_R, coeffs_G, coeffs_B], axis=-1)

#         Swt_batch[i,:,:,:] = coeffs
#         # print(coeffs.shape)
#     return Swt_batch


# def batch_ISwt(batch):
#     '''
#     Args:
#         batch: Tensor of batch [16,h,w,12]
#     Returns:
#         Idwt_batch: Tensor of Inverse wavelet transform [16,h*2,w*2,3]
#     '''

#     swt_batch = np.zeros([batch.shape[0], batch.shape[1], batch.shape[2], 3])

#     for i in range(batch.shape[0]):
#         Iswt_R = pywt.iswt2((batch[i,:,:,0],(batch[i,:,:,1],batch[i,:,:,2],batch[i,:,:,3])), wavelet='haar')
#         Iswt_G = pywt.iswt2((batch[i,:,:,4],(batch[i,:,:,5],batch[i,:,:,6],batch[i,:,:,7])), wavelet='haar')
#         Iswt_B = pywt.iswt2((batch[i,:,:,8],(batch[i,:,:,9],batch[i,:,:,10],batch[i,:,:,11])), wavelet='haar')

#         coeffs = cv2.merge([Iswt_R, Iswt_G, Iswt_B])
#         swt_batch[i,:,:,:] = coeffs
#         # print(coeffs.shape)
#     return swt_batch


if __name__ == '__main__':
    img = cv2.imread('./img/17_90000_out.png',0) # BGR channel 
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print('Image Shape: %s'%str(img.shape))
    print('Image Shape: ',img.ndim)
    '''看官方文件'''
    # (LL_r, (LH_r, HL_r, HH_r)) = pywt.swt2(img[:,:,0], 'haar', level=1)[0]
    # print(pywt.swt2(img[:,:,0], 'haar', level=1)[0])
    # (LL_r1,(LH_r1,HL_r1,HH_r1)) = pywt.swt2(img[:,:,0], 'haar', level=2)[1]
    # (LL_r2,(LH_r2,HL_r2,HH_r2)) = pywt.swt2(img[:,:,0], 'haar', level=2)[0]
    # (LL_g1,(LH_g1,HL_g1,HH_g1)) = pywt.swt2(img[:,:,1], 'haar', level=2)[1]
    # (LL_g2,(LH_g2,HL_g2,HH_g2)) = pywt.swt2(img[:,:,1], 'haar', level=2)[0]
    # (LL_b1,(LH_b1,HL_b1,HH_b1)) = pywt.swt2(img[:,:,2], 'haar', level=2)[1]
    # (LL_b2,(LH_b2,HL_b2,HH_b2)) = pywt.swt2(img[:,:,2], 'haar', level=2)[0]
    # result_LL1 = np.stack((LL_r1,LL_g1,LL_b1),axis=-1)
    # result_LL1 = np.clip(np.abs(result_LL1),0,255).astype('uint8')
    # result_LL2 = np.stack((LL_r2,LL_g2,LL_b2),axis=-1)
    # result_LL2 = np.clip(np.abs(result_LL2),0,255).astype('uint8')

    # original = img.copy()
    # cv2.imshow('Original', original)
    # cv2.imshow('img', img)

    # print(original[np.newaxis].shape)
    # print(img[np.newaxis].shape)
    # rgb_swt_result = batch_Swt(original[np.newaxis], level=2)
    # rgb_swt_result = batch_Swt(img[np.newaxis], level=2)
    # print(rgb_swt_result.shape)


    # rgb_Iswt_result = batch_ISwt(rgb_swt_result)
    # print(rgb_Iswt_result.shape)
    # rgb_Iswt_result = np.squeeze(rgb_Iswt_result)
    # print(rgb_Iswt_result.shape)
    
    # cv2.imshow('ISWT result', rgb_Iswt_result.astype('uint8'))
    # cv2.waitKey(0)

    coffes = pywt.swt2(img, 'haar', level=2)

    # level_3_coffe = coffes[0]
    # level_2_coffe = coffes[1]
    # level_1_coffe = coffes[2]

    level_2_coffe = coffes[0]
    # print(level_2_coffe)
    level_1_coffe = coffes[1]

    # (LL_level_3, (LH_level_3, HL_level_3, HH_level_3)) = level_3_coffe
    # (LL_level_2, (LH_level_2, HL_level_2, HH_level_2)) = level_2_coffe
    # (LL_level_1, (LH_level_1, HL_level_1, HH_level_1)) = level_1_coffe

    (LL_level_2, (LH_level_2, HL_level_2, HH_level_2)) = level_2_coffe
    (LL_level_1, (LH_level_1, HL_level_1, HH_level_1)) = level_1_coffe

    print(len(coffes))
    # appro_level_3 = np.clip(np.abs(LL_level_3),0,255).astype('uint8')
    # appro_level_2 = np.clip(np.abs(LL_level_2),0,255).astype('uint8')
    # appro_level_1 = np.clip(np.abs(LL_level_1),0,255).astype('uint8')

    # LH_level_3 = np.abs(LH_level_3).astype('uint8')
    # LH_level_2 = np.abs(LH_level_2).astype('uint8')
    # LH_level_1 = np.abs(LH_level_1).astype('uint8')    
    # HL_level_3 = np.abs(HL_level_3).astype('uint8')
    # HL_level_2 = np.abs(HL_level_2).astype('uint8')
    # HL_level_1 = np.abs(HL_level_1).astype('uint8')
    # HH_level_3 = np.abs(HH_level_3).astype('uint8')
    # HH_level_2 = np.abs(HH_level_2).astype('uint8')
    # HH_level_1 = np.abs(HH_level_1).astype('uint8')

    appro_level_2 = np.clip(np.abs(LL_level_2),0,255).astype('uint8')
    appro_level_1 = np.clip(np.abs(LL_level_1),0,255).astype('uint8')
    LH_level_2 = np.abs(LH_level_2).astype('uint8')
    LH_level_1 = np.abs(LH_level_1).astype('uint8')    
    HL_level_2 = np.abs(HL_level_2).astype('uint8')
    HL_level_1 = np.abs(HL_level_1).astype('uint8')
    HH_level_2 = np.abs(HH_level_2).astype('uint8')
    HH_level_1 = np.abs(HH_level_1).astype('uint8')

    # cv2.imwrite('Approximation level-2.jpg', appro_level_2)
    # cv2.imwrite('Approximation level-1.jpg', appro_level_1)    
    # cv2.imwrite('Approximation level-3.jpg', appro_level_3)

    # cv2.imwrite('LH level-3.jpg', LH_level_3)
    # cv2.imwrite('LH level-2.jpg', LH_level_2)
    # cv2.imwrite('LH level-1.jpg', LH_level_1)
    # cv2.imwrite('HL level-3.jpg', HL_level_3)
    # cv2.imwrite('HL level-2.jpg', HL_level_2)
    # cv2.imwrite('HL level-1.jpg', HL_level_1)
    # cv2.imwrite('HH level-3.jpg', HH_level_3)
    # cv2.imwrite('HH level-2.jpg', HH_level_2)
    # cv2.imwrite('HH level-1.jpg', HH_level_1)

    # cv2.imshow('Approximation level-3.jpg', appro_level_3)
    # cv2.imshow('Approximation level-2.jpg', appro_level_2)
    # cv2.imshow('Approximation level-1.jpg', appro_level_1)
    # cv2.imshow('Approximation level-1.jpg', result_LL1)   
    # cv2.imshow('Approximation level-2.jpg', result_LL2)   
    # cv2.imshow('LH level-3.jpg', LH_level_3)
    # cv2.imshow('LH level-2.jpg', LH_level_2)
    # cv2.imshow('LH level-1.jpg', LH_level_1)
    # cv2.imshow('HL level-3.jpg', HL_level_3)
    # cv2.imshow('HL level-2.jpg', HL_level_2)
    # cv2.imshow('HL level-1.jpg', HL_level_1)
    # cv2.imshow('HH level-3.jpg', HH_level_3)
    # cv2.imshow('HH level-2.jpg', HH_level_2)
    # cv2.imshow('HH level-1.jpg', HH_level_1)

    
    # cv2.waitKey(0)


    