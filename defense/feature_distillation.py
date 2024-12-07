import numpy as np
from scipy.fftpack import dct, idct, rfft, irfft

num = 8
q_table = np.ones((num,num))*30
q_table[0:4,0:4] = 25

def dct2 (block):
	return dct(dct(block.T, norm = 'ortho').T, norm = 'ortho')
def idct2(block):
	return idct(idct(block.T, norm = 'ortho').T, norm = 'ortho')
def rfft2 (block):
	return rfft(rfft(block.T).T)
def irfft2(block):
	return irfft(irfft(block.T).T)

def FD_jpeg_encode(input_matrix):
    output = []
    input_matrix = input_matrix * 255

    n = input_matrix.shape[0]
    c = input_matrix.shape[1]
    h = input_matrix.shape[2]
    w = input_matrix.shape[3]
    horizontal_blocks_num = w / num
    output2 = np.zeros((c, h, w))
    output3 = np.zeros((n, 3, h, w))
    vertical_blocks_num = h / num
    n_block = np.split(input_matrix, n, axis=0)
    for i in range(1, n):
        c_block = np.split(n_block[i], c, axis=1)
        j = 0
        for ch_block in c_block:
            vertical_blocks = np.split(ch_block, vertical_blocks_num, axis=2)
            k = 0
            for block_ver in vertical_blocks:
                hor_blocks = np.split(block_ver, horizontal_blocks_num, axis=3)
                m = 0
                for block in hor_blocks:
                    block = np.reshape(block, (num, num))
                    block = dct2(block)
                    # quantization
                    table_quantized = np.matrix.round(np.divide(block, q_table))
                    table_quantized = np.squeeze(np.asarray(table_quantized))
                    # de-quantization
                    table_unquantized = table_quantized * q_table
                    IDCT_table = idct2(table_unquantized)
                    if m == 0:
                        output = IDCT_table
                    else:
                        output = np.concatenate((output, IDCT_table), axis=1)
                    m = m + 1
                if k == 0:
                    output1 = output
                else:
                    output1 = np.concatenate((output1, output), axis=0)
                k = k + 1
            output2[j] = output1
            j = j + 1
        output3[i] = output2

    output3 = output3 / 255
    output3 = np.clip(np.float32(output3), 0.0, 1.0)
    return output3