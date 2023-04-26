/*
Author: Hector Piteau (hector.piteau@gmail.com)
CudaVolume3DBuffer.hpp (c) 2023
Desc: CudaVolume3DBuffer
Created:  2023-04-23T09:19:04.460Z
Modified: 2023-04-23T09:21:39.116Z
*/
#include "CudaBuffer.hpp"

class CudaVolume3DBuffer
{
private:
    CudaBuffer<int>* m_indPool0;
    CudaBuffer<int>* m_indPool1;
    CudaBuffer<float>* m_dataBuffer;

public:
    CudaVolume3DBuffer(/* args */);
    ~CudaVolume3DBuffer();
};