#include "model.h"

#define MAX_BATCH_SIZE 1000 // 최대 배치 크기를 정의합니다.

model::model(unsigned num_layers){
    m_num_layers = num_layers;
}

model::~model(){
    for(auto iter = m_layers.begin(); iter != m_layers.end(); iter++){
        free((*iter)->weights);
        free((*iter)->bias);
        free((*iter)->result);

        cudaFree((*iter)->d_weights);
        cudaFree((*iter)->d_bias);
        cudaFree((*iter)->d_result);

        free((*iter));
    }
    m_layers.clear();
}

bool model::read_weights(const char *file, bool activation){
    bool ret = true;
    FILE *fp = fopen(file, "r");
    
    unsigned x = 0, y = 0;
    fLayer *layer = nullptr;

    if(!fp){
        printf("[ERROR] Weight Input file does not exist\n");
        ret = false;
        goto cleanup;
    }

    fscanf(fp, "%u x %u\n", &y, &x);

    if(!x && !y){
        printf("[ERROR] On weight dimension\n");
        ret = false;
        goto cleanup;
    }

    layer = (fLayer *)malloc(sizeof(fLayer));
    layer->num_weight[0] = y;
    layer->num_weight[1] = x;
    layer->weights = (u2f *)malloc(sizeof(u2f) * x * y);

    for(int i = 0; i < y; i++){
        for(int j = 0; j < x; j++){
            fscanf(fp, "0x%x ", &layer->weights[j * y + i].uint);
        }
    }

    fscanf(fp, "%u x %u\n", &y, &x);
    if((!x && !y) || x != 1){ 
        printf("[ERROR] On bias dimension\n");
        free(layer->weights);
        free(layer);
        ret = false;
        goto cleanup;
    }

    layer->num_bias[0] = y;
    layer->num_bias[1] = x;
    layer->bias = (u2f *)malloc(sizeof(u2f) * y * x);

    for(int i = 0; i < y; i++){
        fscanf(fp, "0x%x ", &layer->bias[i].uint);
    }
    
    assert(layer->num_weight[1] == layer->num_bias[0]);
    // 결과를 배치 크기에 맞게 할당합니다.
    layer->result = (float *)malloc(sizeof(float) * layer->num_weight[1]
                                    * MAX_BATCH_SIZE);
    
    layer->perf_act = activation;

    m_layers.push_back(layer);
    printf("Loaded Layer %u = W:(%u x %u) + B:(%u x %u)\n",
           (unsigned)m_layers.size(), layer->num_weight[0],
           layer->num_weight[1], layer->num_bias[0], layer->num_bias[1]);

    // 호스트에서 디바이스로 데이터 복사
    this->copy_weights_into_device(layer);

cleanup:
    fclose(fp);
    
    return ret;
}

void model::copy_weights_into_device(fLayer *layer){
    layer->weightSize = sizeof(float) * layer->num_weight[0]
                        * layer->num_weight[1];
    cudaError_t err = cudaMalloc((void **)&layer->d_weights, layer->weightSize);
    checkCudaError(err);
    layer->biasSize = sizeof(float) * layer->num_bias[0] * layer->num_bias[1];
    err = cudaMalloc((void **)&layer->d_bias, layer->biasSize);
    checkCudaError(err);

    // 결과 버퍼를 배치 크기에 맞게 할당합니다.
    layer->resultSize = sizeof(float) * layer->num_weight[1] * MAX_BATCH_SIZE;
    err = cudaMalloc((void **)&layer->d_result, layer->resultSize);
    checkCudaError(err);

    err = cudaMemcpy(layer->d_weights, layer->weights, layer->weightSize,
                     cudaMemcpyHostToDevice);
    checkCudaError(err);
    err = cudaMemcpy(layer->d_bias, layer->bias, layer->biasSize,
                     cudaMemcpyHostToDevice);
    checkCudaError(err);
}

__global__
void perf_fc_exec_device(float *input, float *weight, float *bias,
                         float *result, const unsigned weightSizeY,
                         const unsigned weightSizeX, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = weightSizeY * batch_size;
    if (idx < total_outputs) {
        int b = idx / weightSizeY; // 배치 인덱스
        int w = idx % weightSizeY; // 출력 뉴런 인덱스
        float sum = bias[w];
        for (int j = 0; j < weightSizeX; ++j) {
            sum += input[b * weightSizeX + j]
                   * weight[w * weightSizeX + j];
        }
        result[b * weightSizeY + w] = sum;
    }
}

__global__
void perf_act_exec_device(float *input, unsigned inputSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < inputSize) {
        input[idx] = max(0.0f, input[idx]);
    }
}

unsigned char* model::perf_forward_exec_on_device_batch(float *h_input_batch,
                                                        int batch_size){
    // 배치 입력 데이터를 디바이스로 복사합니다.
    unsigned inputSize = sizeof(float) * IMG_SIZE * IMG_SIZE * batch_size;
    float *d_input_batch;
    cudaError_t err = cudaMalloc((void **)&d_input_batch, inputSize);
    checkCudaError(err);

    err = cudaMemcpy(d_input_batch, h_input_batch, inputSize,
                     cudaMemcpyHostToDevice);
    checkCudaError(err);

    float *input = d_input_batch;
    for(auto iter = m_layers.begin(); iter != m_layers.end(); iter++){
        const unsigned tbSize = 128;
        dim3 blockSize(tbSize, 1, 1);
        unsigned total_outputs = (*iter)->num_weight[1] * batch_size;
        dim3 gridSize((total_outputs + tbSize - 1) / tbSize, 1, 1);

        // 완전 연결 계층 실행
        perf_fc_exec_device<<<gridSize, blockSize>>>(
            input, (*iter)->d_weights, (*iter)->d_bias, (*iter)->d_result,
            (*iter)->num_weight[1], (*iter)->num_weight[0], batch_size);
        err = cudaGetLastError();
        checkCudaError(err);

        if((*iter)->perf_act){
            // 활성화 함수 적용
            perf_act_exec_device<<<gridSize, blockSize>>>(
                (*iter)->d_result, total_outputs);
            err = cudaGetLastError();
            checkCudaError(err);
        }
        err = cudaDeviceSynchronize();
        checkCudaError(err);
        input = (*iter)->d_result;
    }

    // 결과를 호스트로 복사합니다.
    float *h_output = new float[10 * batch_size];
    err = cudaMemcpy(h_output, input, sizeof(float) * 10 * batch_size,
                     cudaMemcpyDeviceToHost);
    checkCudaError(err);

    // 각 입력에 대한 예측 레이블을 찾습니다.
    unsigned char *pred_labels = new unsigned char[batch_size];
    for(int b = 0; b < batch_size; b++){
        unsigned char largestIdx = 0;
        float largestO = h_output[b*10 + 0];
        for(unsigned char i = 1; i < 10; i++){
            if(h_output[b*10 + i] > largestO){
                largestIdx = i;
                largestO = h_output[b*10 + i];
            }
        }
        pred_labels[b] = largestIdx;
    }

    // 메모리 해제
    delete[] h_output;
    cudaFree(d_input_batch);

    return pred_labels; // 예측 레이블 배열을 반환합니다.
}

// 누락된 함수 정의 추가
unsigned char model::perf_forward_exec(m_data *img){
    float *input = img->nor_data.oneD;
    return perf_forward_exec(input);
}

// 기존의 perf_forward_exec(float *input) 함수도 정의되어 있어야 합니다.
unsigned char model::perf_forward_exec(float *input){
    for(auto iter = m_layers.begin(); iter != m_layers.end(); iter++){
        perf_fc_exec((*iter), input);    
        if((*iter)->perf_act) perf_act_exec((*iter));
        input = (*iter)->result;
    }
    unsigned char largestIdx = 0;
    float largestO = input[0];
    for(unsigned char i = 1; i < 10; i++){
        if(input[i] > largestO){
            largestIdx = i;
            largestO = input[i];
        }
    }
    return largestIdx;
}

// CPU에서 사용하는 perf_fc_exec와 perf_act_exec 함수도 정의되어 있어야 합니다.
void model::perf_fc_exec(fLayer *layer, float *img){
    for(int w = 0; w < layer->num_weight[1]; w++){
        layer->result[w] = layer->bias[w].fp;
    }

    for(int i = 0; i < layer->num_weight[1]; i++){
        for(int w = 0; w < layer->num_weight[0]; w++){
            unsigned idx = i * layer->num_weight[0] + w;
            layer->result[i] += img[w] * layer->weights[idx].fp;
        }    
    }
}

void model::perf_act_exec(fLayer *layer){
    for(int w = 0; w < layer->num_weight[1]; w++){
        if(layer->result[w] < 0.0){
            layer->result[w] = 0.0f;
        }
    }
}
