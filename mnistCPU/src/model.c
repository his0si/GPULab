/*
 * =====================================================================================
 *
 *       Filename:  model.c
 *
 *    Description: To load weight from files and execute inference
 *
 *        Version:  1.0
 *        Created:  07/29/2024 12:46:39 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Myung Kuk Yoon (MK), myungkuk.yoon@ewha.ac.kr
 *   Organization:  Department of Computer Science and Engineering
 *
 * =====================================================================================
 */

#include "model.h"
#include <algorithm> // std::max 사용을 위해 추가

model::model(unsigned num_layers){
	m_num_layers = num_layers;
}

model::~model(){
	for(auto iter = m_layers.begin(); iter != m_layers.end(); iter++){
		free((*iter)->weights);
		free((*iter)->bias);
		free((*iter));
	}
	m_layers.clear();
}

bool model::read_weights(const char *file, bool activation){
	bool ret = true;
	FILE *fp = fopen(file, "r");
	
	unsigned x = 0, y = 0;
	fLayer *layer = nullptr;
	//u2f tmp;


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
	}

	layer->num_bias[0] = y;
	layer->num_bias[1] = x;
	layer->bias = (u2f *)malloc(sizeof(u2f) * y * x);

	for(int i = 0; i < y; i++){
		fscanf(fp, "0x%x ", &layer->bias[i].uint);
	}
	
	assert(layer->num_weight[1] == layer->num_bias[0]);
	layer->result = (float *)malloc(sizeof(float) * layer->num_weight[1]);
	
	layer->perf_act = activation;

	m_layers.push_back(layer);
	printf("Loaded Layer %u = W:(%u x %u) + B:(%u x %u)\n", (unsigned)m_layers.size(), layer->num_weight[0], layer->num_weight[1], layer->num_bias[0], layer->num_bias[1]);

cleanup:
	fclose(fp);
	
	return ret;
}

unsigned char model::perf_forward_exec(m_data *img){
	float *input = img->nor_data.oneD;
	return perf_forward_exec(input);
}

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

void model::perf_fc_exec(fLayer *layer, float *input){
    unsigned input_size = layer->num_weight[0];
    unsigned output_size = layer->num_weight[1];
    
    // 각 출력 뉴런에 대해 계산
    for(unsigned k = 0; k < output_size; k++){
        float sum = 0.0f;
        // 입력 벡터와 가중치의 내적 계산
        for(unsigned i = 0; i < input_size; i++){
            sum += (input[i] * layer->weights[k * input_size + i].fp);
        }
        // 편향 추가
        sum += layer->bias[k].fp;
        // 결과 저장
        layer->result[k] = sum;
		//printf("%u - %f\n", k, sum);
    }
}

void model::perf_act_exec(fLayer *layer){
    unsigned size = layer->num_weight[1];
    // ReLU 활성화 함수 적용: f(x) = max(0, x)
    for(unsigned i = 0; i < size; i++){
        layer->result[i] = std::max(0.0f, layer->result[i]);
		//printf("%f\n", layer->result[i]);
    }
}