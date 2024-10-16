/*
 * =====================================================================================
 *
 *       Filename:  main.cu
 *
 *    Description:  image blur cpu code
 *
 *        Version:  1.0
 *        Created:  07/14/2021 10:41:21 PM
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Yoon, Myung Kuk, myungkuk.yoon@ewha.ac.kr
 *   Organization:  EWHA Womans University
 *
 * =====================================================================================
 */

#include <iostream>
#include <cstdlib>      // For malloc and free
#include "ppm.h"
#include "clockMeasure.h"

#define BLUR_SIZE 5

using namespace std;

const int MAX_ITER = 10;

// 박스 블러를 수행하는 함수
void cpuCode(unsigned char *outArray, const unsigned char *inArray, const int w, const int h){
    int radius = BLUR_SIZE / 2;

    for(int y = 0; y < h; y++){
        for(int x = 0; x < w; x++){
            int sumR = 0, sumG = 0, sumB = 0;
            int count = 0;

            // 블러 윈도우 내의 픽셀들을 순회
            for(int dy = -radius; dy <= radius; dy++){
                for(int dx = -radius; dx <= radius; dx++){
                    // 현재 픽셀의 좌표 계산
                    int ny = y + dy;
                    int nx = x + dx;

                    // 이미지 경계를 벗어나지 않는지 확인
                    if(ny >= 0 && ny < h && nx >= 0 && nx < w){
                        int idx = (ny * w + nx) * 3;
                        sumR += inArray[idx];
                        sumG += inArray[idx + 1];
                        sumB += inArray[idx + 2];
                        count++;
                    }
                }
            }

            // 평균값 계산 및 출력 배열에 저장
            int outIdx = (y * w + x) * 3;
            outArray[outIdx]     = static_cast<unsigned char>(sumR / count);
            outArray[outIdx + 1] = static_cast<unsigned char>(sumG / count);
            outArray[outIdx + 2] = static_cast<unsigned char>(sumB / count);
        }
    }
}

int main(){
    int w, h;
    unsigned char *h_imageArray = nullptr;
    unsigned char *h_outImageArray = nullptr;

    // PPM 파일에서 이미지 데이터를 로드
    ppmLoad("./data/ewha_picture.ppm", &h_imageArray, &w, &h);
    if(h_imageArray == nullptr){
        cerr << "Failed to load the PPM image." << endl;
        return EXIT_FAILURE;
    }

    // 출력 이미지 배열 메모리 할당 (R, G, B 각각 1바이트씩)
    h_outImageArray = (unsigned char*)malloc(w * h * 3 * sizeof(unsigned char));
    if(h_outImageArray == nullptr){
        cerr << "Failed to allocate memory for output image." << endl;
        free(h_imageArray); // 입력 이미지 메모리 해제
        return EXIT_FAILURE;
    }

    // clockMeasure 객체 생성
    clockMeasure *ckCpu = new clockMeasure("CPU CODE");

    ckCpu->clockReset();

    // 블러 연산을 MAX_ITER 횟수만큼 반복
    for(int i = 0; i < MAX_ITER; i++){
        ckCpu->clockResume();
        cpuCode(h_outImageArray, h_imageArray, w, h); 
        ckCpu->clockPause();
    }

    // 실행 시간 출력
    ckCpu->clockPrint();

    // 결과 이미지를 PPM 파일로 저장
    ppmSave("ewha_picture_cpu.ppm", h_outImageArray, w, h);

    // 동적으로 할당한 메모리 해제
    free(h_imageArray);
    free(h_outImageArray);

    // clockMeasure 객체 해제
    delete ckCpu;

    return 0;
}
