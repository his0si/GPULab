/*
 * =====================================================================================
 *
 *       Filename:  main.cu
 *
 *    Description:  Fully-Connected MNIST
 *
 *        Version:  1.0
 *        Created:  07/08/2024 09:54:25 AM
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  KIM HEE SEO
		   ID    :  2276093
 *   Organization:  EWHA Womans University
 *
 * =====================================================================================
 */

#include "./src/predefine.h"
#include "./src/dataReader.h"
#include "./src/model.h"
#include "./src/clockMeasure.h"

const char *train_image = "./data/train-images";
const char *train_label = "./data/train-labels";
const char *test_image = "./data/test-images";
const char *test_label = "./data/test-labels";

int main(){
    // 학습 및 테스트 이미지 로드
    dataReader *reader = new dataReader(train_image, train_label, test_image, test_label);

    reader->print_file_info();

    reader->read_train_files();
    reader->read_test_files();

    reader->calculate_std_mean();
    reader->apply_nor_into_trainDB();
    reader->apply_nor_into_testDB();

    // 이미지와 레이블 출력
    reader->print_image_and_label(10, false); 

    // 완전 연결 신경망 생성 (가중치 읽기)
    model *fModel = new model(2);
    fModel->read_weights("./layer/Layer01.db");
    fModel->read_weights("./layer/Layer02.db", false);

    clockMeasure *ckCpu = new clockMeasure("CPU CODE");
    ckCpu->clockReset();
    clockMeasure *ckGpu = new clockMeasure("GPU CODE");
    ckGpu->clockReset();

    // 메인 루프 10번 실행
    for(unsigned i = 0; i < 1; i++){
        // CPU 추론 시작 시간 측정
        ckCpu->clockResume();
        // 서브 루프 1000번 실행
        for(int j = 0; j < 1000; j++){
            unsigned idx = i * 1000 + j;
            m_data *img = reader->get_mnist_db(idx, false);
            unsigned cpu_ret = fModel->perf_forward_exec(img);
        }
        // CPU 추론 종료 시간 측정
        ckCpu->clockPause();

        // GPU 추론 시작 시간 측정
        ckGpu->clockResume();
        // 배치 입력 생성
        float *batch_input = new float[1000 * IMG_SIZE * IMG_SIZE];
        for(int j = 0; j < 1000; j++){
            unsigned idx = i * 1000 + j;
            m_data *img = reader->get_mnist_db(idx, false);
            memcpy(batch_input + j * IMG_SIZE * IMG_SIZE, img->nor_data.oneD, sizeof(float) * IMG_SIZE * IMG_SIZE);
        }
        // GPU에서 배치 추론 실행
        unsigned char *gpu_rets = fModel->perf_forward_exec_on_device_batch(batch_input, 1000);
        // GPU 추론 종료 시간 측정
        ckGpu->clockPause();

        // 메모리 해제
        delete[] batch_input;
        delete[] gpu_rets;
    }

    ckCpu->clockPrint();
    ckGpu->clockPrint();

    return 0;
}
