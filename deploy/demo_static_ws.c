/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <tvm/runtime/c_runtime_api.h>
#include <string.h>
#include <time.h>

#include "bundle.h"

extern const char build_graph_c_json[];
extern unsigned int build_graph_c_json_len;

extern const char build_params_c_bin[];
extern unsigned int build_params_c_bin_len;

#define OUTPUT_LEN 3

#define LOADING_QUEUE_FILE "loadingQueue.txt"
#define FINISH_QUEUE_FILE "finishQueue.txt"
#define RESULT_FILE "result.csv"
#define MAX_RESULTS 10
#define BUFFER_SIZE 1024

int getLargestId();
void registerLoadingId(int id);
void registerFinishId(int id);
double generateRandomWeight();
void getCurrentTime(char *buffer, size_t bufferSize);
void insertResult(int id, const char **nameList, const double *confidenceList, double weight, int count);
void insertResult(int id, const char **nameList, const double *confidenceList, double weight, int count);
void insertResult(int id, const char **nameList, const double *confidenceList, double weight, int count);
void beginRecognition();
void finishRecognition(const char **nameList, const double *confidenceList, double weight, int count);

const char *nameList[] = {"Apple", "Banana", "Cherry"};

int main(int argc, char** argv) {
  srand(time(NULL));
  beginRecognition();
  assert(argc == 2 && "Usage: demo_static <cat.bin>");

  char* json_data = (char*)(build_graph_c_json);
  char* params_data = (char*)(build_params_c_bin);
  uint64_t params_size = build_params_c_bin_len;

  void* handle = tvm_runtime_create(json_data, params_data, params_size, argv[0]);

  float input_storage[1 * 3 * 224 * 224];
  FILE* fp = fopen(argv[1], "rb");
  (void)fread(input_storage, 3 * 224 * 224, 4, fp);
  fclose(fp);

  DLTensor input;
  input.data = input_storage;
  DLDevice dev = {kDLCPU, 0};
  input.device = dev;
  input.ndim = 4;
  DLDataType dtype = {kDLFloat, 32, 1};
  input.dtype = dtype;
  int64_t shape[4] = {1, 3, 224, 224};
  input.shape = shape;
  input.strides = NULL;
  input.byte_offset = 0;

  tvm_runtime_set_input(handle, "data", &input);

  tvm_runtime_run(handle);

  float output_storage[OUTPUT_LEN];
  DLTensor output;
  output.data = output_storage;
  DLDevice out_dev = {kDLCPU, 0};
  output.device = out_dev;
  output.ndim = 2;
  DLDataType out_dtype = {kDLFloat, 32, 1};
  output.dtype = out_dtype;
  int64_t out_shape[2] = {1, OUTPUT_LEN};
  output.shape = out_shape;
  output.strides = NULL;
  output.byte_offset = 0;

  tvm_runtime_get_output(handle, 0, &output);

  float max_iter = -FLT_MAX;
  int32_t max_index = -1;
  for (int i = 0; i < OUTPUT_LEN; ++i) {
    if (output_storage[i] > max_iter) {
      max_iter = output_storage[i];
      max_index = i;
    }
  }

  double confidenceList[OUTPUT_LEN];

  for (int i = 0; i < OUTPUT_LEN; i++) {
    confidenceList[i] = (double)output_storage[i];
  }

  tvm_runtime_destroy(handle);

  printf("The maximum position in output vector is: %d, with max-value %f.\n", max_index, max_iter);

  double weight = generateRandomWeight();
  int count = sizeof(nameList) / sizeof(nameList[0]);

  finishRecognition(nameList, confidenceList, weight, count);

  return 0;
}

int getLargestId() {
    FILE *file = fopen(RESULT_FILE, "r");
    if (!file) {
        perror("Error opening result file");
        return -1;
    }

    int largestId = 0, id;
    char buffer[BUFFER_SIZE];
    
    while (fgets(buffer, sizeof(buffer), file)) {
        sscanf(buffer, "%d,", &id);
        if (id > largestId) {
            largestId = id;
        }
    }

    fclose(file);
    return largestId;
}

void registerLoadingId(int id) {
    FILE *file = fopen(LOADING_QUEUE_FILE, "a");
    if (!file) {
        perror("Error opening loading queue file");
        return;
    }

    fprintf(file, "%d\n", id);
    fclose(file);
}

void registerFinishId(int id) {
    FILE *file = fopen(FINISH_QUEUE_FILE, "a");
    if (!file) {
        perror("Error opening finish queue file");
        return;
    }

    fprintf(file, "%d\n", id);
    fclose(file);
}

double generateRandomConfidence() {
    return ((double)rand() / RAND_MAX) * 0.9 + 0.1;
}

double generateRandomWeight() {
    return ((double)rand() / RAND_MAX) * 100 + 0.5; // Random weight between 0.5 and 100.5
}

void getCurrentTime(char *buffer, size_t bufferSize) {
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    strftime(buffer, bufferSize, "%Y-%m-%dT%H:%M:%S", tm_info);
}

void insertResult(int id, const char **nameList, const double *confidenceList, double weight, int count) {
    FILE *file = fopen(RESULT_FILE, "a");
    if (!file) {
        perror("Error opening result file");
        return;
    }
    char timeStr[25];
    getCurrentTime(timeStr, sizeof(timeStr));

    char results[BUFFER_SIZE] = "";
    for (int i = 0; i < count; i++) {
        char result[100];
        snprintf(result, sizeof(result), "%s:%.2f", nameList[i], confidenceList[i]);
        strcat(results, result);
        if (i < count - 1) {
            strcat(results, "; ");
        }
    }

    int isConfirmed = 0;
    const char *correctedItemName = "";

    fprintf(file, "%d,%s,%s,%.2f,%d,%s\n", id, results, timeStr, weight, isConfirmed, correctedItemName);
    fclose(file);

    printf("Result updated for ID %d: %s, time %s\n", id, results, timeStr);
}

void beginRecognition() {
    int largestId = getLargestId();
    int newId = largestId + 1;

    registerLoadingId(newId);
    printf("Begin recognition for ID %d\n", newId);
}

void finishRecognition(const char **nameList, const double *confidenceList, double weight, int count) {
    int largestId = getLargestId();
    int newId = largestId + 1;

    insertResult(newId, nameList, confidenceList, weight, count);
    registerFinishId(newId);
    printf("Finish recognition for ID %d\n", newId);
}