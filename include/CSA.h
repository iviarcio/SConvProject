#ifndef CSAINFO_H
#define CSAINFO_H

#include <stdint.h>

typedef enum { IS = 1, WS } Scheduling;

typedef struct {
  int64_t input_channels;
  int64_t output_rows;
  int64_t output_cols;
  int64_t kernel_rows;
  int64_t kernel_cols;
  int64_t num_filters;
  uint8_t data_size; // bytes -- 4 (4B)
} ConvInfo;

typedef struct {
  uint32_t l1_size;     // bytes -- 32768 (32KiB)
  uint32_t l2_size;     // bytes -- 262144 (256KiB)
  uint32_t l3_size;     // bytes -- 1572864 (1.5MiB)
  uint32_t l1_latency;  // cycles -- e.g 4
  uint32_t l2_latency;  // cycles -- e.g 20
  uint32_t l3_latency;  // cycles -- e.g 50
  uint32_t mem_latency; // cycles -- e.g 150
  uint32_t cache_line;  // bytes -- 64B
} ArchInfo;

typedef struct {
  uint8_t nwindows;    // 16 for MMA
  uint8_t num_filters; // 8 for MMA
  uint16_t noutput;    // 128 for MMA
} mKInfo;

typedef struct {
  Scheduling schd;
  uint32_t k2;
  uint32_t extra_k2;
  uint32_t k3;
  uint32_t extra_k3;
  uint32_t tile_c;
  uint32_t extra_tile_c;
} CSAStrategy;

class CSA {
public:
  CSA(ArchInfo &arch, ConvInfo &conv, mKInfo mK)
      : arch_(arch), conv_(conv), mK_(mK) {}

  CSAStrategy operator()();

  ArchInfo &arch_;
  ConvInfo &conv_;
  mKInfo mK_;
};

CSA createCSAPass(ConvInfo &conv);

#endif
