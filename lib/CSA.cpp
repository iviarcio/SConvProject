#include "CSA.h"

#include <cstdint>
#include <math.h>

#define DEBUG 0
#define MIN(a, b) (a) > (b) ? (a) : (b)

const char *get_schd_name(Scheduling schd) {
  if (schd == IS)
    return "IS";
  else
    return "WS";
}

class Strategies {
public:
  Strategies(ArchInfo &arch, ConvInfo &conv, mKInfo &mK)
      : arch_(arch), conv_(conv), mK_(mK) {}

  virtual Scheduling schd() = 0;
  virtual uint32_t tileSizeL1(uint32_t tileChannels) = 0;
  virtual uint32_t tileSizeL2(uint32_t k2) = 0;
  virtual uint32_t tileSizeL3(uint32_t K3) = 0;
  virtual void computeK2() = 0;
  virtual void computeK3() = 0;
  virtual uint64_t cost_model() = 0;

  uint32_t halfHeuristic(uint32_t initial,
                         uint32_t (Strategies::*func)(uint32_t),
                         uint32_t cache_size) {
    uint32_t solution = initial;
    uint32_t tiles_size = (this->*func)(solution);
    while (tiles_size > cache_size) {
      solution /= 2;
      tiles_size = (this->*func)(solution);
    }
    return solution;
  }

  uint32_t binarySearchHeuristic(uint32_t initial,
                                 uint32_t (Strategies::*func)(uint32_t),
                                 uint32_t cache_size) {
    uint32_t solution = initial;

    // Test initial
    uint32_t tiles_size = (this->*func)(solution);
    if (tiles_size <= cache_size)
      return solution;

    // Bin Search
    uint32_t low = 1, high = initial, mid;
    while (low <= high) {
      mid = low + (high - low) / 2;
      tiles_size = (this->*func)(mid);

      if (tiles_size <= cache_size) {
        solution = mid;
        low = mid + 1;
      } else {
        high = mid - 1;
      }
    }

    return solution;
  }

  void computeTileC() {
    tile_c = (this->*heuristic)(conv_.input_channels, &Strategies::tileSizeL1,
                                arch_.l1_size);
  }

  uint64_t compute() {
    // 1) Identify the number of channels for the IN/W tiles
    // Constraint: |IN_TILE| + |W_TILE| + |OUT_TILE| <= |L1|
    in_size =
        mK_.nwindows * conv_.kernel_rows * conv_.kernel_cols * conv_.data_size;
    w_size = mK_.num_filters * conv_.kernel_rows * conv_.kernel_cols *
             conv_.data_size;
    out_size = mK_.noutput * conv_.data_size;

    computeTileC();
    in_size *= tile_c;
    w_size *= tile_c;

    // 2) Calculate tCH
    tCH = conv_.input_channels / tile_c;
    extra_tCH = conv_.input_channels % tile_c;

    // 3) Calculate the number of W and IN tiles following the mK
    // restrictions
    in_tiles_per_tch = (uint32_t)ceil((conv_.output_rows * conv_.output_cols) /
                                      (double)mK_.nwindows);
    w_tiles_per_tch =
        (uint32_t)ceil(conv_.num_filters / (double)mK_.num_filters);

    // 4) Compute K2 and K3
    computeK2();
    computeK3();

    return cost_model();
  }

  CSAStrategy get_result() {
#if DEBUG > 1
    std::cout << "\nK2: " << k2 << " K2Rem: " << extra_k2 << " K3: " << k3
              << " K3Rem: " << extra_k3 << " schd :" << get_schd_name(schd())
              << " tCH: " << tile_c << " tCHRem: " << extra_tCH;
#endif
#if DEBUG > 0
    uint32_t l2_k = schd() == IS ? k2 : k3;
    uint32_t l3_k = schd() == IS ? k3 : k2;
    std::cout << "\nTile size (L1): " << in_size + w_size + out_size << "("
              << arch_.l1_size << ") Tile size (L2): " << tileSizeL2(l2_k)
              << "(" << arch_.l2_size
              << ") Tile size (L3): " << tileSizeL3(l3_k) << "("
              << arch_.l3_size << ")";
#endif
    return (CSAStrategy){schd(), k2, extra_k2, k3, extra_k3, tile_c, extra_tCH};
  }

protected:
  ArchInfo &arch_;
  ConvInfo &conv_;
  mKInfo &mK_;
  // csa
  uint32_t k2;
  uint32_t k3;
  uint32_t extra_k2;
  uint32_t extra_k3;
  uint32_t tCH;
  uint32_t extra_tCH;
  uint32_t tile_c;
  // others
  uint32_t in_size;
  uint32_t w_size;
  uint32_t out_size;
  uint32_t in_tiles_per_tch;
  uint32_t w_tiles_per_tch;

  ~Strategies() = default;

  // CSA heuristic
  uint32_t (Strategies::*heuristic)(
      uint32_t, uint32_t (Strategies::*)(uint32_t),
      uint32_t) = &Strategies::halfHeuristic;

public:
  // memory accesses
  uint64_t l1;
  uint64_t l2;
  uint64_t l3;
  uint64_t mem;
};

class InputStationary : public Strategies {
public:
  InputStationary(ArchInfo &arch, ConvInfo &conv, mKInfo &mK)
      : Strategies(arch, conv, mK) {}

  Scheduling schd() override { return Scheduling::IS; }

  uint32_t tileSizeL1(uint32_t tileChannels) override {
    uint32_t size = in_size * tileChannels; // in
    size += w_size * tileChannels;          // w
    size += out_size;                       // out
    return size;
  }
  uint32_t tileSizeL2(uint32_t k2) override {
    return (in_size) + (k2 * w_size) + (k2 * out_size);
  }
  uint32_t tileSizeL3(uint32_t k3) override {
    return (k3 * in_size) + (k2 * w_size) + (k2 * k3 * out_size);
  }
  void computeK2() override {
    k2 = (this->*heuristic)(w_tiles_per_tch, &Strategies::tileSizeL2,
                            arch_.l2_size);
    extra_k2 = w_tiles_per_tch % k2;
  }
  void computeK3() override {
    k3 = (this->*heuristic)(in_tiles_per_tch, &Strategies::tileSizeL3,
                            arch_.l3_size);
    extra_k3 = in_tiles_per_tch % k3;
  }
  uint64_t cost_model() override {
    // EQ1 -- first access to any tile comes from memory
    // Sum up all elements and then check the number of cache lines to load
    // them from MEM
    uint64_t in_tiles_total = in_tiles_per_tch * tCH;
    uint64_t w_tiles_total = w_tiles_per_tch * tCH;
    mem =
        (uint64_t)ceil(((in_tiles_total * in_size) + (w_tiles_total * w_size)) /
                       arch_.cache_line);

    // EQ2
    // More reads from MEM are required when the two following constraints
    // are met:
    //   1 - w_tiles_per_tch is smaller than k2
    //   2 -  k3 is smaller than in_tiles_per_tch
    int w_fit = MIN((w_tiles_per_tch / k2) - 1, 1);
    int in_fit = (in_tiles_per_tch / k3) - 1;
    mem += tCH * (uint64_t)ceil(w_fit * in_fit * w_tiles_per_tch * w_size) /
           arch_.cache_line;

    // EQ3
    // This case applies when the number of tiles of filters is greater than
    // k2. If this is the case, then the IN tiles has to be loaded more
    // times from L3
    w_fit = (w_tiles_per_tch / k2) - 1;
    l3 = tCH * (uint64_t)ceil((w_fit * in_tiles_per_tch * in_size) /
                              arch_.cache_line);
    // EQ4
    // For the first IN tile, the W tiles are brought from MEM
    // But for the other IN tiles, the W tiles are brought from L2.
    int ntiles = in_tiles_per_tch - 1;
    l2 = tCH *
         (uint64_t)ceil((ntiles * w_tiles_per_tch * w_size) / arch_.cache_line);

    // EQ5
    l1 = 2 * (uint64_t)(conv_.num_filters * conv_.output_cols *
                        conv_.output_rows * conv_.kernel_cols *
                        conv_.kernel_rows * conv_.input_channels);
    l1 -= (l3 + l2 + mem);

    // EQ6 -- load data back from * to L1
    if (tCH > 1) {
      uint64_t depth_size = (uint64_t)(conv_.input_channels / tCH) *
                            conv_.kernel_cols * conv_.kernel_rows;
      uint64_t access_distance =
          conv_.output_cols * conv_.output_rows * depth_size; // IN
      access_distance += conv_.num_filters * depth_size;      // W
      access_distance +=
          conv_.num_filters * conv_.output_cols * conv_.output_rows; // W
      access_distance *= conv_.data_size;

      uint64_t *m;
      if (access_distance < arch_.l1_size) {
        m = &l1;
      } else if (access_distance < arch_.l2_size) {
        m = &l2;
      } else if (access_distance < arch_.l3_size) {
        m = &l3;
      } else {
        m = &mem;
      }

      uint64_t total_loads_output =
          (tCH - 1) *
          (conv_.num_filters * conv_.output_cols * conv_.output_rows);
      uint64_t total_accessed_cache_lines_output =
          (uint64_t)((total_loads_output * conv_.data_size) / arch_.cache_line);
      *m += total_accessed_cache_lines_output;
      l1 += total_loads_output - total_accessed_cache_lines_output;
    }

    // Latency
    return (l1 * arch_.l1_latency + l2 * arch_.l2_latency +
            l3 * arch_.l3_latency + mem * arch_.mem_latency);
  }
};

class WeightStationary : public Strategies {
public:
  WeightStationary(ArchInfo &arch, ConvInfo &conv, mKInfo &mK)
      : Strategies(arch, conv, mK) {}

  Scheduling schd() override { return Scheduling::WS; }

  uint32_t tileSizeL1(uint32_t tileChannels) override {
    int size = in_size * tileChannels; // in
    size += w_size * tileChannels;     // w
    size += out_size;                  // out
    return size;
  }
  uint32_t tileSizeL2(uint32_t k2) override {
    return (k2 * in_size) + (w_size) + (k2 * out_size);
  }
  uint32_t tileSizeL3(uint32_t k3) override {
    return (k2 * in_size) + (k3 * w_size) + (k2 * k3 * out_size);
  }
  void computeK2() override {
    k2 = (this->*heuristic)(in_tiles_per_tch, &Strategies::tileSizeL2,
                            arch_.l2_size);
    extra_k2 = in_tiles_per_tch % k2;
  }
  void computeK3() override {
    k3 = (this->*heuristic)(w_tiles_per_tch, &Strategies::tileSizeL3,
                            arch_.l3_size);
    extra_k3 = w_tiles_per_tch % k3;
  }
  uint64_t cost_model() override {
    // EQ1
    uint64_t in_tiles_total = in_tiles_per_tch * tCH;
    uint64_t w_tiles_total = w_tiles_per_tch * tCH;
    mem =
        (uint64_t)ceil(((in_tiles_total * in_size) + (w_tiles_total * w_size)) /
                       arch_.cache_line);

    // EQ2
    int in_fit = MIN((in_tiles_per_tch / k2) - 1, 1);
    int w_fit = (w_tiles_per_tch / k3) - 1;
    mem += tCH * (uint64_t)ceil(in_fit * w_fit * in_tiles_per_tch * in_size) /
           arch_.cache_line;

    // EQ3
    in_fit = (in_tiles_per_tch / k2) - 1;
    l3 = tCH *
         (uint64_t)ceil((in_fit * w_tiles_per_tch * w_size) / arch_.cache_line);
    // EQ4
    uint32_t ntiles = w_tiles_per_tch - 1;
    l2 = tCH * (uint64_t)ceil((ntiles * in_tiles_per_tch * in_size) /
                              arch_.cache_line);

    // EQ5
    l1 = 2 * (uint64_t)(conv_.num_filters * conv_.output_cols *
                        conv_.output_rows * conv_.kernel_cols *
                        conv_.kernel_rows * conv_.input_channels);
    l1 -= (l3 + l2 + mem);

    // EQ 6 -- load data back from * to L1
    if (tCH > 1) {
      uint64_t depth_size = (uint64_t)(conv_.input_channels / tCH) *
                            conv_.kernel_cols * conv_.kernel_rows;
      uint64_t access_distance =
          conv_.output_cols * conv_.output_rows * depth_size; // IN
      access_distance += conv_.num_filters * depth_size;      // W
      access_distance +=
          conv_.num_filters * conv_.output_cols * conv_.output_rows; // W
      access_distance *= conv_.data_size;

      uint64_t *m;
      if (access_distance < arch_.l1_size) {
        m = &l1;
      } else if (access_distance < arch_.l2_size) {
        m = &l2;
      } else if (access_distance < arch_.l3_size) {
        m = &l3;
      } else {
        m = &mem;
      }

      uint64_t total_loads_output =
          (tCH - 1) *
          (conv_.num_filters * conv_.output_cols * conv_.output_rows);
      uint64_t total_accessed_cache_lines_output =
          (uint64_t)((total_loads_output * conv_.data_size) / arch_.cache_line);
      *m += total_accessed_cache_lines_output;
      l1 += total_loads_output - total_accessed_cache_lines_output;
    }

    // Latency
    return (l1 * arch_.l1_latency + l2 * arch_.l2_latency +
            l3 * arch_.l3_latency + mem * arch_.mem_latency);
  }
};

CSAStrategy CSA::operator()() {
  InputStationary *is = new InputStationary(arch_, conv_, mK_);
  WeightStationary *ws = new WeightStationary(arch_, conv_, mK_);

  uint64_t cost_is = is->compute();
  uint64_t cost_ws = ws->compute();

  if (cost_ws > cost_is) {
    return is->get_result();
  } else {
    return ws->get_result();
  }
}

CSA createCSAPass(ConvInfo &conv) {
  ArchInfo arch = {
      (uint32_t)(32768 * 0.9),   /* 32KB */
      (uint32_t)(1048576 * 0.9), /* 1MB */
      (uint32_t)(4194304 * 0.9), /* 4MB */
      2,                         /* Latency L1 */
      10,                        /* Latency L2 */
      30,                        /* Latency L3 */
      300,                       /* Latency MEM */
      128                        /* Cache Line Size */
  };

  mKInfo mK = {16, 8, 128};
  return CSA(arch, conv, mK);
}
