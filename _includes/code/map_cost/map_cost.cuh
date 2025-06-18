/**
 * @file map_cost.cuh
 * @brief simple Map Cost
 * @author Bogdan Vlahov
 * @version 0.0.1
 * @date 2025-06-10
 */
#pragma once
#include <mppi/cost_functions/cost.cuh>
#include <mppi/utils/texture_helpers/two_d_texture_helper.cuh>

template <class CLASS_T, class DYN_T, class PARAMS_T>
class MapCostImpl : public Cost<CLASS_T, PARAMS_T, typename DYN_T::DYN_PARAMS_T>
{
public:
  using PARENT_CLASS = Cost<CLASS_T, PARAMS_T, typename DYN_T::DYN_PARAMS_T>;
  using output_array = typename PARENT_CLASS::output_array;
  using DYN_P = typename PARENT_CLASS::TEMPLATED_DYN_PARAMS;

  MapCostImpl(cudaStream_t stream = 0);

  ~MapCostImpl();

  std::string getCostFunctionName() const override
  {
    return std::string("Map Cost");
  }

  void bindToStream(cudaStream_t stream);

  void GPUSetup();

  void freeCudaMem();

  void paramsToDevice();

  TwoDTextureHelper<float>* getTexHelper()
  {
    return tex_helper_;
  }

  float computeStateCost(const Eigen::Ref<const output_array> s, int timestep, int* crash_status);

  float terminalCost(const Eigen::Ref<const output_array> s);

  __device__ float computeStateCost(const float* __restrict__ s, int timestep, float* __restrict__ theta_c,
                                    int* __restrict__ crash_status);

  __device__ float terminalCost(const float* __restrict__ s, float* __restrict__ theta_c);

protected:
  TwoDTextureHelper<float>* tex_helper_ = nullptr;
};

template <class DYN_T>
class MapCost : public MapCostImpl<MapCost<DYN_T>, DYN_T, CostParams<DYN_T::CONTROL_DIM>>
{
public:
  using PARENT_CLASS = MapCostImpl<MapCost, DYN_T, CostParams<DYN_T::CONTROL_DIM>>;
  using PARENT_CLASS::MapCostImpl; // Use MapCostImpl constructors
};

#ifdef __CUDACC__
#include "map_cost.cu"
#endif
